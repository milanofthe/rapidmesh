//! Boundary recovery and region tagging: TaggedPlc to a conforming tet mesh.
//!
//! The constraints are the paper model of Diazzi, Panozzo, Vaxman, Attene
//! (SIGGRAPH Asia 2023): every PLC triangle edge is a SEGMENT that must appear
//! as a union of DT edges, and every PLC triangle is a FACET that must be
//! tiled by DT faces. Boundary recovery is the constructive CDT core in
//! [`crate::cdt`]: segment recovery places exact implicit Steiner points
//! ([`Point3::Lnc`]) on the carriers, face recovery retetrahedrizes pierced
//! cavities by gift wrapping. The recovery is constructive and panics loudly
//! on divergence; it never abandons a constraint.
//!
//! Because Steiner points stay EXACTLY on their carriers (implicit until a
//! final rounding pass), the triangulation has no near-degenerate in-plane
//! slivers, so tile membership is decided purely COMBINATORIALLY: a vertex
//! carries the set of facets whose closed triangle it lies on
//! (`on_facet`, maintained incrementally), and a DT face is a tile of facet
//! F iff all three of its vertices carry F. The intersection holds at most
//! one facet for a non-degenerate face (two distinct PLC triangles share at
//! most an edge); finding two is a contradiction and panics.

use crate::cdt::{self, FacetRef, SegmentChains};
use crate::delaunay::DelaunayBuilder;
use rapidmesh_csg::classify::point_inside_solid;
use rapidmesh_csg::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind, TaggedPlc};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
/// FxHasher is unseeded (deterministic) and much faster than SipHash on the
/// short integer keys these maps use.
type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;
type DSet<T> = HashSet<T, DState>;

/// Per-vertex set of facet ids whose closed triangle the vertex lies on
/// (the combinatorial provenance that decides tile membership).
type OnFacet = DSet<u32>;

/// A conforming surface face of the tet mesh, with its PLC tags.
#[derive(Debug, Clone)]
pub struct SurfaceFace {
    /// Global vertex indices.
    pub tri: [usize; 3],
    /// Face tag inherited from the PLC patch (sheets, ports).
    pub face_tag: FaceTag,
    /// Region tags on (front, back) of the source patch.
    pub regions: [RegionTag; 2],
    /// Identity of the source patch (faces of one patch are coplanar and may
    /// be re-tiled by the optimizer).
    pub patch: u32,
    /// Analytic surface this face approximates (index into
    /// [TetMesh::surfaces]); curved kinds let the optimizer move surface
    /// vertices on the true surface.
    pub surface: u32,
}

/// A region-tagged conforming tetrahedral mesh.
#[derive(Debug)]
pub struct TetMesh {
    /// Mesh vertices (PLC vertices plus Steiner points).
    pub points: Vec<[f64; 3]>,
    /// Positively oriented tets.
    pub tets: Vec<[usize; 4]>,
    /// Region of every tet.
    pub tet_regions: Vec<RegionTag>,
    /// The mesh faces tiling the PLC patches, with tags.
    pub faces: Vec<SurfaceFace>,
    /// The analytic surfaces referenced by [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid index (scene insertion order, voids included);
    /// `u32::MAX` for sheet surfaces. Parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// Patches the recovery gave up on. The CDT recovery is constructive and
    /// never abandons a patch (it panics on divergence instead), so this is
    /// always empty; kept for API stability.
    pub abandoned_patches: Vec<u32>,
    /// `points[..plc_points]` are the PLC's own vertices (the geometry);
    /// everything after is a Steiner point the mesher added. The optimizer's
    /// edge collapse may remove Steiner points but never PLC vertices.
    pub plc_points: usize,
}

impl TetMesh {
    /// Feature (crease) edges of the final surface mesh, derived from the
    /// faces so they stay valid through optimizer rewrites. An edge is a
    /// feature edge iff it is not interior to one smooth surface group:
    /// boundary/non-manifold incidence (face count != 2), or the two faces
    /// differ in analytic surface, face tag, or region pair. Within ONE
    /// `Plane` surface entry that collects several non-coplanar walls (loft
    /// flanks, pipe segments), the planar patch id discriminates, so true
    /// geometric creases survive while the facet seams of curved analytic
    /// surfaces (cylinder barrel) stay smooth.
    pub fn feature_edges(&self) -> Vec<[usize; 2]> {
        // group key per face: planes split by patch, curved by surface
        let face_key = |sf: &SurfaceFace| -> (u32, u32, u32, u32, u32) {
            let smooth = match self.surfaces[sf.surface as usize] {
                SurfaceKind::Plane => sf.patch,
                _ => u32::MAX,
            };
            let (r0, r1) = (sf.regions[0].0.min(sf.regions[1].0), sf.regions[0].0.max(sf.regions[1].0));
            (sf.surface, smooth, sf.face_tag.0, r0, r1)
        };
        let mut edges: DMap<(usize, usize), (u32, (u32, u32, u32, u32, u32), bool)> =
            DMap::default();
        for sf in &self.faces {
            let key = face_key(sf);
            for k in 0..3 {
                let (a, b) = (sf.tri[k], sf.tri[(k + 1) % 3]);
                let e = (a.min(b), a.max(b));
                let entry = edges.entry(e).or_insert((0, key, false));
                entry.0 += 1;
                if entry.1 != key {
                    entry.2 = true;
                }
            }
        }
        let mut out: Vec<[usize; 2]> = edges
            .iter()
            .filter(|(_, &(cnt, _, mixed))| cnt != 2 || mixed)
            .map(|(&(a, b), _)| [a, b])
            .collect();
        out.sort_unstable();
        out
    }
}

/// A maximal coplanar group of equally tagged PLC facets.
struct Patch {
    /// Member facet (PLC triangle) indices.
    member_indices: Vec<usize>,
    face_tag: FaceTag,
    regions: [RegionTag; 2],
    /// Analytic surface of the members (index into the PLC surface table).
    surface: u32,
}

/// Graded size value for a new point: the regional cap, tightened by the
/// Lipschitz envelopes of its parent vertices (h_parent + grading * dist).
/// Every insertion inherits from the vertices of the simplex it refines, so
/// the per-point size field is Lipschitz by construction and sizes GROW
/// gradually away from fine features instead of jumping at interfaces.
fn child_h(
    pos: [f64; 3],
    parents: &[usize],
    points: &[[f64; 3]],
    point_h: &[f64],
    grading: f64,
    cap: f64,
    size_points: &[([f64; 3], f64)],
) -> f64 {
    let mut h = cap;
    for &v in parents {
        let d = (0..3)
            .map(|k| (pos[k] - points[v][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        h = h.min(point_h[v] + grading * d);
    }
    // Point sources act directly (the inherited field only carries them
    // outward from existing vertices; a source far from any vertex would
    // otherwise never bite).
    for (sp, sh) in size_points {
        let d = (0..3)
            .map(|k| (pos[k] - sp[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        h = h.min(sh + grading * d);
    }
    h
}

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// The four vertex-index triples spanning a tet's faces (unoriented; users
/// sort the result for keying).
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// The facet a DT face tiles, by the combinatorial provenance rule: the face
/// is a tile of facet F iff all three vertices carry F in `on_facet`. The
/// intersection holds at most one facet for a non-degenerate face (two
/// distinct PLC triangles share at most an edge, so cannot both contain three
/// common vertices); a second one is a contradiction and panics loudly.
fn facet_of_tile(on_facet: &[OnFacet], f: [usize; 3]) -> Option<u32> {
    let (s0, s1, s2) = (&on_facet[f[0]], &on_facet[f[1]], &on_facet[f[2]]);
    if s0.is_empty() || s1.is_empty() || s2.is_empty() {
        return None;
    }
    let mut found: Option<u32> = None;
    for &ff in s0 {
        if s1.contains(&ff) && s2.contains(&ff) {
            assert!(
                found.is_none(),
                "tet face {f:?} lies on more than one facet (combinatorial tile rule violated)",
            );
            found = Some(ff);
        }
    }
    found
}

/// Builds the maximal coplanar same-tag patches by union-find over facets
/// sharing an edge.
fn build_patches(plc: &TaggedPlc) -> Vec<Patch> {
    let n = plc.triangles.len();
    let tri = |i: usize| -> [usize; 3] {
        let t = plc.triangles[i];
        [t[0] as usize, t[1] as usize, t[2] as usize]
    };
    let pt = |v: usize| Point3::Explicit(plc.vertices[v]);
    let coplanar = |a: usize, b: usize| -> bool {
        let pa = tri(a);
        tri(b).iter().all(|&v| {
            orient3d(&pt(pa[0]), &pt(pa[1]), &pt(pa[2]), &pt(v)) == Some(Sign::Zero)
        })
    };

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            let r = find(parent, parent[i]);
            parent[i] = r;
            r
        } else {
            i
        }
    }
    let mut by_edge: DMap<(usize, usize), Vec<usize>> = DMap::default();
    for i in 0..n {
        let t = tri(i);
        for e in 0..3 {
            by_edge.entry(sorted2(t[e], t[(e + 1) % 3])).or_default().push(i);
        }
    }
    for owners in by_edge.values() {
        for w in owners.windows(2) {
            let (a, b) = (w[0], w[1]);
            if plc.face_tags[a] == plc.face_tags[b]
                && plc.region_tags[a] == plc.region_tags[b]
                && coplanar(a, b)
            {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
        }
    }

    let mut groups: DMap<usize, Vec<usize>> = DMap::default();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    // Deterministic patch order (map iteration order must not shape the mesh).
    let mut group_list: Vec<Vec<usize>> = groups.into_values().collect();
    group_list.sort_by_key(|m| m.iter().copied().min());
    group_list
        .into_iter()
        .map(|members| Patch {
            face_tag: plc.face_tags[members[0]],
            regions: plc.region_tags[members[0]],
            surface: plc.surface_refs[members[0]].0,
            member_indices: members,
        })
        .collect()
}

/// Sizing and quality parameters for [`mesh_plc_with`].
#[derive(Debug, Clone)]
pub struct MeshParams {
    /// Target maximum edge length (creases, patch tiles, and tet edges).
    pub maxh: f64,
    /// Per-region target edge length, overriding maxh inside that region
    /// (Maxwell FEM sizes regions by local wavelength, h ~ lambda/sqrt(eps)).
    /// Interfaces and creases follow the finer adjacent region; transitions
    /// into coarser regions grade naturally through Delaunay refinement.
    pub region_maxh: Vec<(u32, f64)>,
    /// Delaunay-refinement quality bound: tets with
    /// circumradius / shortest-edge above this get their circumcenter
    /// inserted. The provable refinement regime is >= 2.0.
    pub radius_edge_bound: f64,
    /// Refinement stops (best effort) once this many points exist.
    pub max_points: usize,
    /// Size grading: the target edge length may grow by at most this factor
    /// per unit distance from finer features (h(x) is Lipschitz with this
    /// constant). 0.5 grows neighbor elements by roughly 1.5x; INFINITY
    /// disables grading (sizes jump at region interfaces).
    pub grading: f64,
    /// Per-face-tag target edge length, overriding the adjacent regions'
    /// targets on those patches (rapidfem's per-plate maxh).
    pub face_maxh: Vec<(u32, f64)>,
    /// Per-solid SURFACE target edge length, keyed by the owner solid index
    /// in [TaggedPlc::surface_owners] (scene insertion order, voids
    /// included): refines the solid's boundary patches and grades into the
    /// surrounding volume. The only sizing handle that reaches a void's
    /// walls (a coax inner conductor has no region and no face tag).
    pub surface_maxh: Vec<(u32, f64)>,
    /// Point size sources `(position, h)`: the target shrinks to `h` at the
    /// point and recovers along the Lipschitz grading away from it
    /// (rapidfem's refine_near_points; the hook for error-driven adaptive
    /// refinement).
    pub size_points: Vec<([f64; 3], f64)>,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            radius_edge_bound: 2.0,
            max_points: 100_000,
            grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
        }
    }
}

/// Circumcenter and circumradius of a tet, `None` if degenerate.
fn tet_circumcenter(p: [[f64; 3]; 4]) -> Option<([f64; 3], f64)> {
    // Rows 2(p_i - p_0), rhs |p_i|^2 - |p_0|^2.
    let row = |i: usize| -> [f64; 3] { std::array::from_fn(|k| 2.0 * (p[i][k] - p[0][k])) };
    let sq = |q: [f64; 3]| -> f64 { q.iter().map(|x| x * x).sum() };
    let (r1, r2, r3) = (row(1), row(2), row(3));
    let b = [sq(p[1]) - sq(p[0]), sq(p[2]) - sq(p[0]), sq(p[3]) - sq(p[0])];
    let det3 = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> f64 {
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    let d = det3(r1, r2, r3);
    let scale: f64 = [r1, r2, r3]
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    if d.abs() < 1e-12 * scale.powi(3) {
        return None;
    }
    let col = |j: usize| -> f64 {
        let mut m = [r1, r2, r3];
        for (i, row) in m.iter_mut().enumerate() {
            row[j] = b[i];
        }
        det3(m[0], m[1], m[2]) / d
    };
    let c = [col(0), col(1), col(2)];
    let r = (0..3).map(|k| (c[k] - p[0][k]).powi(2)).sum::<f64>().sqrt();
    Some((c, r))
}

/// In-plane circumcenter and circumradius of a 3D triangle, `None` if
/// degenerate.
fn tri_circumcenter(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<([f64; 3], f64)> {
    let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
    let v: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
    let dot = |x: [f64; 3], y: [f64; 3]| -> f64 { (0..3).map(|k| x[k] * y[k]).sum() };
    let (uu, uv, vv) = (dot(u, u), dot(u, v), dot(v, v));
    let det = uu * vv - uv * uv;
    if det.abs() < 1e-12 * uu * vv + f64::MIN_POSITIVE {
        return None;
    }
    let alpha = 0.5 * (uu * vv - uv * vv) / det;
    let beta = 0.5 * (uu * vv - uv * uu) / det;
    let cc: [f64; 3] = std::array::from_fn(|k| a[k] + alpha * u[k] + beta * v[k]);
    let r = (0..3).map(|k| (cc[k] - a[k]).powi(2)).sum::<f64>().sqrt();
    Some((cc, r))
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh without
/// sizing or quality refinement. Background (region 0) tets are dropped.
pub fn mesh_plc(plc: &TaggedPlc) -> TetMesh {
    mesh_plc_with(
        plc,
        &MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            radius_edge_bound: f64::INFINITY,
            max_points: usize::MAX,
            grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
        },
    )
}

/// Extends the per-vertex bookkeeping to cover the builder's current vertex
/// count (new entries are the segment Steiner points the last recovery pass
/// created) and re-applies the chain-provenance sweep: every vertex on the
/// chain of segment `s` carries that segment's incident facets, and each new
/// Steiner point inherits a graded size from its segment's original endpoints.
#[allow(clippy::too_many_arguments)]
fn sync_chains(
    builder: &DelaunayBuilder,
    chains: &SegmentChains,
    segments: &[(usize, usize)],
    seg_facets: &[Vec<u32>],
    grading: f64,
    size_points: &[([f64; 3], f64)],
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    on_facet: &mut Vec<OnFacet>,
    point_index: &mut DMap<[u64; 3], usize>,
) {
    let first_new = points.len();
    while points.len() < builder.len() {
        let v = points.len();
        let pos = builder.approx_point(v);
        points.push(pos);
        on_facet.push(OnFacet::default());
        point_h.push(f64::INFINITY);
        point_index.entry(pos.map(|x| (x + 0.0).to_bits())).or_insert(v);
    }
    for s in 0..chains.segment_count() {
        let (ea, eb) = segments[s];
        for v in chains.chain(s) {
            for &f in &seg_facets[s] {
                on_facet[v].insert(f);
            }
            if v >= first_new {
                let h = child_h(points[v], &[ea, eb], points, point_h, grading, f64::INFINITY, size_points);
                point_h[v] = h;
            }
        }
    }
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh, refined to
/// the given sizing and quality targets (best effort under
/// `params.max_points`).
pub fn mesh_plc_with(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    let trace = std::env::var_os("RAPIDMESH_TRACE").is_some();
    let mut points: Vec<[f64; 3]> = plc.vertices.clone();
    let patches = build_patches(plc);

    // Facet -> patch (PLC triangle index -> patch index).
    let mut facet_patch: Vec<usize> = vec![0; plc.triangles.len()];
    for (pi, p) in patches.iter().enumerate() {
        for &ti in &p.member_indices {
            facet_patch[ti] = pi;
        }
    }

    // Segments (unique PLC triangle edges), facets (every PLC triangle), and
    // the per-segment incident-facet lists.
    let mut seg_ids: DMap<(usize, usize), usize> = DMap::default();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_facets: Vec<Vec<u32>> = Vec::new();
    let mut facets: Vec<FacetRef> = Vec::new();
    // Provenance of PLC corner vertices: every incident facet.
    let mut on_facet: Vec<OnFacet> = vec![OnFacet::default(); points.len()];
    for (ti, t) in plc.triangles.iter().enumerate() {
        let corners = [t[0] as usize, t[1] as usize, t[2] as usize];
        let edges: [usize; 3] = std::array::from_fn(|e| {
            let key = sorted2(corners[e], corners[(e + 1) % 3]);
            let id = *seg_ids.entry(key).or_insert_with(|| {
                segments.push(key);
                seg_facets.push(Vec::new());
                segments.len() - 1
            });
            seg_facets[id].push(ti as u32);
            id
        });
        for &v in &corners {
            on_facet[v].insert(ti as u32);
        }
        facets.push(FacetRef { corners, edges });
    }
    let acute = cdt::acute_vertices(&plc.vertices, &segments);
    // Feature segments are the geometric creases (patch boundaries, non-
    // manifold or sheet-rim edges); a segment interior to a single patch is
    // just a triangulation diagonal. Both are RECOVERED as DT edges, but only
    // features drive eager sizing splits: a diagonal is refined on demand
    // (when a surface tile's circumcenter encroaches it), matching the patch
    // model and keeping smooth coplanar faces from over-refining.
    let seg_is_feature: Vec<bool> = (0..segments.len())
        .map(|s| {
            !(seg_facets[s].len() == 2
                && facet_patch[seg_facets[s][0] as usize] == facet_patch[seg_facets[s][1] as usize])
        })
        .collect();

    // Initial Delaunay of the PLC vertices.
    let mut blo = [f64::MAX; 3];
    let mut bhi = [f64::MIN; 3];
    for p in &points {
        for k in 0..3 {
            blo[k] = blo[k].min(p[k]);
            bhi[k] = bhi[k].max(p[k]);
        }
    }
    let mut builder = DelaunayBuilder::enclosing(blo, bhi);
    let mut point_index: DMap<[u64; 3], usize> = DMap::default();
    for (i, &p) in points.iter().enumerate() {
        builder.insert(p);
        point_index.insert(p.map(|x| (x + 0.0).to_bits()), i);
    }

    // Per-point graded size targets (see child_h): PLC vertices start from
    // the finest adjacent patch target.
    let h_of_region_init = |r: u32| -> f64 {
        if r == 0 {
            return f64::INFINITY;
        }
        params
            .region_maxh
            .iter()
            .find(|(rr, _)| *rr == r)
            .map(|&(_, h)| h)
            .unwrap_or(params.maxh)
    };
    let h_of_face = |tag: FaceTag| -> f64 {
        params
            .face_maxh
            .iter()
            .find(|(t, _)| *t == tag.0)
            .map(|&(_, h)| h)
            .unwrap_or(f64::INFINITY)
    };
    let h_of_surface = |surface: u32| -> f64 {
        let owner = plc
            .surface_owners
            .get(surface as usize)
            .copied()
            .unwrap_or(u32::MAX);
        params
            .surface_maxh
            .iter()
            .find(|(s, _)| *s == owner)
            .map(|&(_, h)| h)
            .unwrap_or(f64::INFINITY)
    };
    let patch_h_init: Vec<f64> = patches
        .iter()
        .map(|p| {
            h_of_region_init(p.regions[0].0)
                .min(h_of_region_init(p.regions[1].0))
                .min(h_of_face(p.face_tag))
                .min(h_of_surface(p.surface))
        })
        .collect();
    let mut point_h: Vec<f64> = (0..points.len())
        .map(|v| {
            let mut h = on_facet[v]
                .iter()
                .map(|&f| patch_h_init[facet_patch[f as usize]])
                .fold(params.maxh, f64::min);
            for (sp, sh) in &params.size_points {
                let d = (0..3)
                    .map(|k| (points[v][k] - sp[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                h = h.min(sh + params.grading * d);
            }
            h
        })
        .collect();
    let point_h = &mut point_h;

    // ------------------------------------------------- recovery + refinement
    // RAPIDMESH_TIMING: coarse wall-clock split of the meshing phases
    // (segment recovery, the per-round face recovery / tile derivation /
    // refinement, implicit rounding), printed once at the end.
    let timing = std::env::var_os("RAPIDMESH_TIMING").is_some();
    let mut t_faces = std::time::Duration::ZERO;
    let mut t_tiles = std::time::Duration::ZERO;
    let mut t_classify = std::time::Duration::ZERO;
    let mut t_refine = std::time::Duration::ZERO;
    let t0 = std::time::Instant::now();
    let mut chains = cdt::recover_segments(&mut builder, &segments, &acute);
    sync_chains(
        &builder, &chains, &segments, &seg_facets, params.grading, &params.size_points,
        &mut points, point_h, &mut on_facet, &mut point_index,
    );
    let t_segments = t0.elapsed();

    let mut tried: DMap<[usize; 4], u8> = DMap::default();
    let mut round = 0usize;
    let mut refine_round = 0usize;
    // Per facet: creation-log position at its last clean recovery sweep
    // (0 = never swept; see recover_one_facet).
    let mut facet_clean: Vec<usize> = vec![0; facets.len()];
    #[allow(clippy::type_complexity)]
    let (tets, patch_faces, tet_region): (Vec<[usize; 4]>, Vec<Vec<[usize; 3]>>, Vec<u32>) = 'outer: loop {
        round += 1;
        assert!(round <= 16384, "boundary recovery did not converge");

        // CDT face recovery until clean: one facet's cavity surgery can unmake
        // another facet or a chain edge, so alternate face and segment
        // recovery until a face pass finds nothing.
        let tf = std::time::Instant::now();
        loop {
            let any = cdt::recover_faces(&mut builder, &facets, &chains, &mut facet_clean);
            cdt::resume_segments(&mut builder, &mut chains);
            sync_chains(
                &builder, &chains, &segments, &seg_facets, params.grading, &params.size_points,
                &mut points, point_h, &mut on_facet, &mut point_index,
            );
            if !any {
                break;
            }
        }
        t_faces += tf.elapsed();

        // Derive tiles (combinatorial provenance), face owners, and tets for
        // this round.
        let tt = std::time::Instant::now();
        let slot_tets = builder.tets_with_slots();
        let dt_tets: Vec<[usize; 4]> = slot_tets.iter().map(|&(_, t)| t).collect();
        let mut all_tilings: Vec<Vec<[usize; 3]>> = vec![Vec::new(); patches.len()];
        let mut tile_facet: DMap<[usize; 3], u32> = DMap::default();
        let mut dt_faces: DMap<[usize; 3], Vec<u32>> = DMap::default();
        for (ti, t) in dt_tets.iter().enumerate() {
            for fv in &TET_FACES {
                let key = sorted3(fv.map(|k| t[k]));
                let owners = dt_faces.entry(key).or_default();
                let first = owners.is_empty();
                owners.push(ti as u32);
                if first {
                    if let Some(facet) = facet_of_tile(&on_facet, key) {
                        tile_facet.insert(key, facet);
                        all_tilings[facet_patch[facet as usize]].push(key);
                    }
                }
            }
        }

        t_tiles += tt.elapsed();

        let tc = std::time::Instant::now();
        let tet_region =
            classify_tet_regions(&points, &dt_tets, &patches, &all_tilings, &dt_faces, (blo, bhi));
        t_classify += tc.elapsed();

        if points.len() >= params.max_points {
            if trace {
                eprintln!("refinement stopped at point budget ({})", points.len());
            }
            break 'outer (dt_tets, all_tilings, tet_region);
        }
        refine_round += 1;
        if refine_round > 1000 {
            if trace {
                eprintln!("refinement stopped at round budget");
            }
            break 'outer (dt_tets, all_tilings, tet_region);
        }

        // Deterministic tile list (face, facet) for refinement.
        let mut tile_list: Vec<([usize; 3], u32)> =
            tile_facet.iter().map(|(&f, &p)| (f, p)).collect();
        tile_list.sort_unstable();
        // Live chain pieces (sorted vertex pair -> segment id).
        let mut live_pieces: DMap<(usize, usize), usize> = DMap::default();
        for s in 0..chains.segment_count() {
            for w in chains.chain(s).windows(2) {
                live_pieces.insert(sorted2(w[0], w[1]), s);
            }
        }

        let tr = std::time::Instant::now();
        let inserted = refine_queue(
            params,
            &plc.surface_owners,
            &patches,
            &facets,
            &facet_patch,
            &segments,
            &seg_facets,
            &seg_is_feature,
            &slot_tets,
            &tile_list,
            &tet_region,
            &mut tried,
            &mut points,
            point_h,
            &mut builder,
            &mut point_index,
            &mut on_facet,
            &mut chains,
            &mut live_pieces,
        );
        t_refine += tr.elapsed();
        if trace && inserted > 0 {
            eprintln!("round {round}: inserted {inserted}, {} points", points.len());
        }
        if inserted == 0 {
            break 'outer (dt_tets, all_tilings, tet_region);
        }
    };

    // Round the implicit Steiner points to plain f64 and refresh the cached
    // positions (rounding may nudge implicit points along their carriers).
    let tro = std::time::Instant::now();
    builder.round_implicit_points();
    for (i, p) in points.iter_mut().enumerate() {
        *p = builder.approx_point(i);
    }
    if timing {
        eprintln!(
            "timing: segments {:?}, faces {:?}, tiles {:?}, classify {:?}, refine {:?}, rounding {:?} ({} rounds, {} points)",
            t_segments, t_faces, t_tiles, t_classify, t_refine, tro.elapsed(), round, points.len(),
        );
    }

    if std::env::var_os("RAPIDMESH_EDGE_DUMP").is_some() {
        let lim = params.maxh * 1.45;
        let mut seen: DSet<(usize, usize)> = DSet::default();
        for t in &tets {
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = sorted2(t[i], t[j]);
                    let d = (0..3)
                        .map(|k| (points[a][k] - points[b][k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if d > lim && seen.insert((a, b)) {
                        eprintln!(
                            "long edge {d:.4}: v{a} h={:.3} {:?} -> v{b} h={:.3} {:?}",
                            point_h[a], points[a], point_h[b], points[b]
                        );
                    }
                }
            }
        }
    }

    // ----------------------------------------------------- output
    let mut kept_tets: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for (ti, t) in tets.iter().enumerate() {
        if tet_region[ti] != 0 {
            kept_tets.push(*t);
            tet_regions.push(RegionTag(tet_region[ti]));
        }
    }

    let mut out_faces: Vec<SurfaceFace> = Vec::new();
    for (pi, patch) in patches.iter().enumerate() {
        for f in &patch_faces[pi] {
            out_faces.push(SurfaceFace {
                tri: *f,
                face_tag: patch.face_tag,
                regions: patch.regions,
                patch: pi as u32,
                surface: patch.surface,
            });
        }
    }

    TetMesh {
        points,
        tets: kept_tets,
        tet_regions,
        faces: out_faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points: plc.vertices.len(),
    }
}

/// Region of every tet by FLOOD FILL through shared faces: crossing a
/// constraint face flips to the face's other region, free faces keep it.
/// Parity ray casting runs once per CONNECTED COMPONENT (per-tet casting is
/// O(tets * boundary faces) and dominated large meshes).
fn classify_tet_regions(
    points: &[[f64; 3]],
    tets: &[[usize; 4]],
    patches: &[Patch],
    tilings: &[Vec<[usize; 3]>],
    face_owners: &DMap<[usize; 3], Vec<u32>>,
    bbox: ([f64; 3], [f64; 3]),
) -> Vec<u32> {
    let mut face_regions: DMap<[usize; 3], [u32; 2]> = DMap::default();
    let mut region_bounds: DMap<u32, Vec<Tri>> = DMap::default();
    for (pi, patch) in patches.iter().enumerate() {
        for f in &tilings[pi] {
            face_regions.insert(sorted3(*f), [patch.regions[0].0, patch.regions[1].0]);
            if patch.regions[0] != patch.regions[1] {
                let t = Tri::new(points[f[0]], points[f[1]], points[f[2]]);
                for tag in patch.regions {
                    if tag.0 != 0 {
                        region_bounds.entry(tag.0).or_default().push(t);
                    }
                }
            }
        }
    }
    let mut region_ids: Vec<u32> = region_bounds.keys().copied().collect();
    region_ids.sort_unstable();

    let mut region_of: Vec<Option<u32>> = vec![None; tets.len()];
    let mut stack: Vec<usize> = Vec::new();
    for seed in 0..tets.len() {
        if region_of[seed].is_some() {
            continue;
        }
        // Parity-classify the component root by its centroid.
        let t = tets[seed];
        let c: [f64; 3] = std::array::from_fn(|k| {
            0.25 * (points[t[0]][k] + points[t[1]][k] + points[t[2]][k] + points[t[3]][k])
        });
        let rep = Point3::Explicit(c);
        let seed_region = region_ids
            .iter()
            .copied()
            .find(|r| point_inside_solid(&rep, &region_bounds[r], bbox))
            .unwrap_or(0);
        region_of[seed] = Some(seed_region);
        stack.push(seed);
        while let Some(ti) = stack.pop() {
            let cur = region_of[ti].expect("set before push");
            let t = tets[ti];
            for fi in TET_FACES {
                let key = sorted3(fi.map(|k| t[k]));
                let next_region = match face_regions.get(&key) {
                    // Crossing a constraint face flips to its other side
                    // (embedded sheets have equal sides: no change).
                    Some(&[a, b]) => {
                        if a == cur {
                            b
                        } else if b == cur {
                            a
                        } else {
                            // Inconsistent: leave for the neighbor's own
                            // component seed.
                            continue;
                        }
                    }
                    None => cur,
                };
                for &nb in &face_owners[&key] {
                    let nb = nb as usize;
                    if nb != ti && region_of[nb].is_none() {
                        region_of[nb] = Some(next_region);
                        stack.push(nb);
                    }
                }
            }
        }
    }
    region_of.into_iter().map(|r| r.unwrap_or(0)).collect()
}

/// A vetoed quality candidate is re-attempted at most this many times after
/// neighborhood changes (its verdict rarely changes beyond that, and
/// unbounded retries re-pay the insertion cavity every round).
const QUALITY_RETRY_LIMIT: u8 = 3;

/// Sizing splits trigger above this multiple of the local target h. Splits
/// roughly halve lengths, so triggering exactly at h lands edges at h/2 and
/// over-refines about twofold against meshers that treat h as a target.
/// Calibrated on the bare WR-90 box (examples/measure_density.rs): 1.3 gave
/// a 0.87 h mean edge (~50% extra tets); 1.45 centers the mean at 0.97 h,
/// cuts tets by ~27% and IMPROVES quality (min dihedral 20 -> 26 deg, the
/// optimizer breathes with fewer crowded vertices). The max edge is pinned
/// by the optimizer's 1.5 h contract independently of this trigger.
const OVERSIZE_FACTOR: f64 = 1.45;

/// Uniform grid over balls for "which ball contains this point" queries:
/// linear scans over all crease/tile balls per refinement candidate are
/// quadratic on surface models with tens of thousands of constraints. Balls
/// spanning more than [`BallGrid::MAX_SPAN`] cells per axis go into a small
/// linearly-checked overflow list.
struct BallGrid {
    cell: f64,
    origin: [f64; 3],
    map: DMap<[i64; 3], Vec<u32>>,
    large: Vec<u32>,
    balls: Vec<([f64; 3], f64)>,
}

impl BallGrid {
    const MAX_SPAN: i64 = 4;

    fn build(balls: Vec<([f64; 3], f64)>) -> BallGrid {
        let mut radii: Vec<f64> = balls.iter().map(|b| b.1).collect();
        radii.sort_by(f64::total_cmp);
        let median_r = radii.get(radii.len() / 2).copied().unwrap_or(1.0);
        let cell = (2.0 * median_r).max(f64::MIN_POSITIVE);
        let origin = balls.first().map(|b| b.0).unwrap_or([0.0; 3]);
        let mut grid = BallGrid {
            cell,
            origin,
            map: DMap::default(),
            large: Vec::new(),
            balls,
        };
        for bi in 0..grid.balls.len() {
            let (c, r) = grid.balls[bi];
            let lo = grid.cell_of(std::array::from_fn(|k| c[k] - r));
            let hi = grid.cell_of(std::array::from_fn(|k| c[k] + r));
            if (0..3).any(|k| hi[k] - lo[k] >= BallGrid::MAX_SPAN) {
                grid.large.push(bi as u32);
                continue;
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        grid.map.entry([x, y, z]).or_default().push(bi as u32);
                    }
                }
            }
        }
        grid
    }

    fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|k| ((p[k] - self.origin[k]) / self.cell).floor() as i64)
    }

    /// Index of the first ball (in insertion order) containing `x` and
    /// accepted by `live`, if any. Matches the linear-scan semantics: the
    /// FIRST hit in ball order wins.
    fn first_containing(&self, x: [f64; 3], live: impl Fn(usize) -> bool) -> Option<usize> {
        let dist2 = |a: [f64; 3], b: [f64; 3]| -> f64 {
            (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
        };
        let mut best: Option<usize> = None;
        let mut consider = |bi: u32| {
            let (c, r) = self.balls[bi as usize];
            if dist2(x, c) < r * r
                && best.is_none_or(|b| (bi as usize) < b)
                && live(bi as usize)
            {
                best = Some(bi as usize);
            }
        };
        if let Some(v) = self.map.get(&self.cell_of(x)) {
            for &bi in v {
                consider(bi);
            }
        }
        for &bi in &self.large {
            consider(bi);
        }
        best
    }
}

fn ins_trace(path: &str, g: usize, pos: [f64; 3]) {
    if std::env::var_os("RAPIDMESH_INS_TRACE").is_some() {
        eprintln!("ins {path} v{g} {pos:?}");
    }
}

/// Queue-driven sizing/quality refinement on a conforming state; returns
/// the number of insertions (0 = all targets met). Shewchuk priority:
/// oversized segment pieces, then oversized patch tiles, then oversized or
/// poor-quality tets (circumcenters, with encroachment redirected to the
/// boundary).
///
/// Tiles are protected combinatorially (the keep set holds the live tile
/// faces and chain pieces); surface refinement inserts EXACT on-facet
/// Steiner points ([`Point3::Pac`]) and on-carrier piece midpoints
/// ([`Point3::Lnc`]), so the conformity invariant survives every split.
/// Splits done here may still knock a constraint out of the DT (a piece
/// midpoint is unguarded); the caller's next recovery pass re-establishes it.
#[allow(clippy::too_many_arguments)]
fn refine_queue(
    params: &MeshParams,
    surface_owners: &[u32],
    patches: &[Patch],
    facets: &[FacetRef],
    facet_patch: &[usize],
    segments: &[(usize, usize)],
    seg_facets: &[Vec<u32>],
    seg_is_feature: &[bool],
    slot_tets: &[(u32, [usize; 4])],
    tile_list: &[([usize; 3], u32)],
    tet_region: &[u32],
    tried: &mut DMap<[usize; 4], u8>,
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    builder: &mut DelaunayBuilder,
    point_index: &mut DMap<[u64; 3], usize>,
    on_facet: &mut Vec<OnFacet>,
    chains: &mut SegmentChains,
    live_pieces: &mut DMap<(usize, usize), usize>,
) -> usize {
    let sized = params.maxh.is_finite()
        || !params.region_maxh.is_empty()
        || !params.face_maxh.is_empty()
        || !params.surface_maxh.is_empty()
        || !params.size_points.is_empty();
    if !sized && params.radius_edge_bound.is_infinite() {
        return 0;
    }
    // Per-region targets: region 0 (background) is unconstrained; interfaces
    // and creases follow their finest adjacent region.
    let h_of_region = |r: u32| -> f64 {
        if r == 0 {
            return f64::INFINITY;
        }
        params
            .region_maxh
            .iter()
            .find(|(rr, _)| *rr == r)
            .map(|&(_, h)| h)
            .unwrap_or(params.maxh)
    };
    let h_of_face = |tag: FaceTag| -> f64 {
        params
            .face_maxh
            .iter()
            .find(|(t, _)| *t == tag.0)
            .map(|&(_, h)| h)
            .unwrap_or(f64::INFINITY)
    };
    let h_of_surface = |surface: u32| -> f64 {
        let owner = surface_owners
            .get(surface as usize)
            .copied()
            .unwrap_or(u32::MAX);
        params
            .surface_maxh
            .iter()
            .find(|(s, _)| *s == owner)
            .map(|&(_, h)| h)
            .unwrap_or(f64::INFINITY)
    };
    let patch_h: Vec<f64> = patches
        .iter()
        .map(|p| {
            h_of_region(p.regions[0].0)
                .min(h_of_region(p.regions[1].0))
                .min(h_of_face(p.face_tag))
                .min(h_of_surface(p.surface))
        })
        .collect();
    // Finest target along a segment: min over the patches of its facets.
    let seg_h = |s: usize| -> f64 {
        seg_facets[s]
            .iter()
            .map(|&f| patch_h[facet_patch[f as usize]])
            .fold(f64::INFINITY, f64::min)
    };
    let dist2 = |a: [f64; 3], b: [f64; 3]| -> f64 {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
    };

    // ------------------------------------ incremental bookkeeping
    // Region per builder slot, seeded from the caller's flood fill (which
    // covers every alive all-real tet; super-touching slots stay 0 = the
    // outside, which is exactly their region).
    let mut region_by_slot: Vec<u32> = vec![0; builder.slot_count()];
    for (&(slot, _), &r) in slot_tets.iter().zip(tet_region) {
        region_by_slot[slot as usize] = r;
    }
    // The live tile set (face -> facet), seeded from the caller's tiling
    // derivation and the keep-set of every guarded insert.
    let mut tile_map: DMap<[usize; 3], u32> = tile_list.iter().copied().collect();

    // Work queues, seeded in deterministic order; entries re-verify on pop
    // (slots are reused, tiles retile, pieces split).
    let mut piece_snapshot: Vec<(usize, usize)> = live_pieces.keys().copied().collect();
    piece_snapshot.sort_unstable();
    let mut crease_q: VecDeque<(usize, usize)> = piece_snapshot.iter().copied().collect();
    let mut tile_q: VecDeque<[usize; 3]> = tile_list.iter().map(|&(f, _)| f).collect();
    let mut tet_q: VecDeque<(u32, [usize; 4])> = slot_tets.iter().copied().collect();

    // Encroachment helper: the chain piece whose diametral ball contains x.
    // A snapshot grid over the entry pieces plus a linearly scanned overlay
    // of pieces split DURING the call, both live-checked against live_pieces.
    let crease_grid = BallGrid::build(
        piece_snapshot
            .iter()
            .map(|&(a, b)| {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                (m, 0.5 * dist2(points[a], points[b]).sqrt())
            })
            .collect(),
    );
    let encroached_crease = |x: [f64; 3],
                             points: &[[f64; 3]],
                             extra: &[(usize, usize)],
                             live: &DMap<(usize, usize), usize>|
     -> Option<(usize, usize)> {
        if let Some(i) =
            crease_grid.first_containing(x, |i| live.contains_key(&piece_snapshot[i]))
        {
            return Some(piece_snapshot[i]);
        }
        extra.iter().copied().find(|&(a, b)| {
            live.contains_key(&(a, b)) && {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                dist2(x, m) < 0.25 * dist2(points[a], points[b])
            }
        })
    };

    // Tile circumballs for circumcenter redirection: a snapshot grid plus a
    // linearly scanned overlay of tiles created during the call; both are
    // live-checked against tile_map.
    let mut tile_balls: Vec<([f64; 3], u32, [usize; 3])> = Vec::new();
    let mut tile_ball_geo: Vec<([f64; 3], f64)> = Vec::new();
    for &(f, facet) in tile_list {
        if let Some((tc, tr)) = tri_circumcenter(points[f[0]], points[f[1]], points[f[2]]) {
            tile_balls.push((tc, facet, f));
            tile_ball_geo.push((tc, tr));
        }
    }
    let tile_grid = BallGrid::build(tile_ball_geo);
    let mut tile_overlay: Vec<([f64; 3], f64, u32, [usize; 3])> = Vec::new();
    let encroached_tile = |x: [f64; 3],
                           tile_map: &DMap<[usize; 3], u32>,
                           overlay: &[([f64; 3], f64, u32, [usize; 3])]|
     -> Option<(u32, [f64; 3], [usize; 3])> {
        if let Some(i) = tile_grid.first_containing(x, |i| {
            let (_, facet, f) = tile_balls[i];
            tile_map.get(&sorted3(f)) == Some(&facet)
        }) {
            let (tc, facet, f) = tile_balls[i];
            return Some((facet, tc, f));
        }
        overlay
            .iter()
            .find(|(tc, tr, facet, f)| {
                dist2(x, *tc) < tr * tr && tile_map.get(&sorted3(*f)) == Some(facet)
            })
            .map(|&(tc, _, facet, f)| (facet, tc, f))
    };

    let cand_trace = std::env::var_os("RAPIDMESH_CAND_TRACE").is_some();
    let mut n = 0usize;
    let mut guarded_ok = 0usize;
    let mut guarded_veto = 0usize;
    let mut n_crease_size = 0usize;
    let mut n_crease_redirect = 0usize;
    let mut n_tile = 0usize;
    let mut n_midpoint = 0usize;
    let mut n_cc_quality = 0usize;
    let mut n_cc_oversized = 0usize;
    // Pieces split during the call (encroachment overlay tail).
    let mut extra_pieces: Vec<(usize, usize)> = Vec::new();

    // Applies one successful insert's cavity deltas to the bookkeeping.
    macro_rules! absorb {
        ($p:expr) => {
            absorb_insert_deltas(
                builder,
                $p,
                points,
                on_facet,
                &mut region_by_slot,
                &mut tile_map,
                &mut tet_q,
                &mut tile_q,
                &mut tile_overlay,
            )
        };
    }
    // Splits a live chain piece at its carrier midpoint (exact Lnc point),
    // updating the chain, bookkeeping, and encroachment overlay. Returns
    // whether a split happened.
    macro_rules! split_piece {
        ($a:expr, $b:expr) => {{
            let key = sorted2($a, $b);
            match live_pieces.get(&key).copied() {
                None => false,
                Some(seg) => match chains.split_piece_mid(builder, seg, $a, $b) {
                    None => false,
                    Some(g) => {
                        debug_assert_eq!(g, points.len());
                        let pos = builder.approx_point(g);
                        points.push(pos);
                        let (ea, eb) = segments[seg];
                        point_h.push(child_h(
                            pos, &[ea, eb], points, point_h, params.grading, f64::INFINITY,
                            &params.size_points,
                        ));
                        point_index.entry(pos.map(|x| (x + 0.0).to_bits())).or_insert(g);
                        let mut fs = OnFacet::default();
                        for &f in &seg_facets[seg] {
                            fs.insert(f);
                        }
                        on_facet.push(fs);
                        live_pieces.remove(&key);
                        let (na, nb) = (sorted2($a, g), sorted2(g, $b));
                        live_pieces.insert(na, seg);
                        live_pieces.insert(nb, seg);
                        extra_pieces.push(na);
                        extra_pieces.push(nb);
                        crease_q.push_back(na);
                        crease_q.push_back(nb);
                        absorb!(g);
                        true
                    }
                },
            }
        }};
    }
    // Inserts an exact on-facet Steiner point (Pac) at the projection of
    // `target` onto facet `facet`, clamped strictly inside the facet
    // triangle. The keep closure protects every OTHER facet's tiles and all
    // live pieces, but lets this facet's own tiles be re-coned (that is the
    // point of a surface refinement). Increments `n` on success.
    macro_rules! insert_facet_point {
        ($facet:expr, $target:expr, $parents:expr, $cap:expr) => {{
            let facet = $facet;
            let cor = facets[facet as usize].corners;
            let aa = points[cor[0]];
            let bb = points[cor[1]];
            let cc3 = points[cor[2]];
            let e1: [f64; 3] = std::array::from_fn(|k| bb[k] - aa[k]);
            let e2: [f64; 3] = std::array::from_fn(|k| cc3[k] - aa[k]);
            let nrm = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let drop = (0..3)
                .max_by(|&i, &j| nrm[i].abs().partial_cmp(&nrm[j].abs()).unwrap())
                .unwrap();
            let (ia, ja) = match drop {
                0 => (1, 2),
                1 => (2, 0),
                _ => (0, 1),
            };
            let det = e1[ia] * e2[ja] - e1[ja] * e2[ia];
            let target = $target;
            if det != 0.0 {
                let w0 = target[ia] - aa[ia];
                let w1 = target[ja] - aa[ja];
                let mut u = (w0 * e2[ja] - w1 * e2[ia]) / det;
                let mut v = (e1[ia] * w1 - e1[ja] * w0) / det;
                let (lo, hi) = (0.02_f64, 0.98_f64);
                u = u.clamp(lo, hi);
                v = v.clamp(lo, hi);
                if u + v > hi {
                    let s = hi / (u + v);
                    u *= s;
                    v *= s;
                }
                let cap: f64 = $cap;
                let md2 = if cap.is_finite() { (0.2 * cap) * (0.2 * cap) } else { 0.0 };
                let admitted = builder.insert_exact_guarded(
                    Point3::pac(aa, bb, cc3, u, v),
                    md2,
                    |rem| match rem {
                        crate::delaunay::Removal::Face(f) => {
                            tile_map.get(&f).is_none_or(|&ff| ff == facet)
                        }
                        crate::delaunay::Removal::Edge(a, b) => {
                            !live_pieces.contains_key(&(a, b))
                        }
                    },
                );
                if let Some(g) = admitted {
                    debug_assert_eq!(g, points.len());
                    let pos = builder.approx_point(g);
                    ins_trace("facet", g, pos);
                    points.push(pos);
                    point_h.push(child_h(
                        pos, $parents, points, point_h, params.grading, cap, &params.size_points,
                    ));
                    point_index.entry(pos.map(|x| (x + 0.0).to_bits())).or_insert(g);
                    let mut fs = OnFacet::default();
                    fs.insert(facet);
                    on_facet.push(fs);
                    absorb!(g);
                    n += 1;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }};
    }
    // Longest-edge fallback for oversized tets whose circumcenter is rejected:
    // a live piece splits its chain, a surface edge inserts an exact on-facet
    // point, an interior edge inserts a guarded midpoint.
    macro_rules! split_longest_edge {
        ($a:expr, $b:expr) => {{
            let (a, b) = ($a, $b);
            let key = sorted2(a, b);
            if live_pieces.contains_key(&key) {
                if split_piece!(a, b) {
                    n += 1;
                }
            } else {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                let marks: OnFacet = on_facet[a].intersection(&on_facet[b]).copied().collect();
                if let Some(ck) = encroached_crease(m, points, &extra_pieces, live_pieces) {
                    if split_piece!(ck.0, ck.1) {
                        n += 1;
                        n_crease_redirect += 1;
                    }
                } else if !marks.is_empty() {
                    let facet = *marks.iter().min().expect("non-empty");
                    let cap = marks
                        .iter()
                        .map(|&f| patch_h[facet_patch[f as usize]])
                        .fold(f64::INFINITY, f64::min);
                    insert_facet_point!(facet, m, &[a, b], cap);
                } else if !point_index.contains_key(&m.map(|x| (x + 0.0).to_bits())) {
                    let admitted = builder.insert_guarded(m, 0.0, |rem| match rem {
                        crate::delaunay::Removal::Face(f) => !tile_map.contains_key(&f),
                        crate::delaunay::Removal::Edge(a, b) => !live_pieces.contains_key(&(a, b)),
                    });
                    if admitted.is_some() {
                        let g = points.len();
                        points.push(m);
                        point_h.push(child_h(
                            m, &[a, b], points, point_h, params.grading, f64::INFINITY,
                            &params.size_points,
                        ));
                        point_index.insert(m.map(|x| (x + 0.0).to_bits()), g);
                        on_facet.push(OnFacet::default());
                        absorb!(g);
                        n += 1;
                        n_midpoint += 1;
                    }
                }
            }
        }};
    }

    // ------------------------------------ the queue loop
    let batch_cap = (slot_tets.len() / 4).max(512);
    loop {
        if n >= batch_cap {
            break;
        }
        if points.len() >= params.max_points {
            break;
        }
        // Priority 1: oversized segment pieces.
        if let Some(key) = crease_q.pop_front() {
            let Some(&seg) = live_pieces.get(&key) else {
                continue;
            };
            if !sized {
                continue;
            }
            // Internal triangulation diagonals are recovered but not eagerly
            // sized; they split only on demand via tile-circumcenter
            // encroachment, like the surface interior they bound.
            if !seg_is_feature[seg] {
                continue;
            }
            // Mean of the endpoint targets: an edge leaving a fine feature is
            // allowed to GROW along the graded field; the min would clamp it
            // to the fine size over its whole length.
            let h = seg_h(seg).min(0.5 * (point_h[key.0] + point_h[key.1]));
            if !(h.is_finite()
                && dist2(points[key.0], points[key.1])
                    > (OVERSIZE_FACTOR * h) * (OVERSIZE_FACTOR * h))
            {
                continue;
            }
            if split_piece!(key.0, key.1) {
                n += 1;
                n_crease_size += 1;
            }
            continue;
        }
        // Priority 2: oversized patch tiles.
        if let Some(key) = tile_q.pop_front() {
            let Some(&facet) = tile_map.get(&key) else {
                continue;
            };
            if !sized {
                continue;
            }
            let pi = facet_patch[facet as usize];
            let h_t = patch_h[pi]
                .min((point_h[key[0]] + point_h[key[1]] + point_h[key[2]]) / 3.0);
            if !h_t.is_finite() {
                continue;
            }
            let Some((cc, r)) =
                tri_circumcenter(points[key[0]], points[key[1]], points[key[2]])
            else {
                continue;
            };
            if r <= 0.5 * OVERSIZE_FACTOR * h_t {
                continue;
            }
            let before = n;
            // Primary placement: the in-plane circumcenter, redirected to a
            // crease split when it encroaches a chain piece, else an exact
            // on-facet Steiner point. (Fat tiles always take this route, so
            // the planar surface density stays as calibrated.)
            if let Some(ck) = encroached_crease(cc, points, &extra_pieces, live_pieces) {
                if split_piece!(ck.0, ck.1) {
                    n += 1;
                }
            } else if !point_index.contains_key(&cc.map(|x| (x + 0.0).to_bits())) {
                insert_facet_point!(facet, cc, &[key[0], key[1], key[2]], patch_h[pi]);
            }
            // Sliver-tile fallback: a tall thin tile (a tile of a tall narrow
            // facet, e.g. a tessellated bore wall) has a huge circumradius but
            // a circumcenter that projects ineffectively; if the primary route
            // made no progress, split the tile's longest edge directly. This
            // is what keeps curved void/surface tilings refining to target.
            if n == before {
                let mut lmax2 = 0.0_f64;
                let mut le = (key[0], key[1]);
                for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 0)] {
                    let d = dist2(points[key[i]], points[key[j]]);
                    if d > lmax2 {
                        lmax2 = d;
                        le = (key[i], key[j]);
                    }
                }
                if lmax2 > (OVERSIZE_FACTOR * h_t) * (OVERSIZE_FACTOR * h_t) {
                    split_longest_edge!(le.0, le.1);
                }
            }
            if n > before {
                n_tile += 1;
            }
            continue;
        }
        // Priority 3: tets, oversized or poor radius-edge quality.
        let Some((slot, tverts)) = tet_q.pop_front() else {
            break;
        };
        if builder.tet_at(slot) != Some(tverts) {
            continue;
        }
        // Region first: it caps every edge target (inherited growth must not
        // undercut the region's own h INSIDE a fine region).
        let region = region_by_slot[slot as usize];
        if region == 0 {
            continue;
        }
        let h_region = h_of_region(region);
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| points[tverts[k]]);
        let mut lmin2 = f64::MAX;
        let mut lmax2 = 0.0_f64;
        let mut longest = (tverts[0], tverts[1]);
        let mut worst_ratio2 = 0.0_f64;
        for i in 0..4 {
            for j in i + 1..4 {
                let d = dist2(p[i], p[j]);
                lmin2 = lmin2.min(d);
                lmax2 = lmax2.max(d);
                let h_ij = h_region.min(0.5 * (point_h[tverts[i]] + point_h[tverts[j]]));
                let ratio2 = if h_ij.is_finite() { d / (h_ij * h_ij) } else { 0.0 };
                if ratio2 > worst_ratio2 {
                    worst_ratio2 = ratio2;
                    longest = (tverts[i], tverts[j]);
                }
            }
        }
        let cc_r = tet_circumcenter(p);
        let bad_quality = params.radius_edge_bound.is_finite()
            && cc_r.is_some_and(|(_, r)| r > params.radius_edge_bound * lmin2.sqrt());
        let maybe_oversized = sized && lmax2 > 0.0;
        if !maybe_oversized && !bad_quality {
            continue;
        }
        let h = h_region
            .min((point_h[tverts[0]]
                + point_h[tverts[1]]
                + point_h[tverts[2]]
                + point_h[tverts[3]])
                / 4.0);
        let oversized = worst_ratio2 > OVERSIZE_FACTOR * OVERSIZE_FACTOR;
        let quality_allowed = bad_quality && (!h.is_finite() || lmin2.sqrt() > 0.2 * h);
        if !oversized && !quality_allowed {
            continue;
        }
        if !oversized && cc_r.is_none() {
            continue;
        }
        let quality_only = !oversized;
        if quality_only {
            let mut tk = tverts;
            tk.sort_unstable();
            let attempts = tried.entry(tk).or_insert(0);
            if *attempts > QUALITY_RETRY_LIMIT {
                continue;
            }
            *attempts = attempts.saturating_add(1);
        }
        let before = n;
        'attempt: {
            let Some((cc, _r)) = cc_r else {
                break 'attempt;
            };
            if let Some(ck) = encroached_crease(cc, points, &extra_pieces, live_pieces) {
                // Pure Delaunay quality steps never touch the boundary.
                if !quality_only && split_piece!(ck.0, ck.1) {
                    n += 1;
                    n_crease_redirect += 1;
                }
                break 'attempt;
            }
            if let Some((facet, tc, tf)) = encroached_tile(cc, &tile_map, &tile_overlay) {
                if quality_only {
                    break 'attempt;
                }
                if let Some(ck) = encroached_crease(tc, points, &extra_pieces, live_pieces) {
                    if split_piece!(ck.0, ck.1) {
                        n += 1;
                        n_crease_redirect += 1;
                    }
                    break 'attempt;
                }
                if point_index.contains_key(&tc.map(|x| (x + 0.0).to_bits())) {
                    break 'attempt;
                }
                insert_facet_point!(facet, tc, &tf, patch_h[facet_patch[facet as usize]]);
                break 'attempt;
            }
            if point_index.contains_key(&cc.map(|x| (x + 0.0).to_bits())) {
                break 'attempt;
            }
            // Guarded for quality AND size splits: an interior point must
            // never remove a constraint. The min-dist floor additionally
            // rejects numerically corrupted circumcenters of near-degenerate
            // tets: for QUALITY candidates an exact circumcenter lands at
            // distance >= radius_edge_bound * lmin, so lmin is a safe floor;
            // for well-shaped OVERSIZED tets the circumcenter sits at
            // ~0.61 lmin (equilateral), so their floor must be lower or every
            // healthy size split bails to the midpoint fallback.
            let floor2 = if quality_only { lmin2 } else { 0.25 * lmin2 };
            let admitted = builder.insert_guarded(cc, floor2, |rem| match rem {
                crate::delaunay::Removal::Face(f) => !tile_map.contains_key(&f),
                crate::delaunay::Removal::Edge(a, b) => !live_pieces.contains_key(&(a, b)),
            });
            if admitted.is_none() {
                guarded_veto += 1;
                break 'attempt;
            }
            guarded_ok += 1;
            if quality_only {
                n_cc_quality += 1;
            } else {
                n_cc_oversized += 1;
            }
            let g = points.len();
            ins_trace("cc", g, cc);
            points.push(cc);
            point_h.push(child_h(
                cc, &tverts, points, point_h, params.grading, f64::INFINITY,
                &params.size_points,
            ));
            point_index.insert(cc.map(|x| (x + 0.0).to_bits()), g);
            on_facet.push(OnFacet::default());
            absorb!(g);
            n += 1;
        }
        // No progress on an oversized tet via the circumcenter routes:
        // longest-edge fallback guarantees the size target.
        if oversized && n == before {
            let (a, b) = longest;
            split_longest_edge!(a, b);
        }
    }
    if cand_trace {
        use std::sync::atomic::Ordering;
        eprintln!(
            "  queue: {n} inserts (crease {n_crease_size} + redirect {n_crease_redirect}, tile {n_tile}, cc {n_cc_oversized} size + {n_cc_quality} quality, midpoint {n_midpoint}), {guarded_veto} veto (nn {}, keep {}), scans {}, ok {guarded_ok}",
            crate::delaunay::GUARDED_NN_BAILS.swap(0, Ordering::Relaxed),
            crate::delaunay::GUARDED_KEEP_VETOES.swap(0, Ordering::Relaxed),
            crate::delaunay::LOCATE_SCANS.swap(0, Ordering::Relaxed),
        );
    }
    n
}

/// Applies one successful insertion's cavity deltas to [`refine_queue`]'s
/// incremental bookkeeping: created tets inherit the region of the removed
/// tet behind their base face (the cone allocates before the cavity is
/// retired, so parents stay readable; regions thereby split correctly across
/// on-facet insertions), tiles swallowed by the cavity leave the live tile
/// set, and the new local faces are (re)evaluated with the combinatorial
/// provenance rule [`facet_of_tile`].
#[allow(clippy::too_many_arguments)]
fn absorb_insert_deltas(
    builder: &DelaunayBuilder,
    p_idx: usize,
    points: &[[f64; 3]],
    on_facet: &[OnFacet],
    region_by_slot: &mut Vec<u32>,
    tile_map: &mut DMap<[usize; 3], u32>,
    tet_q: &mut VecDeque<(u32, [usize; 4])>,
    tile_q: &mut VecDeque<[usize; 3]>,
    tile_overlay: &mut Vec<([f64; 3], f64, u32, [usize; 3])>,
) {
    region_by_slot.resize(builder.slot_count(), 0);
    let created: Vec<u32> = builder.last_created().to_vec();
    let parents: Vec<u32> = builder.last_parents().collect();
    for (i, &nt) in created.iter().enumerate() {
        region_by_slot[nt as usize] = region_by_slot[parents[i] as usize];
        if let Some(tv) = builder.tet_at(nt) {
            tet_q.push_back((nt, tv));
        }
    }
    // Faces that survived the cavity: the base faces of the cone.
    let mut surviving: DSet<[usize; 3]> = DSet::default();
    for &nt in &created {
        if let Some(tv) = builder.tet_at(nt) {
            let mut base = [0usize; 3];
            let mut k = 0;
            for &v in &tv {
                if v != p_idx && k < 3 {
                    base[k] = v;
                    k += 1;
                }
            }
            if k == 3 {
                surviving.insert(sorted3(base));
            }
        }
    }
    // Tiles swallowed by the cavity (a removed super-corner tet still owns one
    // all-real face, e.g. a hull tile).
    for &rm in builder.last_removed() {
        let vs = builder.verts_of_slot(rm);
        for fi in TET_FACES {
            let f = fi.map(|k| vs[k]);
            let (Some(a), Some(b), Some(c)) = (f[0], f[1], f[2]) else {
                continue;
            };
            let key = sorted3([a, b, c]);
            if !surviving.contains(&key) {
                tile_map.remove(&key);
            }
        }
    }
    // New and re-based local faces: re-evaluate tile candidacy.
    let mut seen: DSet<[usize; 3]> = DSet::default();
    for &nt in &created {
        let Some(tv) = builder.tet_at(nt) else {
            continue;
        };
        for fv in TET_FACES.iter() {
            let key = sorted3(fv.map(|k| tv[k]));
            if !seen.insert(key) {
                continue;
            }
            let tiled = facet_of_tile(on_facet, key);
            match (tiled, tile_map.get(&key).copied()) {
                (Some(facet), prev) => {
                    if prev != Some(facet) {
                        tile_map.insert(key, facet);
                    }
                    tile_q.push_back(key);
                    if let Some((tc, tr)) =
                        tri_circumcenter(points[key[0]], points[key[1]], points[key[2]])
                    {
                        tile_overlay.push((tc, tr, facet, key));
                    }
                }
                (None, Some(_)) => {
                    tile_map.remove(&key);
                }
                (None, None) => {}
            }
        }
    }
}

/// Quality summary of a tet mesh.
#[derive(Debug, Clone, Copy)]
pub struct QualityStats {
    /// Number of tets.
    pub n_tets: usize,
    /// Smallest dihedral angle in degrees (sliver indicator; the load-bearing
    /// metric for Nedelec conditioning).
    pub min_dihedral_deg: f64,
    /// Largest circumradius / shortest-edge ratio.
    pub max_radius_edge: f64,
    /// Longest edge in the mesh.
    pub max_edge: f64,
}

/// Computes quality statistics over all tets.
pub fn quality_stats(mesh: &TetMesh) -> QualityStats {
    let mut min_dihedral = f64::MAX;
    let mut max_re = 0.0_f64;
    let mut max_edge2 = 0.0_f64;
    for t in &mesh.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| mesh.points[t[k]]);
        let mut lmin2 = f64::MAX;
        for i in 0..4 {
            for j in i + 1..4 {
                let d2: f64 = (0..3).map(|k| (p[i][k] - p[j][k]).powi(2)).sum();
                lmin2 = lmin2.min(d2);
                max_edge2 = max_edge2.max(d2);
            }
        }
        if let Some((_, r)) = tet_circumcenter(p) {
            max_re = max_re.max(r / lmin2.sqrt());
        }
        // Dihedral angle at each of the 6 edges: angle between the projections
        // of the two opposite vertices onto the plane normal to the edge.
        for i in 0..4 {
            for j in i + 1..4 {
                let others: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
                let (a, b) = (p[i], p[j]);
                let tlen: f64 = (0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt();
                let tv: [f64; 3] = std::array::from_fn(|k| (b[k] - a[k]) / tlen);
                let perp = |q: [f64; 3]| -> [f64; 3] {
                    let w: [f64; 3] = std::array::from_fn(|k| q[k] - a[k]);
                    let s: f64 = (0..3).map(|k| w[k] * tv[k]).sum();
                    std::array::from_fn(|k| w[k] - s * tv[k])
                };
                let (u, v) = (perp(p[others[0]]), perp(p[others[1]]));
                let nu: f64 = (0..3).map(|k| u[k] * u[k]).sum::<f64>().sqrt();
                let nv: f64 = (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
                if nu * nv == 0.0 {
                    continue;
                }
                let cosang =
                    ((0..3).map(|k| u[k] * v[k]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
                min_dihedral = min_dihedral.min(cosang.acos().to_degrees());
            }
        }
    }
    QualityStats {
        n_tets: mesh.tets.len(),
        min_dihedral_deg: min_dihedral,
        max_radius_edge: max_re,
        max_edge: max_edge2.sqrt(),
    }
}
