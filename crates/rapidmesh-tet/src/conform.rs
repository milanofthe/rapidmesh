//! Boundary recovery and region tagging: TaggedPlc to a conforming tet mesh.
//!
//! The PLC's true constraints are planar tagged PATCHES (maximal connected
//! coplanar groups of facets with equal tags) and their boundary CREASE
//! edges, not the individual facet triangles: requiring every assembler
//! diagonal as a mesh edge over-constrains and triggers encroachment
//! ping-pong at the acute angles those diagonals create. So:
//!
//! 1. Crease edges are recovered by midpoint splitting (1D features meet at
//!    benign angles in EM geometry).
//! 2. Each patch must be exactly TILED by mesh faces lying on it. The check
//!    is an exact area comparison (expansion-arithmetic shoelace in the
//!    patch projection); a deficit is repaired by inserting Steiner points
//!    where tet edges pierce the patch.
//!
//! Steiner points are rounded f64 and may sit within an ulp of the patch
//! plane; patch membership of points is therefore tracked combinatorially
//! (a point belongs to the patches it was created on), never re-derived
//! geometrically.

use crate::delaunay::DelaunayBuilder;
use rapidmesh_csg::classify::point_inside_solid;
use rapidmesh_csg::Tri;
use rapidmesh_exact::geom::det4;
use rapidmesh_exact::{orient3d, Axis, Expansion, Point3, Sign};
use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind, TaggedPlc};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
/// FxHasher is unseeded (deterministic) and much faster than SipHash on the
/// short integer keys these maps use.
type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;
type DSet<T> = HashSet<T, DState>;

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
    /// Patches the conformity loop gave up on (stagnant tiling deficit).
    /// Empty for a fully conforming mesh; non-empty means the faces of
    /// these patches may carry holes or double layers and the mesh needs
    /// review before physics runs on it.
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
    /// Member facet triangles (global vertex indices).
    members: Vec<[usize; 3]>,
    /// Three non-collinear points defining the plane.
    plane: [usize; 3],
    /// Projection axis and 2D orientation of the members.
    axis: Axis,
    orientation: Sign,
    face_tag: FaceTag,
    regions: [RegionTag; 2],
    /// Analytic surface of the members (index into the PLC surface table).
    surface: u32,
    /// Exact double area of the patch in the projection.
    area2: Expansion,
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

/// Registers a crease child edge produced by a split. The child may ALREADY
/// be a crease (f64 welding in scene assembly can create T-junctions whose
/// split point is an existing crease endpoint): then its mark sets merge
/// instead of duplicating the entry, which would desync `creases` from
/// `crease_marks`.
fn push_crease_child(
    creases: &mut Vec<(usize, usize)>,
    crease_marks: &mut DMap<(usize, usize), DSet<usize>>,
    key: (usize, usize),
    marks: &DSet<usize>,
) {
    match crease_marks.entry(key) {
        std::collections::hash_map::Entry::Occupied(mut o) => {
            o.get_mut().extend(marks.iter().copied());
        }
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(marks.clone());
            creases.push(key);
        }
    }
}

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Which side of a patch plane a tet lies on: the sign of the SUM of its four
/// vertices' plane offsets (4x the centroid offset; the determinant is linear
/// in the query point, so the sum needs no division and stays a polynomial in
/// the inputs).
///
/// On-patch Steiner points are rounded f64 and may hover an ulp off the exact
/// plane (non-axis-aligned patches only); the Delaunay triangulation can then
/// contain a flat sliver tet WITHIN the patch whose floor and fan faces would
/// both pass a marks+containment tile test, double-covering the projected
/// area. The side sum is the disambiguator: a face is a true tile only when
/// its two adjacent tets lie on opposite sides.
///
/// A fast f64 evaluation with a static error bound resolves every tet with a
/// genuinely off-plane vertex; only near-plane slivers pay the exact
/// expansion fallback. An exactly-zero sum (constructible from symmetric ulp
/// offsets) deterministically counts as Positive.
fn tet_plane_side(patch: &Patch, points: &[[f64; 3]], t: [usize; 4]) -> Sign {
    plane_side(patch, points, t.map(|v| points[v]))
}

/// [`tet_plane_side`] over explicit positions (used for the pseudo-tet of a
/// hull face and its super corner).
fn plane_side(patch: &Patch, points: &[[f64; 3]], verts: [[f64; 3]; 4]) -> Sign {
    let pl = patch.plane.map(|i| points[i]);
    let sub = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| a[k] - b[k]) };
    let u = sub(pl[1], pl[0]);
    let v = sub(pl[2], pl[0]);
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let inf_norm = |q: [f64; 3]| -> f64 { q.iter().fold(0.0_f64, |m, x| m.max(x.abs())) };
    let mut sum = 0.0_f64;
    let mut wmax = 0.0_f64;
    for &vt in &verts {
        let w = sub(vt, pl[0]);
        sum += (0..3).map(|k| n[k] * w[k]).sum::<f64>();
        wmax = wmax.max(inf_norm(w));
    }
    // Static filter: the f64 evaluation of each offset n.w errs by at most
    // c * eps * max|u,v|^2 * max|w| with a small constant; 256 covers the
    // cross product, dot product, and the 4-term sum with margin.
    let m = inf_norm(u).max(inf_norm(v));
    let err = 256.0 * f64::EPSILON * m * m * wmax;
    if sum.abs() > err {
        return Sign::of_f64(sum);
    }
    let hom = |p: [f64; 3]| Point3::Explicit(p).hom::<Expansion>();
    let base = [hom(pl[0]), hom(pl[1]), hom(pl[2])];
    let mut acc = Expansion::from_f64(0.0);
    for &vt in &verts {
        let rows = [
            base[0].clone(),
            base[1].clone(),
            base[2].clone(),
            hom(vt),
        ];
        acc = acc.add(&det4(&rows));
    }
    match acc.sign() {
        Sign::Zero => Sign::Positive,
        s => s,
    }
}

/// The four vertex-index triples spanning a tet's faces (unoriented; users
/// sort the result for keying).
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// Exact double signed area of the projected triangle, as an expansion.
fn area2_exact(points: &[[f64; 3]], f: [usize; 3], axis: Axis) -> Expansion {
    let (u, v) = match axis {
        Axis::X => (1, 2),
        Axis::Y => (2, 0),
        Axis::Z => (0, 1),
    };
    let p = |i: usize, k: usize| Expansion::from_f64(points[f[i]][k]);
    // au(bv-cv) + bu(cv-av) + cu(av-bv), fully expanded into exact products.
    let term = |a: usize, b: usize, c: usize| p(a, u).mul(&p(b, v)).add(&p(a, u).mul(&p(c, v)).neg());
    term(0, 1, 2)
        .add(&term(1, 2, 0))
        .add(&term(2, 0, 1))
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
        .map(|members| {
            let first = tri(members[0]);
            let t0 = Tri::new(
                plc.vertices[first[0]],
                plc.vertices[first[1]],
                plc.vertices[first[2]],
            );
            let (axis, orientation) = t0.projection_axis();
            let member_tris: Vec<[usize; 3]> = members.iter().map(|&i| tri(i)).collect();
            let mut area2 = Expansion::from_f64(0.0);
            for &f in &member_tris {
                let mut a = area2_exact(&plc.vertices, f, axis);
                if a.sign() == Sign::Negative {
                    a = a.neg();
                }
                area2 = area2.add(&a);
            }
            Patch {
                members: member_tris,
                plane: first,
                axis,
                orientation,
                face_tag: plc.face_tags[members[0]],
                regions: plc.region_tags[members[0]],
                surface: plc.surface_refs[members[0]].0,
                area2,
            }
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
            size_points: Vec::new(),
        },
    )
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh, refined to
/// the given sizing and quality targets (best effort under
/// `params.max_points`).
pub fn mesh_plc_with(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    let trace = std::env::var_os("RAPIDMESH_TRACE").is_some();
    let mut points: Vec<[f64; 3]> = plc.vertices.clone();
    let patches = build_patches(plc);

    // Combinatorial point-on-patch membership.
    let mut on_patch: Vec<DSet<usize>> = vec![DSet::default(); points.len()];
    for (pi, p) in patches.iter().enumerate() {
        for f in &p.members {
            for &v in f {
                on_patch[v].insert(pi);
            }
        }
    }

    // Crease sub-edges: PLC edges that are not interior to a single patch.
    let mut edge_owner_patches: DMap<(usize, usize), DSet<usize>> = DMap::default();
    let mut edge_count: DMap<(usize, usize), usize> = DMap::default();
    for (pi, p) in patches.iter().enumerate() {
        for f in &p.members {
            for e in 0..3 {
                let key = sorted2(f[e], f[(e + 1) % 3]);
                edge_owner_patches.entry(key).or_default().insert(pi);
                *edge_count.entry(key).or_default() += 1;
            }
        }
    }
    let mut creases: Vec<(usize, usize)> = edge_count
        .iter()
        .filter(|(key, &cnt)| {
            // Interior to one patch: exactly two member facets of the same
            // single patch share it. Everything else is a feature.
            !(cnt == 2 && edge_owner_patches[*key].len() == 1)
        })
        .map(|(&key, _)| key)
        .collect();
    let crease_patches: DMap<(usize, usize), DSet<usize>> = creases
        .iter()
        .map(|&key| (key, edge_owner_patches[&key].clone()))
        .collect();
    let mut crease_marks: DMap<(usize, usize), DSet<usize>> = crease_patches.clone();

    // ------------------------------------------------- recovery loop
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

    let plane_sign = |patch: &Patch, points: &[[f64; 3]], v: usize| -> Sign {
        orient3d(
            &Point3::Explicit(points[patch.plane[0]]),
            &Point3::Explicit(points[patch.plane[1]]),
            &Point3::Explicit(points[patch.plane[2]]),
            &Point3::Explicit(points[v]),
        )
        .expect("explicit")
    };
    // Per-patch member index: PLC vertex positions never move, so this is
    // built once for the whole run.
    let patch_grids: Vec<Tri2Grid> = patches
        .iter()
        .map(|p| Tri2Grid::build(&points, &p.members, p.axis))
        .collect();
    let inside_patch = |pi: usize, patch: &Patch, points: &[[f64; 3]], q: [f64; 3]| -> bool {
        let qp = Point3::Explicit(q);
        let (u, v) = Tri2Grid::axis_uv(patch.axis);
        patch_grids[pi].candidates([q[u], q[v]]).any(|mi| {
            let f = patch.members[mi as usize];
            Tri::new(points[f[0]], points[f[1]], points[f[2]]).contains_coplanar(
                &qp,
                patch.axis,
                patch.orientation,
            )
        })
    };
    // Strictly inside some member triangle: keeps Steiner points off the
    // patch boundary (the creases), which midpoint recovery could otherwise
    // never restore.
    let strictly_inside_patch = |pi: usize, patch: &Patch, points: &[[f64; 3]], q: [f64; 3]| -> bool {
        let qp = Point3::Explicit(q);
        let (u, v) = Tri2Grid::axis_uv(patch.axis);
        patch_grids[pi].candidates([q[u], q[v]]).any(|mi| {
            let f = patch.members[mi as usize];
            (0..3).all(|e| {
                rapidmesh_exact::orient2d(
                    &Point3::Explicit(points[f[e]]),
                    &Point3::Explicit(points[f[(e + 1) % 3]]),
                    &qp,
                    patch.axis,
                )
                .expect("valid") == patch.orientation
            })
        })
    };

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
    let patch_h_init: Vec<f64> = patches
        .iter()
        .map(|p| {
            h_of_region_init(p.regions[0].0)
                .min(h_of_region_init(p.regions[1].0))
                .min(h_of_face(p.face_tag))
        })
        .collect();
    let mut point_h: Vec<f64> = (0..points.len())
        .map(|v| {
            let mut h = on_patch[v]
                .iter()
                .map(|&pi| patch_h_init[pi])
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

    let mut round = 0;
    let mut refine_round = 0;
    // Patches whose tiling repair stagnated (see the stagnation guard).
    let mut abandoned: DSet<usize> = DSet::default();
    let mut patch_progress: DMap<usize, (f64, usize)> = DMap::default();
    let mut tried: DMap<[usize; 4], u8> = DMap::default();
    #[allow(clippy::type_complexity)]
    let (tets, patch_faces, tet_region): (Vec<[usize; 4]>, Vec<Vec<[usize; 3]>>, Vec<u32>) = 'outer: loop {
        round += 1;
        assert!(round <= 16384, "boundary recovery did not converge");

        let t_round = std::time::Instant::now();
        let slot_tets = builder.tets_with_slots();
        let dt_tets: Vec<[usize; 4]> = slot_tets.iter().map(|&(_, t)| t).collect();
        // Convex-hull faces with the super corner on their far side: a
        // candidate tile with a single real owner compares against the
        // outside via this corner (see the tiling rule below).
        let hull_super: DMap<[usize; 3], [f64; 3]> = builder
            .hull_faces()
            .into_iter()
            .map(|(f, s)| (sorted3(f), s))
            .collect();
        // Edges are only ever queried between crease endpoints (the missing
        // check below); collecting all 6 edges of every tet would dominate
        // the round.
        let crease_endpoint: DSet<usize> =
            creases.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut dt_edges: DSet<(usize, usize)> = DSet::default();
        let mut dt_faces: DMap<[usize; 3], Vec<u32>> = DMap::default();
        for (ti, t) in dt_tets.iter().enumerate() {
            for i in 0..4 {
                for j in i + 1..4 {
                    if crease_endpoint.contains(&t[i]) && crease_endpoint.contains(&t[j]) {
                        dt_edges.insert(sorted2(t[i], t[j]));
                    }
                }
                dt_faces
                    .entry(sorted3(TET_FACES[i].map(|k| t[k])))
                    .or_default()
                    .push(ti as u32);
            }
        }

        let t_scan = t_round.elapsed();
        // 1. Missing crease sub-edges: midpoint split.
        let missing: Vec<(usize, usize)> = creases
            .iter()
            .copied()
            .filter(|key| !dt_edges.contains(key))
            .collect();
        if !missing.is_empty() {
            if trace {
                let len = |&(a, b): &(usize, usize)| -> f64 {
                    (0..3)
                        .map(|k| (points[a][k] - points[b][k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                let min = missing.iter().map(len).fold(f64::MAX, f64::min);
                eprintln!(
                    "round {round}: {} missing crease edges (min len {min:.2e})",
                    missing.len()
                );
                if std::env::var_os("RAPIDMESH_CREASE_TRACE").is_some() {
                    for &(a, b) in missing.iter().take(4) {
                        eprintln!("  missing ({a},{b}): {:?} -> {:?}", points[a], points[b]);
                    }
                }
            }
            for (a, b) in missing {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                // A vertex may already sit exactly at the midpoint (e.g. a
                // piercing Steiner point from a symmetric tet edge): reuse
                // it as the chain split point instead of duplicating.
                let g = match point_index.get(&m.map(|x| (x + 0.0).to_bits())) {
                    Some(&g) => g,
                    None => {
                        let g = points.len();
                        points.push(m);
                        point_h.push(child_h(
                            m,
                            &[a, b],
                            &points,
                            point_h,
                            params.grading,
                            f64::INFINITY,
                    &params.size_points,
                ));
                        builder.insert(m);
                        point_index.insert(m.map(|x| (x + 0.0).to_bits()), g);
                        on_patch.push(DSet::default());
                        g
                    }
                };
                if g == a || g == b {
                    // The midpoint rounded onto an endpoint: the crease is at
                    // f64 resolution and cannot be split further. Leave it;
                    // looping a zero-length child forever helps nobody.
                    continue;
                }
                let marks = crease_marks.remove(&(a, b)).expect("known crease");
                on_patch[g].extend(marks.iter().copied());
                creases.retain(|&e| e != (a, b));
                push_crease_child(&mut creases, &mut crease_marks, sorted2(a, g), &marks);
                push_crease_child(&mut creases, &mut crease_marks, sorted2(g, b), &marks);
            }
            continue;
        }

        // 2. Patch tiling check. Candidate patches per face come from the
        // intersection of its vertices' patch memberships (iterating all
        // faces per patch is O(patches * faces) and dominated large meshes),
        // and the area gate runs in f64: it is tolerant anyway (Steiner
        // chain points sit within rounding of patch boundaries, leaving
        // ~1e-16-relative slivers no refinement can remove), and the f64
        // shoelace noise (~1e-15 relative) is far below the 1e-9 gate.
        // Structural conformity stays exact: tiles are actual tet faces.
        let area2_f64 = |f: &[usize; 3], axis: Axis| -> f64 {
            let (u, v) = match axis {
                Axis::X => (1, 2),
                Axis::Y => (2, 0),
                Axis::Z => (0, 1),
            };
            let (a, b, c) = (points[f[0]], points[f[1]], points[f[2]]);
            ((b[u] - a[u]) * (c[v] - a[v]) - (b[v] - a[v]) * (c[u] - a[u])).abs()
        };
        let scale = (0..3).map(|k| bhi[k] - blo[k]).fold(1.0_f64, f64::max);
        let mut all_tilings: Vec<Vec<[usize; 3]>>;
        let mut tile_area: Vec<f64>;
        // The tiling scan re-runs after adoptions (marks only grow, so this
        // terminates; adoptions are rare one-offs).
        loop {
            all_tilings = vec![Vec::new(); patches.len()];
            tile_area = vec![0.0; patches.len()];
            // Stray vertices to adopt: a vertex without the patch mark whose
            // faces nevertheless form the patch's separating terrain, sitting
            // within rounding of the exact plane (an unguarded insertion that
            // landed ulp-close). Without adoption its faces are uncountable
            // and the patch reports a permanent unreparable deficit.
            let mut adoptions: Vec<(usize, usize)> = Vec::new();
            for (f, owners) in &dt_faces {
                let sets = [&on_patch[f[0]], &on_patch[f[1]], &on_patch[f[2]]];
                if sets.iter().filter(|s| !s.is_empty()).count() < 2 {
                    continue;
                }
                let mut cand: Vec<usize> = Vec::new();
                for s in sets {
                    for &pi in s.iter() {
                        if !cand.contains(&pi) {
                            cand.push(pi);
                        }
                    }
                }
                for pi in cand {
                    let have: [bool; 3] = std::array::from_fn(|i| sets[i].contains(&pi));
                    let n_marked = have.iter().filter(|x| **x).count();
                    if n_marked < 2 {
                        continue;
                    }
                    let patch = &patches[pi];
                    let c: [f64; 3] = std::array::from_fn(|k| {
                        (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                    });
                    if !inside_patch(pi, patch, &points, c) {
                        continue;
                    }
                    // A true tile separates the two sides of the patch; a
                    // face between two SAME-side tets is the floor or fan of
                    // a flat sliver around a hovering Steiner point (see
                    // [tet_plane_side]) and double-covers the projection.
                    // The side test is meaningful ONLY for that flat
                    // all-marked complex: distant tets are judged against
                    // the INFINITE plane, which re-enters concave geometry
                    // (a torus bore's big void tets can sit on either side
                    // wholesale), so a face whose opposite vertices are not
                    // both on the patch counts unconditionally. Hull faces
                    // (one real owner) compare against the super corner on
                    // their far side: a hover that dipped an ulp OUTSIDE
                    // the hull fans same-side hull faces.
                    let opp_marked = |owner: u32| -> bool {
                        dt_tets[owner as usize]
                            .iter()
                            .find(|v| !f.contains(v))
                            .is_some_and(|v| on_patch[*v].contains(&pi))
                    };
                    // A tet with ALL FOUR vertices marked lies entirely in
                    // the patch plane (marks imply on-patch): a flat sliver
                    // tet whose two cap pairs are both valid triangulations
                    // of the same projected quad. Counting all four caps
                    // double-covers the projection (the horn-loft -2.6%
                    // "uncovered" giveup). Faces interior to such a complex
                    // never count; cap faces count only on the floor side
                    // (outer neighbor on the Negative side of the plane), so
                    // the complex tiles once and the separating terrain
                    // stays watertight (its tets classify into the front
                    // region, like the hover-sliver floor rule below).
                    let all_marked = |owner: u32| -> bool {
                        dt_tets[owner as usize]
                            .iter()
                            .all(|v| on_patch[*v].contains(&pi))
                    };
                    let rejected = if owners.len() == 2 {
                        let m0 = all_marked(owners[0]);
                        let m1 = all_marked(owners[1]);
                        if m0 && m1 {
                            true
                        } else if m0 || m1 {
                            let outer = if m0 { owners[1] } else { owners[0] };
                            tet_plane_side(patch, &points, dt_tets[outer as usize])
                                == Sign::Positive
                        } else {
                            opp_marked(owners[0])
                                && opp_marked(owners[1])
                                && tet_plane_side(patch, &points, dt_tets[owners[0] as usize])
                                    == tet_plane_side(patch, &points, dt_tets[owners[1] as usize])
                        }
                    } else {
                        match hull_super.get(f) {
                            Some(&sc) => {
                                let super_side = plane_side(
                                    patch,
                                    &points,
                                    [points[f[0]], points[f[1]], points[f[2]], sc],
                                );
                                if all_marked(owners[0]) {
                                    super_side == Sign::Positive
                                } else {
                                    opp_marked(owners[0])
                                        && tet_plane_side(
                                            patch,
                                            &points,
                                            dt_tets[owners[0] as usize],
                                        ) == super_side
                                }
                            }
                            None => false,
                        }
                    };
                    if rejected {
                        continue;
                    }
                    if n_marked == 3 {
                        tile_area[pi] += area2_f64(f, patch.axis);
                        all_tilings[pi].push(*f);
                        continue;
                    }
                    // Two marked vertices and a separating face: adoption
                    // candidate, gated on exact-plane proximity (ulp class
                    // only; genuinely off-plane points must not bend the
                    // patch).
                    let miss = (0..3).find(|&i| !have[i]).expect("one unmarked");
                    let v = f[miss];
                    let pl = patch.plane.map(|i| points[i]);
                    let eu: [f64; 3] = std::array::from_fn(|k| pl[1][k] - pl[0][k]);
                    let ev: [f64; 3] = std::array::from_fn(|k| pl[2][k] - pl[0][k]);
                    let nrm = [
                        eu[1] * ev[2] - eu[2] * ev[1],
                        eu[2] * ev[0] - eu[0] * ev[2],
                        eu[0] * ev[1] - eu[1] * ev[0],
                    ];
                    let nn: f64 = nrm.iter().map(|x| x * x).sum();
                    if nn == 0.0 {
                        continue;
                    }
                    let d: f64 = (0..3).map(|k| nrm[k] * (points[v][k] - pl[0][k])).sum();
                    let tol = 1e-12 * scale;
                    if d * d > tol * tol * nn {
                        continue;
                    }
                    if patch_inside_closed(&patch_grids[pi], patch, &points, points[v]) {
                        adoptions.push((v, pi));
                    }
                }
            }
            if adoptions.is_empty() {
                break;
            }
            if trace {
                eprintln!("round {round}: adopting {} stray on-plane vertices", adoptions.len());
                if std::env::var_os("RAPIDMESH_HOLE_TRACE").is_some() {
                    for &(v, pi) in adoptions.iter().take(12) {
                        let mut marks: Vec<usize> = on_patch[v].iter().copied().collect();
                        marks.sort_unstable();
                        eprintln!("    adopt v{v} -> patch {pi} (had {marks:?}) at {:?}", points[v]);
                    }
                }
            }
            for (v, pi) in adoptions {
                on_patch[v].insert(pi);
            }
        }
        let mut deficits: Vec<usize> = Vec::new();
        for (pi, patch) in patches.iter().enumerate() {
            if abandoned.contains(&pi) {
                continue;
            }
            let want = patch.area2.approx().abs();
            if (tile_area[pi] - want).abs() > 1e-9 * want {
                deficits.push(pi);
            }
        }
        // Stagnation guard: a deficit that repair points stop improving is
        // structurally stuck (near-degenerate input facets whose neighbor
        // planes are within float rounding: off-plane Steiner points of the
        // neighbor block exact in-plane tiles forever). Give the patch up
        // and keep its partial tiling instead of looping; such inputs need
        // mesh repair, which is out of scope for a PLC mesher.
        for &pi in &deficits {
            let want = patches[pi].area2.approx().abs();
            let e = patch_progress.entry(pi).or_insert((f64::MIN, 0usize));
            if tile_area[pi] > e.0 + 1e-6 * want {
                *e = (tile_area[pi], 0);
            } else {
                e.1 += 1;
                if e.1 >= 10 {
                    eprintln!(
                        "rapidmesh: giving up on patch {pi} tiling ({:.1}% area uncovered); input likely needs mesh repair",
                        100.0 * (want - tile_area[pi]) / want
                    );
                    if std::env::var_os("RAPIDMESH_HOLE_TRACE").is_some() && tile_area[pi] > want {
                        // OVERCOUNT: more projected area counted than the
                        // patch owns. Dump every counted tile with its
                        // distance to the patch plane and its side pattern.
                        let patch = &patches[pi];
                        let pl = patch.plane.map(|i| points[i]);
                        let eu: [f64; 3] = std::array::from_fn(|k| pl[1][k] - pl[0][k]);
                        let ev: [f64; 3] = std::array::from_fn(|k| pl[2][k] - pl[0][k]);
                        let nrm = [
                            eu[1] * ev[2] - eu[2] * ev[1],
                            eu[2] * ev[0] - eu[0] * ev[2],
                            eu[0] * ev[1] - eu[1] * ev[0],
                        ];
                        let nlen = nrm.iter().map(|x| x * x).sum::<f64>().sqrt();
                        eprintln!(
                            "  patch {pi}: axis {:?} members {} want {want:.6e} counted {:.6e}",
                            patch.axis,
                            patch.members.len(),
                            tile_area[pi]
                        );
                        for f in &all_tilings[pi] {
                            let dmax = f
                                .iter()
                                .map(|&v| {
                                    ((0..3)
                                        .map(|k| nrm[k] * (points[v][k] - pl[0][k]))
                                        .sum::<f64>()
                                        / nlen)
                                        .abs()
                                })
                                .fold(0.0_f64, f64::max);
                            let c: [f64; 3] = std::array::from_fn(|k| {
                                (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                            });
                            eprintln!(
                                "    tile {f:?} area {:.3e} plane-dist {dmax:.3e} c {c:?}",
                                area2_f64(f, patch.axis)
                            );
                        }
                        // Pairwise projected-overlap scan: which tiles double
                        // up? (diagnosis only, f64 SAT on the projection)
                        let (u, v) = match patch.axis {
                            Axis::X => (1, 2),
                            Axis::Y => (2, 0),
                            Axis::Z => (0, 1),
                        };
                        let p2 = |i: usize| -> [f64; 2] { [points[i][u], points[i][v]] };
                        let overlap2d = |a: &[usize; 3], b: &[usize; 3]| -> bool {
                            let ta = a.map(p2);
                            let tb = b.map(p2);
                            let axes = |t: &[[f64; 2]; 3]| -> Vec<[f64; 2]> {
                                (0..3)
                                    .map(|i| {
                                        let (p, q) = (t[i], t[(i + 1) % 3]);
                                        [q[1] - p[1], p[0] - q[0]]
                                    })
                                    .collect()
                            };
                            for ax in axes(&ta).into_iter().chain(axes(&tb)) {
                                let pr = |t: &[[f64; 2]; 3]| {
                                    let d: Vec<f64> = t
                                        .iter()
                                        .map(|p| p[0] * ax[0] + p[1] * ax[1])
                                        .collect();
                                    (d.iter().cloned().fold(f64::MAX, f64::min),
                                     d.iter().cloned().fold(f64::MIN, f64::max))
                                };
                                let (alo, ahi) = pr(&ta);
                                let (blo2, bhi2) = pr(&tb);
                                let eps = 1e-12 * (ahi - alo).abs().max((bhi2 - blo2).abs());
                                if ahi <= blo2 + eps || bhi2 <= alo + eps {
                                    return false;
                                }
                            }
                            true
                        };
                        let tiles = &all_tilings[pi];
                        for i in 0..tiles.len() {
                            for j in i + 1..tiles.len() {
                                if overlap2d(&tiles[i], &tiles[j]) {
                                    eprintln!(
                                        "    OVERLAP {:?} <-> {:?}",
                                        tiles[i], tiles[j]
                                    );
                                }
                            }
                        }
                    }
                    if std::env::var_os("RAPIDMESH_HOLE_TRACE").is_some() {
                        // IN-PLANE faces inside the patch that do not count:
                        // the hole's roof, categorized by which filter
                        // rejects them.
                        let patch = &patches[pi];
                        let pl = patch.plane.map(|i| points[i]);
                        let eu: [f64; 3] = std::array::from_fn(|k| pl[1][k] - pl[0][k]);
                        let ev: [f64; 3] = std::array::from_fn(|k| pl[2][k] - pl[0][k]);
                        let nrm = [
                            eu[1] * ev[2] - eu[2] * ev[1],
                            eu[2] * ev[0] - eu[0] * ev[2],
                            eu[0] * ev[1] - eu[1] * ev[0],
                        ];
                        let nlen = nrm.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let pdist = |v: usize| -> f64 {
                            (0..3)
                                .map(|k| nrm[k] * (points[v][k] - pl[0][k]))
                                .sum::<f64>()
                                / nlen
                        };
                        for (f, owners) in &dt_faces {
                            let c: [f64; 3] = std::array::from_fn(|k| {
                                (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                            });
                            if !inside_patch(pi, patch, &points, c) {
                                continue;
                            }
                            // Only the actual plane: skip the projection noise.
                            if f.iter().any(|&v| pdist(v).abs() > 1e-9 * scale) {
                                continue;
                            }
                            if f.iter().all(|&v| on_patch[v].contains(&pi)) {
                                continue; // counted (or rejected as sliver cap)
                            }
                            let side_info = if owners.len() == 2 {
                                let s0 = tet_plane_side(patch, &points, dt_tets[owners[0] as usize]);
                                let s1 = tet_plane_side(patch, &points, dt_tets[owners[1] as usize]);
                                format!("sides {s0:?}/{s1:?}")
                            } else {
                                "hull".into()
                            };
                            eprintln!(
                                "  roof {f:?} area {:.3e} {side_info}",
                                area2_f64(f, patch.axis)
                            );
                            for &v in f {
                                let mut marks: Vec<usize> =
                                    on_patch[v].iter().copied().collect();
                                marks.sort_unstable();
                                eprintln!(
                                    "    v{v} marks {marks:?} at {:?} h={:.4}",
                                    points[v], point_h[v]
                                );
                            }
                        }
                    }
                    abandoned.insert(pi);
                }
            }
        }
        deficits.retain(|pi| !abandoned.contains(pi));
        if deficits.is_empty() {
            // Conforming. Classify every tet ONCE per round (flood fill;
            // the per-candidate ray casting it replaces dominated
            // refinement), then apply one sizing/quality refinement step;
            // when it makes no insertion (or the point budget is spent),
            // finish.
            let t_tile = t_round.elapsed();
            let tet_region =
                classify_tet_regions(&points, &dt_tets, &patches, &all_tilings, &dt_faces, (blo, bhi));
            let t_classify = t_round.elapsed();
            if points.len() >= params.max_points {
                if trace {
                    eprintln!("refinement stopped at point budget ({})", points.len());
                }
                break 'outer (dt_tets, all_tilings, tet_region);
            }
            refine_round += 1;
            if refine_round > 1000 {
                // Sizing/quality are targets, not hard bounds (as in gmsh):
                // stop best-effort instead of chasing corner configurations.
                if trace {
                    eprintln!("refinement stopped at round budget");
                }
                break 'outer (dt_tets, all_tilings, tet_region);
            }
            let inserted = refine_queue(
                params,
                &patches,
                &patch_grids,
                &slot_tets,
                &all_tilings,
                &tet_region,
                &mut tried,
                &mut points,
                point_h,
                &mut builder,
                &mut point_index,
                &mut on_patch,
                &mut creases,
                &mut crease_marks,
            );
            if trace && inserted > 0 {
                eprintln!(
                    "refine round {refine_round}: inserted {inserted}, {} points, scan {:.1?} tile {:.1?} classify {:.1?} refine {:.1?}",
                    points.len(),
                    t_scan,
                    t_tile - t_scan,
                    t_classify - t_tile,
                    t_round.elapsed() - t_classify
                );
            }
            if inserted == 0 {
                break 'outer (dt_tets, all_tilings, tet_region);
            }
            continue;
        }
        if trace {
            eprintln!("round {round}: {} patches with tiling deficit", deficits.len());
            for &pi in deficits.iter().take(8) {
                let want = patches[pi].area2.approx().abs();
                eprintln!(
                    "  patch {pi}: rel deficit {:.3e} ({} tiles)",
                    (tile_area[pi] - want) / want,
                    all_tilings[pi].len()
                );
            }
        }

        // 3. Repair: Steiner points where tet edges pierce a deficient
        // patch. ONE point per patch per round: every insertion changes the
        // DT, so batch insertions computed against a stale DT mostly land
        // redundantly and feed encroachment cascades.
        // The piercing search needs ALL tet edges (built lazily: deficits
        // only exist in the few recovery rounds, not during refinement).
        let mut all_edges: DSet<(usize, usize)> = DSet::default();
        for t in &dt_tets {
            for i in 0..4 {
                for j in i + 1..4 {
                    all_edges.insert(sorted2(t[i], t[j]));
                }
            }
        }
        let mut inserted = 0;
        let mut new_pts: Vec<([f64; 3], usize, (usize, usize))> = Vec::new();
        for &pi in &deficits {
            let patch = &patches[pi];
            for &(a, b) in &all_edges {
                let sa = plane_sign(patch, &points, a);
                let sb = plane_sign(patch, &points, b);
                if sa.combine(sb) != Sign::Negative {
                    continue;
                }
                // f64 crossing parameter from approximate plane distances.
                let pl = patch.plane.map(|i| points[i]);
                let n = {
                    let u: [f64; 3] = std::array::from_fn(|k| pl[1][k] - pl[0][k]);
                    let v: [f64; 3] = std::array::from_fn(|k| pl[2][k] - pl[0][k]);
                    [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ]
                };
                let dist = |p: [f64; 3]| -> f64 {
                    (0..3).map(|k| n[k] * (p[k] - pl[0][k])).sum()
                };
                let (da, db) = (dist(points[a]), dist(points[b]));
                let t = da / (da - db);
                if !t.is_finite() {
                    continue;
                }
                let x: [f64; 3] =
                    std::array::from_fn(|k| points[a][k] + t * (points[b][k] - points[a][k]));
                if strictly_inside_patch(pi, patch, &points, x) {
                    new_pts.push((x, pi, (a, b)));
                }
            }
        }
        // Piercings into UNCOVERED area first: an edge crossing the plane
        // underneath a hovering terrain point pierces inside already-covered
        // area; repairing it just re-splits covered tiles (and can spawn the
        // next such edge), starving the actual hole under the
        // one-insertion-per-patch policy.
        let covered = |pi: usize, x: [f64; 3]| -> bool {
            let (u, v) = Tri2Grid::axis_uv(patches[pi].axis);
            let q = [x[u], x[v]];
            all_tilings[pi].iter().any(|f| {
                let p: [[f64; 2]; 3] =
                    std::array::from_fn(|i| [points[f[i]][u], points[f[i]][v]]);
                let s = |a: [f64; 2], b: [f64; 2]| -> f64 {
                    (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                };
                let (d0, d1, d2) = (s(p[0], p[1]), s(p[1], p[2]), s(p[2], p[0]));
                (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
                    || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
            })
        };
        new_pts.sort_by_key(|&(x, pi, _)| covered(pi, x));
        // Batch repair with a spatial gate: candidates are computed against
        // the same (stale) DT, so two nearby candidates would land
        // redundantly and feed encroachment cascades. But candidates far
        // apart repair independent spots; gating on the local target size
        // admits them all in ONE round. The old one-per-patch-per-round trickle
        // lost the race against sizing refinement re-piercing a coarse patch
        // squeezed between fine volume clouds (trace face_maxh starved the
        // substrate-top tiling into the stagnation guard).
        let mut placed: Vec<[f64; 3]> = Vec::new();
        let gate_ok = |placed: &[[f64; 3]], x: [f64; 3], r: f64| -> bool {
            placed.iter().all(|p| {
                (0..3).map(|k| (p[k] - x[k]).powi(2)).sum::<f64>() > r * r
            })
        };
        for (x, pi, (a, b)) in new_pts {
            if placed.len() >= 1024 {
                break;
            }
            let r_gate = 0.4 * child_h(
                x,
                &[a, b],
                &points,
                point_h,
                params.grading,
                patch_h_init[pi],
                &params.size_points,
            );
            if !gate_ok(&placed, x, r_gate) {
                continue;
            }
            // A piercing point may coincide with (or sit within rounding of)
            // an existing vertex, e.g. a tet edge passing exactly through a
            // crease midpoint. Inserting it would create a duplicate that
            // poisons the triangulation; skip it, other piercings progress.
            if point_index.contains_key(&x.map(|x| (x + 0.0).to_bits())) {
                continue;
            }
            if points.iter().any(|q| {
                (0..3).all(|k| (q[k] - x[k]).abs() < 1e-12 * scale)
            }) {
                continue;
            }
            // A piercing point within rounding distance of a crease would
            // make that crease unrecoverable by midpoint splitting. Snap it
            // onto the crease and split the chain there instead (the gmsh
            // recipe for near-boundary facet Steiner points).
            let mut snapped: Option<((usize, usize), [f64; 3])> = None;
            for &(a, b) in creases.iter() {
                let d: [f64; 3] = std::array::from_fn(|k| points[b][k] - points[a][k]);
                let w: [f64; 3] = std::array::from_fn(|k| x[k] - points[a][k]);
                let dd: f64 = (0..3).map(|k| d[k] * d[k]).sum();
                let t = (0..3).map(|k| w[k] * d[k]).sum::<f64>() / dd;
                if !(1e-6..=1.0 - 1e-6).contains(&t) {
                    continue;
                }
                let dist = (0..3)
                    .map(|k| (w[k] - t * d[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < 1e-9 * scale {
                    let xs: [f64; 3] =
                        std::array::from_fn(|k| points[a][k] + t * d[k]);
                    snapped = Some(((a, b), xs));
                    break;
                }
            }
            if let Some(((a, b), xs)) = snapped {
                let g = match point_index.get(&xs.map(|x| (x + 0.0).to_bits())) {
                    Some(&g) => g,
                    None => {
                        let g = points.len();
                        points.push(xs);
                        point_h.push(child_h(
                            xs,
                            &[a, b],
                            &points,
                            point_h,
                            params.grading,
                            f64::INFINITY,
                    &params.size_points,
                ));
                        builder.insert(xs);
                        point_index.insert(xs.map(|x| (x + 0.0).to_bits()), g);
                        on_patch.push(DSet::default());
                        g
                    }
                };
                if g != a && g != b {
                    let marks = crease_marks.remove(&(a, b)).expect("known crease");
                    on_patch[g].extend(marks.iter().copied());
                    on_patch[g].insert(pi);
                    creases.retain(|&e| e != (a, b));
                    push_crease_child(&mut creases, &mut crease_marks, sorted2(a, g), &marks);
                    push_crease_child(&mut creases, &mut crease_marks, sorted2(g, b), &marks);
                    inserted += 1;
                    placed.push(xs);
                }
                continue;
            }
            let g = points.len();
            points.push(x);
            point_h.push(child_h(
                x,
                &[a, b],
                &points,
                point_h,
                params.grading,
                patch_h_init[pi],
                    &params.size_points,
                ));
            builder.insert(x);
            point_index.insert(x.map(|x| (x + 0.0).to_bits()), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            inserted += 1;
            placed.push(x);
        }
        if trace {
            eprintln!("round {round}: inserted {inserted} repair points, {} total", points.len());
        }
        if inserted == 0 {
            // No piercing edge to repair with: the missing area is covered
            // by faces whose vertices carry the wrong patch marks (a
            // bookkeeping corner the tessellation lottery can produce on
            // curved patches). Give those patches up like the stagnation
            // guard does; panicking would take the whole library down.
            for &pi in &deficits {
                eprintln!(
                    "rapidmesh: giving up on patch {pi} tiling (no piercing edges); geometry near this patch needs review"
                );
                if std::env::var_os("RAPIDMESH_HOLE_TRACE").is_some() {
                    // Which filter rejects the faces that geometrically
                    // cover this patch?
                    let patch = &patches[pi];
                    let mut diag: DMap<&'static str, usize> = DMap::default();
                    for (f, owners) in &dt_faces {
                        let c: [f64; 3] = std::array::from_fn(|k| {
                            (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                        });
                        if !inside_patch(pi, patch, &points, c) {
                            continue;
                        }
                        let marked =
                            f.iter().all(|&v| on_patch[v].contains(&pi));
                        let s0 = tet_plane_side(patch, &points, dt_tets[owners[0] as usize]);
                        let s_other = if owners.len() == 2 {
                            tet_plane_side(patch, &points, dt_tets[owners[1] as usize])
                        } else {
                            match hull_super.get(f) {
                                Some(&sc) => plane_side(
                                    patch,
                                    &points,
                                    [points[f[0]], points[f[1]], points[f[2]], sc],
                                ),
                                None => s0.flip(),
                            }
                        };
                        let key = match (marked, s0 != s_other) {
                            (true, true) => "counted",
                            (true, false) => "side-reject",
                            (false, true) => "marks-reject",
                            (false, false) => "both-reject",
                        };
                        *diag.entry(key).or_default() += 1;
                    }
                    let mut dv: Vec<(&str, usize)> = diag.into_iter().collect();
                    dv.sort_unstable();
                    eprintln!("  patch {pi} containment-passing faces: {dv:?}");
                    let pl = patch.plane.map(|i| points[i]);
                    let eu: [f64; 3] = std::array::from_fn(|k| pl[1][k] - pl[0][k]);
                    let ev: [f64; 3] = std::array::from_fn(|k| pl[2][k] - pl[0][k]);
                    let nrm = [
                        eu[1] * ev[2] - eu[2] * ev[1],
                        eu[2] * ev[0] - eu[0] * ev[2],
                        eu[0] * ev[1] - eu[1] * ev[0],
                    ];
                    let nlen = nrm.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let dist = |v: usize| -> f64 {
                        (0..3)
                            .map(|k| nrm[k] * (points[v][k] - pl[0][k]))
                            .sum::<f64>()
                            / nlen
                    };
                    let want = patch.area2.approx().abs();
                    eprintln!(
                        "  patch {pi}: want {want:.6e} got {:.6e}, {} tiles",
                        tile_area[pi],
                        all_tilings[pi].len()
                    );
                    let (au, av) = Tri2Grid::axis_uv(patch.axis);
                    for f in &all_tilings[pi] {
                        let ds: Vec<f64> = f.iter().map(|&v| dist(v)).collect();
                        if ds.iter().any(|d| d.abs() > 0.0) {
                            let (a, b, c) = (points[f[0]], points[f[1]], points[f[2]]);
                            let area = ((b[au] - a[au]) * (c[av] - a[av])
                                - (b[av] - a[av]) * (c[au] - a[au]))
                                .abs();
                            eprintln!("  off-plane tile {f:?} dists {ds:?} area {area:.3e}");
                        }
                    }
                }
                abandoned.insert(pi);
            }
        }
    };

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
    // Regions come from the last round's classification.
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

    let mut abandoned_patches: Vec<u32> = abandoned.iter().map(|&pi| pi as u32).collect();
    abandoned_patches.sort_unstable();

    TetMesh {
        points,
        tets: kept_tets,
        tet_regions,
        faces: out_faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches,
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

/// Splits the crease sub-edge `key` at its midpoint (reusing an exactly
/// coincident existing vertex if present). Returns false if the chain entry
/// vanished (already split).
#[allow(clippy::too_many_arguments)]
fn split_crease_midpoint(
    key: (usize, usize),
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    grading: f64,
    size_points: &[([f64; 3], f64)],
    builder: &mut DelaunayBuilder,
    point_index: &mut DMap<[u64; 3], usize>,
    on_patch: &mut Vec<DSet<usize>>,
    creases: &mut Vec<(usize, usize)>,
    crease_marks: &mut DMap<(usize, usize), DSet<usize>>,
) -> bool {
    let Some(marks) = crease_marks.remove(&key) else {
        return false;
    };
    let (a, b) = key;
    let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
    let g = match point_index.get(&m.map(|x| (x + 0.0).to_bits())) {
        Some(&g) => g,
        None => {
            let g = points.len();
            if std::env::var_os("RAPIDMESH_INS_TRACE").is_some() {
                eprintln!("ins crease v{g} {m:?} (of {a},{b})");
            }
            points.push(m);
            point_h.push(child_h(
                m,
                &[a, b],
                points,
                point_h,
                grading,
                f64::INFINITY,
                size_points,
            ));
            builder.insert(m);
            point_index.insert(m.map(|x| (x + 0.0).to_bits()), g);
            on_patch.push(DSet::default());
            g
        }
    };
    on_patch[g].extend(marks.iter().copied());
    creases.retain(|&e| e != key);
    push_crease_child(creases, crease_marks, (a.min(g), a.max(g)), &marks);
    push_crease_child(creases, crease_marks, (g.min(b), g.max(b)), &marks);
    true
}

/// Per-patch 2D index over member triangles (in the patch's projection
/// axis): point-in-patch queries on large coplanar patches (tessellated flat
/// CAD faces hold thousands of members) would otherwise scan every member
/// with exact predicates, once per tile per round. Members are static for
/// the whole meshing run, so the grids are built once.
struct Tri2Grid {
    cell: f64,
    origin: [f64; 2],
    map: DMap<[i64; 2], Vec<u32>>,
    /// Members spanning more than MAX_SPAN cells per axis.
    large: Vec<u32>,
}

impl Tri2Grid {
    const MAX_SPAN: i64 = 4;

    fn axis_uv(axis: Axis) -> (usize, usize) {
        match axis {
            Axis::X => (1, 2),
            Axis::Y => (2, 0),
            Axis::Z => (0, 1),
        }
    }

    fn build(points: &[[f64; 3]], members: &[[usize; 3]], axis: Axis) -> Tri2Grid {
        let (u, v) = Tri2Grid::axis_uv(axis);
        let bbox = |f: &[usize; 3]| -> ([f64; 2], [f64; 2]) {
            let mut lo = [f64::MAX; 2];
            let mut hi = [f64::MIN; 2];
            for &w in f {
                let q = [points[w][u], points[w][v]];
                for k in 0..2 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            (lo, hi)
        };
        // Cell size from the median member extent.
        let mut ext: Vec<f64> = members
            .iter()
            .map(|f| {
                let (lo, hi) = bbox(f);
                (hi[0] - lo[0]).max(hi[1] - lo[1])
            })
            .collect();
        ext.sort_by(f64::total_cmp);
        let cell = ext
            .get(ext.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(f64::MIN_POSITIVE);
        let origin = members
            .first()
            .map(|f| {
                let (lo, _) = bbox(f);
                lo
            })
            .unwrap_or([0.0; 2]);
        let mut grid = Tri2Grid { cell, origin, map: DMap::default(), large: Vec::new() };
        for (mi, f) in members.iter().enumerate() {
            let (lo, hi) = bbox(f);
            let clo = grid.cell_of(lo);
            let chi = grid.cell_of(hi);
            if (0..2).any(|k| chi[k] - clo[k] >= Tri2Grid::MAX_SPAN) {
                grid.large.push(mi as u32);
                continue;
            }
            for x in clo[0]..=chi[0] {
                for y in clo[1]..=chi[1] {
                    grid.map.entry([x, y]).or_default().push(mi as u32);
                }
            }
        }
        grid
    }

    fn cell_of(&self, q: [f64; 2]) -> [i64; 2] {
        std::array::from_fn(|k| ((q[k] - self.origin[k]) / self.cell).floor() as i64)
    }

    /// Member candidates whose grid cells contain the query point.
    fn candidates(&self, q: [f64; 2]) -> impl Iterator<Item = u32> + '_ {
        self.map
            .get(&self.cell_of(q))
            .into_iter()
            .flatten()
            .copied()
            .chain(self.large.iter().copied())
    }
}

/// A vetoed quality candidate is re-attempted at most this many times after
/// neighborhood changes (its verdict rarely changes beyond that, and
/// unbounded retries re-pay the insertion cavity every round).
const QUALITY_RETRY_LIMIT: u8 = 3;

/// Exact closed point-in-patch test (inside or on the boundary of some
/// member), grid-indexed. Points exactly on internal member edges count as
/// inside: the member triangulation is bookkeeping, not geometry.
fn patch_inside_closed(
    grid: &Tri2Grid,
    patch: &Patch,
    points: &[[f64; 3]],
    q: [f64; 3],
) -> bool {
    let qp = Point3::Explicit(q);
    let (u, v) = Tri2Grid::axis_uv(patch.axis);
    grid.candidates([q[u], q[v]]).any(|mi| {
        let f = patch.members[mi as usize];
        Tri::new(points[f[0]], points[f[1]], points[f[2]]).contains_coplanar(
            &qp,
            patch.axis,
            patch.orientation,
        )
    })
}

/// Sizing splits trigger above this multiple of the local target h. Splits
/// roughly halve lengths, so triggering exactly at h lands edges at h/2 and
/// over-refines about twofold against meshers that treat h as a target
/// (gmsh, tetgen); triggering at 1.2 h centers the result near h and leaves
/// the optimizer headroom inside the documented 1.5 h max-edge contract.
const OVERSIZE_FACTOR: f64 = 1.3;

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

/// Queue-driven sizing/quality refinement on a conforming state; returns
/// the number of insertions (0 = all targets met). Shewchuk priority:
/// oversized creases, then oversized patch tiles, then oversized or
/// poor-quality tets (circumcenters, with encroachment redirected to the
/// boundary).
///
/// Everything the criteria and guards need (per-tet regions, the live tile
/// set, encroachment balls) is maintained INCREMENTALLY through each
/// insertion's cavity deltas, so one call refines to a fixpoint. The
/// caller's global rescans (tiling check, flood-fill classification) then
/// run once per CALL as verification instead of once per insertion batch,
/// which is what made surface models pay O(rounds * tets).
fn ins_trace(path: &str, g: usize, pos: [f64; 3]) {
    if std::env::var_os("RAPIDMESH_INS_TRACE").is_some() {
        eprintln!("ins {path} v{g} {pos:?}");
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_queue(
    params: &MeshParams,
    patches: &[Patch],
    patch_grids: &[Tri2Grid],
    slot_tets: &[(u32, [usize; 4])],
    tilings: &[Vec<[usize; 3]>],
    tet_region: &[u32],
    tried: &mut DMap<[usize; 4], u8>,
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    builder: &mut DelaunayBuilder,
    point_index: &mut DMap<[u64; 3], usize>,
    on_patch: &mut Vec<DSet<usize>>,
    creases: &mut Vec<(usize, usize)>,
    crease_marks: &mut DMap<(usize, usize), DSet<usize>>,
) -> usize {
    use std::collections::VecDeque;
    let sized = params.maxh.is_finite() || !params.region_maxh.is_empty();
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
    let patch_h: Vec<f64> = patches
        .iter()
        .map(|p| {
            h_of_region(p.regions[0].0)
                .min(h_of_region(p.regions[1].0))
                .min(h_of_face(p.face_tag))
        })
        .collect();
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
    // The live tile set (face -> patch), seeded from the caller's tiling
    // scan and the keep-set of every guarded insert.
    let mut tile_map: DMap<[usize; 3], usize> = DMap::default();
    for (pi, tiles) in tilings.iter().enumerate() {
        for f in tiles {
            tile_map.insert(sorted3(*f), pi);
        }
    }

    // Work queues, seeded in deterministic Vec order; entries re-verify on
    // pop (slots are reused, tiles retile, creases split).
    let mut crease_q: VecDeque<(usize, usize)> = creases.iter().copied().collect();
    let mut tile_q: VecDeque<[usize; 3]> = tilings
        .iter()
        .flat_map(|ts| ts.iter().map(|f| sorted3(*f)))
        .collect();
    let mut tet_q: VecDeque<(u32, [usize; 4])> = slot_tets.iter().copied().collect();

    // Encroachment helper: the crease sub-edge whose diametral ball contains
    // x, if any. Grid-indexed over the creases at entry; creases split
    // DURING the call are the overlay tail of the vec, scanned linearly, and
    // dead snapshot entries are filtered via crease_marks (the canonical
    // live set).
    let crease_snapshot: Vec<(usize, usize)> = creases.clone();
    let crease_grid = BallGrid::build(
        crease_snapshot
            .iter()
            .map(|&(a, b)| {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                (m, 0.5 * dist2(points[a], points[b]).sqrt())
            })
            .collect(),
    );
    let encroached_crease = |x: [f64; 3],
                             points: &[[f64; 3]],
                             creases: &[(usize, usize)],
                             live: &DMap<(usize, usize), DSet<usize>>|
     -> Option<(usize, usize)> {
        if let Some(i) =
            crease_grid.first_containing(x, |i| live.contains_key(&crease_snapshot[i]))
        {
            return Some(crease_snapshot[i]);
        }
        creases[crease_snapshot.len().min(creases.len())..]
            .iter()
            .copied()
            .find(|&(a, b)| {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                dist2(x, m) < 0.25 * dist2(points[a], points[b])
            })
    };

    // Tile circumballs for circumcenter redirection: a snapshot grid plus a
    // linearly scanned overlay of tiles created during the call; both are
    // live-checked against tile_map.
    let mut tile_balls: Vec<([f64; 3], usize, [usize; 3])> = Vec::new();
    let mut tile_ball_geo: Vec<([f64; 3], f64)> = Vec::new();
    for (pi, tiles) in tilings.iter().enumerate() {
        for f in tiles {
            if let Some((tc, tr)) = tri_circumcenter(points[f[0]], points[f[1]], points[f[2]]) {
                tile_balls.push((tc, pi, *f));
                tile_ball_geo.push((tc, tr));
            }
        }
    }
    let tile_grid = BallGrid::build(tile_ball_geo);
    let mut tile_overlay: Vec<([f64; 3], f64, usize, [usize; 3])> = Vec::new();
    let encroached_tile = |x: [f64; 3],
                           tile_map: &DMap<[usize; 3], usize>,
                           overlay: &[([f64; 3], f64, usize, [usize; 3])]|
     -> Option<(usize, [f64; 3], [usize; 3])> {
        if let Some(i) = tile_grid.first_containing(x, |i| {
            let (_, pi, f) = tile_balls[i];
            tile_map.get(&sorted3(f)) == Some(&pi)
        }) {
            let (tc, pi, f) = tile_balls[i];
            return Some((pi, tc, f));
        }
        overlay
            .iter()
            .find(|(tc, tr, pi, f)| {
                dist2(x, *tc) < tr * tr && tile_map.get(&sorted3(*f)) == Some(pi)
            })
            .map(|&(tc, _, pi, f)| (pi, tc, f))
    };

    let cand_trace = std::env::var_os("RAPIDMESH_CAND_TRACE").is_some();
    if cand_trace {
        let mut counts: DMap<u32, usize> = DMap::default();
        for &(slot, _) in slot_tets {
            *counts.entry(region_by_slot[slot as usize]).or_default() += 1;
        }
        let mut cv: Vec<(u32, usize)> = counts.into_iter().collect();
        cv.sort_unstable();
        eprintln!("  queue seed regions: {cv:?}");
    }
    let mut n = 0usize;
    let mut guarded_ok = 0usize;
    let mut guarded_veto = 0usize;
    let mut n_crease_size = 0usize;
    let mut n_crease_redirect = 0usize;
    let mut n_tile = 0usize;
    let mut n_midpoint = 0usize;
    let mut n_cc_quality = 0usize;
    let mut n_cc_oversized = 0usize;

    // Applies one successful insert's cavity deltas to the bookkeeping.
    macro_rules! absorb {
        ($p:expr) => {
            absorb_insert_deltas(
                builder,
                $p,
                points,
                on_patch,
                patches,
                patch_grids,
                &mut region_by_slot,
                &mut tile_map,
                &mut tet_q,
                &mut tile_q,
                &mut tile_overlay,
            )
        };
    }
    // Splits a crease sub-edge, absorbing the insert (if the midpoint was
    // not an existing vertex) and enqueueing the children.
    macro_rules! split_crease {
        ($key:expr) => {{
            let pts_before = points.len();
            let creases_before = creases.len();
            let ok = split_crease_midpoint(
                $key,
                points,
                point_h,
                params.grading,
                &params.size_points,
                builder,
                point_index,
                on_patch,
                creases,
                crease_marks,
            );
            if points.len() > pts_before {
                absorb!(points.len() - 1);
            }
            let _ = creases_before;
            for ci in creases.len().saturating_sub(2)..creases.len() {
                crease_q.push_back(creases[ci]);
            }
            ok
        }};
    }
    // Longest-edge midpoint fallback for oversized tets whose circumcenter
    // is rejected: crease edges split their chain, surface edges inherit the
    // shared patch marks, interior edges insert guarded (an unguarded
    // midpoint near the boundary can eat tiles whose replacement faces then
    // route off-plane through the new point, holing the patch tiling).
    macro_rules! split_longest_edge {
        ($a:expr, $b:expr) => {{
            let (a, b) = ($a, $b);
            let key = (a.min(b), a.max(b));
            if crease_marks.contains_key(&key) {
                if split_crease!(key) {
                    n += 1;
                }
            } else {
                let m: [f64; 3] =
                    std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                let marks: DSet<usize> = on_patch[a]
                    .intersection(&on_patch[b])
                    .copied()
                    .collect();
                // Crease-encroachment redirect: a midpoint inside a crease
                // sub-edge's diametral ball (e.g. the chord of two points ON
                // the crease line) must split that crease instead -- a plain
                // insert there knocks the chain edge out of the DT behind
                // the bookkeeping's back.
                if let Some(ck) = encroached_crease(m, points, creases, crease_marks) {
                    if split_crease!(ck) {
                        n += 1;
                        n_crease_redirect += 1;
                    }
                } else if !point_index.contains_key(&m.map(|x| (x + 0.0).to_bits())) {
                    let admitted = if marks.is_empty() {
                        builder
                            .insert_guarded(m, 0.0, |rem| match rem {
                                crate::delaunay::Removal::Face(f) => {
                                    !tile_map.contains_key(&f)
                                }
                                crate::delaunay::Removal::Edge(a, b) => {
                                    !crease_marks.contains_key(&(a, b))
                                }
                            })
                            .is_some()
                    } else {
                        builder.insert(m);
                        true
                    };
                    if admitted {
                        let g = points.len();
                        if std::env::var_os("RAPIDMESH_INS_TRACE").is_some() {
                            let ma: Vec<usize> = on_patch[a].iter().copied().collect();
                            let mb: Vec<usize> = on_patch[b].iter().copied().collect();
                            eprintln!(
                                "ins midpoint v{g} {m:?} of v{a} {:?} {ma:?} / v{b} {:?} {mb:?}",
                                points[a], points[b]
                            );
                        }
                        points.push(m);
                        let cap = marks
                            .iter()
                            .map(|&pi| patch_h[pi])
                            .fold(f64::INFINITY, f64::min);
                        point_h.push(child_h(
                            m,
                            &[a, b],
                            points,
                            point_h,
                            params.grading,
                            cap,
                    &params.size_points,
                ));
                        point_index.insert(m.map(|x| (x + 0.0).to_bits()), g);
                        on_patch.push(marks);
                        absorb!(g);
                        n += 1;
                        n_midpoint += 1;
                    }
                }
            }
        }};
    }

    // ------------------------------------ the queue loop
    //
    // Insertion budget per call: on a COARSE mesh, cavities are huge, plain
    // on-constraint splits legitimately eat foreign tiles (the caller's
    // recovery rounds repair them), and region inheritance degrades with
    // every such breach -- so coarse calls return early for a fresh global
    // verification + classification, exactly like the old round-based
    // refinement. Once the mesh is dense, cavities are local, bookkeeping
    // stays sound, and one call refines to the fixpoint.
    let batch_cap = (slot_tets.len() / 4).max(512);
    loop {
        if n >= batch_cap {
            break;
        }
        if points.len() >= params.max_points {
            break;
        }
        // Priority 1: oversized creases.
        if let Some(key) = crease_q.pop_front() {
            let Some(marks) = crease_marks.get(&key) else {
                continue;
            };
            if !sized {
                continue;
            }
            // Mean of the endpoint targets: an edge leaving a fine feature
            // is allowed to GROW along the graded field; the min would clamp
            // it to the fine size over its whole length.
            let ch = marks
                .iter()
                .map(|&pi| patch_h[pi])
                .fold(f64::INFINITY, f64::min);
            let h = ch.min(0.5 * (point_h[key.0] + point_h[key.1]));
            if !(h.is_finite()
                && dist2(points[key.0], points[key.1])
                    > (OVERSIZE_FACTOR * h) * (OVERSIZE_FACTOR * h))
            {
                continue;
            }
            if split_crease!(key) {
                n += 1;
                n_crease_size += 1;
            }
            continue;
        }
        // Priority 2: oversized patch tiles.
        if let Some(key) = tile_q.pop_front() {
            let Some(&pi) = tile_map.get(&key) else {
                continue;
            };
            if !sized {
                continue;
            }
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
            if let Some(ck) = encroached_crease(cc, points, creases, crease_marks) {
                if split_crease!(ck) {
                    n += 1;
                }
                continue;
            }
            if point_index.contains_key(&cc.map(|x| (x + 0.0).to_bits())) {
                continue;
            }
            // A rim tile's circumcenter can land just OUTSIDE its patch
            // (e.g. behind the neighboring barrel facet of a tessellated
            // cylinder). Inserting it with this patch's mark poisons the
            // neighbor's tiling bookkeeping, so off-patch centers are
            // skipped; the size target is still reached through the
            // longest-edge fallback.
            if !patch_inside_closed(&patch_grids[pi], &patches[pi], points, cc) {
                continue;
            }
            let g = points.len();
            ins_trace("tile", g, cc);
            builder.insert(cc);
            points.push(cc);
            point_h.push(child_h(
                cc,
                &[key[0], key[1], key[2]],
                points,
                point_h,
                params.grading,
                patch_h[pi],
                    &params.size_points,
                ));
            point_index.insert(cc.map(|x| (x + 0.0).to_bits()), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            absorb!(g);
            n += 1;
            n_tile += 1;
            continue;
        }
        // Priority 3: tets, oversized or poor radius-edge quality.
        let Some((slot, tverts)) = tet_q.pop_front() else {
            break;
        };
        if builder.tet_at(slot) != Some(tverts) {
            continue;
        }
        // Region first: it caps every edge target (inherited growth must
        // not undercut the region's own h INSIDE a fine region).
        let region = region_by_slot[slot as usize];
        if region == 0 {
            continue;
        }
        let h_region = h_of_region(region);
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| points[tverts[k]]);
        let mut lmin2 = f64::MAX;
        let mut lmax2 = 0.0_f64;
        let mut longest = (tverts[0], tverts[1]);
        // Worst sizing violation over the edges, each judged against the
        // MEAN of its endpoint targets (edges leaving fine features may
        // grow along the graded field; the min would clamp them fine).
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
        // Near-degenerate (sliver) tets have no usable circumcenter. Their
        // QUALITY is the optimizer's job, but their oversized edges still
        // violate sizing and must split: cc-less candidates skip straight
        // to the longest-edge fallback.
        let cc_r = tet_circumcenter(p);
        let bad_quality = params.radius_edge_bound.is_finite()
            && cc_r.is_some_and(|(_, r)| r > params.radius_edge_bound * lmin2.sqrt());
        let maybe_oversized = sized && lmax2 > 0.0;
        if !maybe_oversized && !bad_quality {
            continue;
        }
        // Graded local target for the quality floor: the region cap
        // tightened by the tet-mean of the per-point field.
        let h = h_region
            .min((point_h[tverts[0]]
                + point_h[tverts[1]]
                + point_h[tverts[2]]
                + point_h[tverts[3]])
                / 4.0);
        let oversized = worst_ratio2 > OVERSIZE_FACTOR * OVERSIZE_FACTOR;
        // Quality splits stop once the local mesh is at (or below) target
        // size: boundary slivers have huge circumradii whose centers all
        // land far away and are the optimizer's job, not insertion's.
        let quality_allowed = bad_quality && (!h.is_finite() || lmin2.sqrt() > 0.2 * h);
        if !oversized && !quality_allowed {
            continue;
        }
        if !oversized && cc_r.is_none() {
            continue;
        }
        // A vetoed quality candidate is not retried on this queue pass (a
        // veto changes nothing); it re-enters with the caller's next
        // verification round, capped by QUALITY_RETRY_LIMIT.
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
            if let Some(ck) = encroached_crease(cc, points, creases, crease_marks) {
                // Pure Delaunay quality steps never touch the boundary.
                if !quality_only && split_crease!(ck) {
                    n += 1;
                    n_crease_redirect += 1;
                }
                break 'attempt;
            }
            if let Some((pi, tc, tf)) = encroached_tile(cc, &tile_map, &tile_overlay) {
                if quality_only {
                    break 'attempt;
                }
                if let Some(ck) = encroached_crease(tc, points, creases, crease_marks) {
                    if split_crease!(ck) {
                        n += 1;
                        n_crease_redirect += 1;
                    }
                    break 'attempt;
                }
                if point_index.contains_key(&tc.map(|x| (x + 0.0).to_bits())) {
                    break 'attempt;
                }
                if !patch_inside_closed(&patch_grids[pi], &patches[pi], points, tc) {
                    break 'attempt;
                }
                let g = points.len();
                ins_trace("redirect", g, tc);
                builder.insert(tc);
                points.push(tc);
                point_h.push(child_h(
                    tc,
                    &tf,
                    points,
                    point_h,
                    params.grading,
                    patch_h[pi],
                    &params.size_points,
                ));
                point_index.insert(tc.map(|x| (x + 0.0).to_bits()), g);
                let mut marks = DSet::default();
                marks.insert(pi);
                on_patch.push(marks);
                absorb!(g);
                n += 1;
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
            // ~0.61 lmin (equilateral), so their floor must be lower or
            // every healthy size split bails to the midpoint fallback,
            // which over-refines by halving edges instead of centering.
            let floor2 = if quality_only { lmin2 } else { 0.25 * lmin2 };
            let admitted = builder.insert_guarded(cc, floor2, |rem| match rem {
                crate::delaunay::Removal::Face(f) => !tile_map.contains_key(&f),
                crate::delaunay::Removal::Edge(a, b) => !crease_marks.contains_key(&(a, b)),
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
                cc,
                &tverts,
                points,
                point_h,
                params.grading,
                f64::INFINITY,
                    &params.size_points,
                ));
            point_index.insert(cc.map(|x| (x + 0.0).to_bits()), g);
            on_patch.push(DSet::default());
            absorb!(g);
            n += 1;
        }
        // No progress on an oversized tet via the circumcenter routes:
        // longest-edge midpoint fallback guarantees the size target.
        if oversized && n == before {
            let (a, b) = longest;
            split_longest_edge!(a, b);
        }
    }
    if cand_trace {
        let mut counts: DMap<u32, usize> = DMap::default();
        for (slot, &r) in region_by_slot.iter().enumerate() {
            if builder.tet_at(slot as u32).is_some() {
                *counts.entry(r).or_default() += 1;
            }
        }
        let mut cv: Vec<(u32, usize)> = counts.into_iter().collect();
        cv.sort_unstable();
        eprintln!("  queue end regions: {cv:?}");
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
/// on-patch insertions), tiles swallowed by the cavity leave the live tile
/// set, and the new local faces are (re)evaluated with the SAME
/// marks+containment+side tile rule the global tiling scan uses.
#[allow(clippy::too_many_arguments)]
fn absorb_insert_deltas(
    builder: &DelaunayBuilder,
    p_idx: usize,
    points: &[[f64; 3]],
    on_patch: &[DSet<usize>],
    patches: &[Patch],
    patch_grids: &[Tri2Grid],
    region_by_slot: &mut Vec<u32>,
    tile_map: &mut DMap<[usize; 3], usize>,
    tet_q: &mut std::collections::VecDeque<(u32, [usize; 4])>,
    tile_q: &mut std::collections::VecDeque<[usize; 3]>,
    tile_overlay: &mut Vec<([f64; 3], f64, usize, [usize; 3])>,
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
    // Tiles swallowed by the cavity (a removed super-corner tet still owns
    // one all-real face, e.g. a hull tile).
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
    // New and re-based local faces: evaluate tile candidacy.
    let mut seen: DSet<[usize; 3]> = DSet::default();
    for &nt in &created {
        let Some(tv) = builder.tet_at(nt) else {
            continue;
        };
        for (fi, fv) in TET_FACES.iter().enumerate() {
            let f = fv.map(|k| tv[k]);
            let key = sorted3(f);
            if !seen.insert(key) {
                continue;
            }
            let (s0, s1, s2) = (&on_patch[f[0]], &on_patch[f[1]], &on_patch[f[2]]);
            let mut tiled: Option<usize> = None;
            if !s0.is_empty() && !s1.is_empty() && !s2.is_empty() {
                for &pi in s0 {
                    if !s1.contains(&pi) || !s2.contains(&pi) {
                        continue;
                    }
                    let patch = &patches[pi];
                    let c: [f64; 3] = std::array::from_fn(|k| {
                        (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                    });
                    if !patch_inside_closed(&patch_grids[pi], patch, points, c) {
                        continue;
                    }
                    // Side rule (see the global tiling scan): the
                    // same-side reject applies only to the flat all-marked
                    // sliver complex; faces against distant tets count.
                    let opp_of = |tt: [usize; 4]| -> Option<usize> {
                        tt.iter().copied().find(|v| !f.contains(v))
                    };
                    let marked = |v: Option<usize>| -> bool {
                        v.is_some_and(|v| on_patch[v].contains(&pi))
                    };
                    let rejected = match builder.neighbor_at(nt, fi) {
                        Some(nb) => match builder.tet_at(nb) {
                            Some(tb) => {
                                marked(opp_of(tv))
                                    && marked(opp_of(tb))
                                    && tet_plane_side(patch, points, tv)
                                        == tet_plane_side(patch, points, tb)
                            }
                            None => match builder.super_corner(nb) {
                                Some(sc) => {
                                    marked(opp_of(tv))
                                        && tet_plane_side(patch, points, tv)
                                            == plane_side(
                                                patch,
                                                points,
                                                [
                                                    points[f[0]],
                                                    points[f[1]],
                                                    points[f[2]],
                                                    sc,
                                                ],
                                            )
                                }
                                None => false,
                            },
                        },
                        None => false,
                    };
                    if rejected {
                        continue;
                    }
                    tiled = Some(pi);
                    break;
                }
            }
            match (tiled, tile_map.get(&key).copied()) {
                (Some(pi), prev) => {
                    if prev != Some(pi) {
                        tile_map.insert(key, pi);
                    }
                    tile_q.push_back(key);
                    if let Some((tc, tr)) =
                        tri_circumcenter(points[f[0]], points[f[1]], points[f[2]])
                    {
                        tile_overlay.push((tc, tr, pi, f));
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
    /// Smallest dihedral angle in degrees (sliver indicator; the
    /// load-bearing metric for Nedelec conditioning).
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
        // Dihedral angle at each of the 6 edges: angle between the
        // projections of the two opposite vertices onto the plane normal to
        // the edge.
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
