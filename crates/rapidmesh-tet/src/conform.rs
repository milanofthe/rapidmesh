//! PLC types, region classification, and quality reporting for the tet mesher.
//!
//! `mesh_plc` / `mesh_plc_with` are the stable entry points; they delegate to
//! the CVT mesher in [`crate::cvt`]. The exact CSG arrangement (the `TaggedPlc`)
//! is produced upstream and meshed here into a conforming, region-tagged tet
//! mesh. This module owns the output types (`TetMesh`, `SurfaceFace`), the mesh
//! parameters (`MeshParams`), the coplanar-patch grouping reused for boundary
//! tagging, the flood-fill region classifier (revived for multi-region CVT),
//! and the quality statistics.

use rapidmesh_csg::classify::point_inside_solid;
use rapidmesh_csg::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind, TaggedPlc};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;

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
    /// Mesh vertices (PLC vertices plus interior/Steiner points).
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
    /// Patches the mesher gave up on. Kept for API stability (always empty).
    pub abandoned_patches: Vec<u32>,
    /// `points[..plc_points]` are the PLC's own vertices (the geometry);
    /// everything after is an interior point the mesher added.
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
        // (surface, smooth-id, face-tag, region-lo, region-hi)
        type FaceKey = (u32, u32, u32, u32, u32);
        let face_key = |sf: &SurfaceFace| -> FaceKey {
            let smooth = match self.surfaces[sf.surface as usize] {
                SurfaceKind::Plane => sf.patch,
                _ => u32::MAX,
            };
            let (r0, r1) = (sf.regions[0].0.min(sf.regions[1].0), sf.regions[0].0.max(sf.regions[1].0));
            (sf.surface, smooth, sf.face_tag.0, r0, r1)
        };
        // edge -> (incidence count, first face key seen, mixed-key flag)
        let mut edges: DMap<(usize, usize), (u32, FaceKey, bool)> = DMap::default();
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
pub(crate) struct Patch {
    /// Member facet (PLC triangle) indices.
    pub(crate) member_indices: Vec<usize>,
    pub(crate) face_tag: FaceTag,
    pub(crate) regions: [RegionTag; 2],
    /// Analytic surface of the members (index into the PLC surface table).
    pub(crate) surface: u32,
}

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// The four vertex-index triples spanning a tet's faces (unoriented).
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// Builds the maximal coplanar same-tag patches by union-find over facets
/// sharing an edge.
pub(crate) fn build_patches(plc: &TaggedPlc) -> Vec<Patch> {
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
    /// into coarser regions grade naturally.
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

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh without
/// sizing or quality refinement. Background (region 0) tets are dropped.
pub fn mesh_plc(plc: &TaggedPlc) -> TetMesh {
    mesh_plc_with(
        plc,
        &MeshParams {
            maxh: f64::INFINITY,
            radius_edge_bound: f64::INFINITY,
            max_points: usize::MAX,
            ..Default::default()
        },
    )
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh, refined to
/// the given sizing and quality targets (best effort under
/// `params.max_points`). Delegates to the CVT mesher.
pub fn mesh_plc_with(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    crate::cvt::mesh(plc, params)
}

/// Region of every tet by FLOOD FILL through shared faces: crossing a
/// constraint face flips to the face's other region, free faces keep it.
/// Parity ray casting runs once per CONNECTED COMPONENT.
///
/// Revived for multi-region CVT (WP5); the single-region WP3 path classifies
/// by centroid inside-test directly.
#[allow(dead_code)]
pub(crate) fn classify_tet_regions(
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
    let margin = 1e-6 * (0..3).map(|k| bbox.1[k] - bbox.0[k]).fold(1.0_f64, f64::max);
    let region_boxes: DMap<u32, rapidmesh_csg::TriBoxes> = region_bounds
        .iter()
        .map(|(&r, tris)| (r, rapidmesh_csg::TriBoxes::build(tris, margin)))
        .collect();

    let mut region_of: Vec<Option<u32>> = vec![None; tets.len()];
    let mut stack: Vec<usize> = Vec::new();
    for seed in 0..tets.len() {
        if region_of[seed].is_some() {
            continue;
        }
        let t = tets[seed];
        let c: [f64; 3] = std::array::from_fn(|k| {
            0.25 * (points[t[0]][k] + points[t[1]][k] + points[t[2]][k] + points[t[3]][k])
        });
        let rep = Point3::Explicit(c);
        let seed_region = region_ids
            .iter()
            .copied()
            .find(|r| point_inside_solid(&rep, c, &region_bounds[r], &region_boxes[r], bbox))
            .unwrap_or(0);
        region_of[seed] = Some(seed_region);
        stack.push(seed);
        while let Some(ti) = stack.pop() {
            let cur = region_of[ti].expect("set before push");
            let t = tets[ti];
            for fi in TET_FACES {
                let key = sorted3(fi.map(|k| t[k]));
                let next_region = match face_regions.get(&key) {
                    Some(&[a, b]) => {
                        if a == cur {
                            b
                        } else if b == cur {
                            a
                        } else {
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

/// Quality summary of a tet mesh, with WHERE the worst element is and a
/// per-region breakdown.
#[derive(Debug, Clone)]
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
    /// Index of the tet holding the smallest dihedral angle (`usize::MAX` for
    /// an empty mesh).
    pub worst_tet: usize,
    /// Centroid of the worst tet: where the worst sliver sits.
    pub worst_location: [f64; 3],
    /// Region tag of the worst tet.
    pub worst_region: u32,
    /// Per region, in ascending tag order: (region, min dihedral deg, tets).
    pub per_region: Vec<(u32, f64, usize)>,
}

/// Smallest dihedral angle (degrees) of one tet, or `f64::MAX` if degenerate.
fn tet_min_dihedral(p: [[f64; 3]; 4]) -> f64 {
    let mut m = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            let others: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
            let (a, b) = (p[i], p[j]);
            let tlen: f64 = (0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt();
            if tlen == 0.0 {
                continue;
            }
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
            let cosang = ((0..3).map(|k| u[k] * v[k]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
            m = m.min(cosang.acos().to_degrees());
        }
    }
    m
}

/// Computes quality statistics over all tets, tracking the worst element's
/// location/region and a per-region min-dihedral breakdown.
pub fn quality_stats(mesh: &TetMesh) -> QualityStats {
    let mut min_dihedral = f64::MAX;
    let mut worst_tet = usize::MAX;
    let mut max_re = 0.0_f64;
    let mut max_edge2 = 0.0_f64;
    let mut per_region: std::collections::BTreeMap<u32, (f64, usize)> =
        std::collections::BTreeMap::new();
    for (ti, t) in mesh.tets.iter().enumerate() {
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
        let md = tet_min_dihedral(p);
        if md < min_dihedral {
            min_dihedral = md;
            worst_tet = ti;
        }
        let region = mesh.tet_regions[ti].0;
        let e = per_region.entry(region).or_insert((f64::MAX, 0));
        e.0 = e.0.min(md);
        e.1 += 1;
    }
    let worst_location = if worst_tet != usize::MAX {
        let t = mesh.tets[worst_tet];
        std::array::from_fn(|k| (0..4).map(|c| mesh.points[t[c]][k]).sum::<f64>() / 4.0)
    } else {
        [0.0; 3]
    };
    let worst_region = if worst_tet != usize::MAX {
        mesh.tet_regions[worst_tet].0
    } else {
        0
    };
    QualityStats {
        n_tets: mesh.tets.len(),
        min_dihedral_deg: min_dihedral,
        max_radius_edge: max_re,
        max_edge: max_edge2.sqrt(),
        worst_tet,
        worst_location,
        worst_region,
        per_region: per_region.into_iter().map(|(r, (m, n))| (r, m, n)).collect(),
    }
}
