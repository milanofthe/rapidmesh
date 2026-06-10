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
) -> f64 {
    let mut h = cap;
    for &v in parents {
        let d = (0..3)
            .map(|k| (pos[k] - points[v][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        h = h.min(point_h[v] + grading * d);
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
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            radius_edge_bound: 2.0,
            max_points: 100_000,
            grading: 0.5,
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
    let patch_h_init: Vec<f64> = patches
        .iter()
        .map(|p| h_of_region_init(p.regions[0].0).min(h_of_region_init(p.regions[1].0)))
        .collect();
    let mut point_h: Vec<f64> = (0..points.len())
        .map(|v| {
            on_patch[v]
                .iter()
                .map(|&pi| patch_h_init[pi])
                .fold(params.maxh, f64::min)
        })
        .collect();
    let point_h = &mut point_h;

    let mut round = 0;
    let mut refine_round = 0;
    // Everything below this index existed before the previous refine step
    // (drives the retry test for once-vetoed quality candidates).
    let mut refine_seen_len = 0usize;
    // Patches whose tiling repair stagnated (see the stagnation guard).
    let mut abandoned: DSet<usize> = DSet::default();
    let mut patch_progress: DMap<usize, (f64, usize)> = DMap::default();
    let mut tried: DMap<[usize; 4], u8> = DMap::default();
    #[allow(clippy::type_complexity)]
    let (tets, patch_faces, tet_region): (Vec<[usize; 4]>, Vec<Vec<[usize; 3]>>, Vec<u32>) = 'outer: loop {
        round += 1;
        assert!(round <= 16384, "boundary recovery did not converge");

        let t_round = std::time::Instant::now();
        let dt_tets = builder.tets();
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
        let mut all_tilings: Vec<Vec<[usize; 3]>> = vec![Vec::new(); patches.len()];
        let mut tile_area: Vec<f64> = vec![0.0; patches.len()];
        let area2_f64 = |f: &[usize; 3], axis: Axis| -> f64 {
            let (u, v) = match axis {
                Axis::X => (1, 2),
                Axis::Y => (2, 0),
                Axis::Z => (0, 1),
            };
            let (a, b, c) = (points[f[0]], points[f[1]], points[f[2]]);
            ((b[u] - a[u]) * (c[v] - a[v]) - (b[v] - a[v]) * (c[u] - a[u])).abs()
        };
        for f in dt_faces.keys() {
            let (s0, s1, s2) = (&on_patch[f[0]], &on_patch[f[1]], &on_patch[f[2]]);
            if s0.is_empty() || s1.is_empty() || s2.is_empty() {
                continue;
            }
            for &pi in s0 {
                if !s1.contains(&pi) || !s2.contains(&pi) {
                    continue;
                }
                let patch = &patches[pi];
                let c: [f64; 3] = std::array::from_fn(|k| {
                    (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                });
                if !inside_patch(pi, patch, &points, c) {
                    continue;
                }
                tile_area[pi] += area2_f64(f, patch.axis);
                all_tilings[pi].push(*f);
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
            let inserted = refine_step(
                params,
                &patches,
                &patch_grids,
                &dt_tets,
                &all_tilings,
                &tet_region,
                &mut tried,
                refine_seen_len,
                &mut points,
                point_h,
                &mut builder,
                &mut point_index,
                &mut on_patch,
                &mut creases,
                &mut crease_marks,
            );
            refine_seen_len = points.len();
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
        let scale = (0..3).map(|k| bhi[k] - blo[k]).fold(1.0_f64, f64::max);
        // One successful insertion per deficient patch per round: every
        // insertion changes the DT, so further candidates computed against
        // the stale DT would mostly land redundantly and feed encroachment
        // cascades. Candidates are tried in order until one inserts.
        let mut patch_done: DSet<usize> = DSet::default();
        for (x, pi, (a, b)) in new_pts {
            if patch_done.contains(&pi) {
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
                    patch_done.insert(pi);
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
            ));
            builder.insert(x);
            point_index.insert(x.map(|x| (x + 0.0).to_bits()), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            inserted += 1;
            patch_done.insert(pi);
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
                abandoned.insert(pi);
            }
        }
    };

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

    TetMesh {
        points,
        tets: kept_tets,
        tet_regions,
        faces: out_faces,
        surfaces: plc.surfaces.clone(),
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
#[allow(clippy::too_many_arguments)]
fn split_crease_midpoint(
    key: (usize, usize),
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    grading: f64,
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
            points.push(m);
            point_h.push(child_h(m, &[a, b], points, point_h, grading, f64::INFINITY));
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

/// One sizing/quality refinement step on a conforming state. Returns the
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
const OVERSIZE_FACTOR: f64 = 1.2;

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

/// number of inserted points (0 = all targets met). Shewchuk phase order:
/// oversized creases, then oversized patch tiles, then oversized or
/// poor-quality tets (circumcenters, with encroachment redirected to the
/// boundary).
#[allow(clippy::too_many_arguments)]
fn refine_step(
    params: &MeshParams,
    patches: &[Patch],
    patch_grids: &[Tri2Grid],
    dt_tets: &[[usize; 4]],
    tilings: &[Vec<[usize; 3]>],
    tet_region: &[u32],
    tried: &mut DMap<[usize; 4], u8>,
    new_since: usize,
    points: &mut Vec<[f64; 3]>,
    point_h: &mut Vec<f64>,
    builder: &mut DelaunayBuilder,
    point_index: &mut DMap<[u64; 3], usize>,
    on_patch: &mut Vec<DSet<usize>>,
    creases: &mut Vec<(usize, usize)>,
    crease_marks: &mut DMap<(usize, usize), DSet<usize>>,
) -> usize {
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
    let patch_h: Vec<f64> = patches
        .iter()
        .map(|p| h_of_region(p.regions[0].0).min(h_of_region(p.regions[1].0)))
        .collect();
    let dist2 = |a: [f64; 3], b: [f64; 3]| -> f64 {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
    };

    // Phase 1: creases longer than maxh (disjoint midpoint splits batch
    // safely).
    if sized {
        let crease_h = |key: &(usize, usize)| -> f64 {
            crease_marks
                .get(key)
                .map(|marks| marks.iter().map(|&pi| patch_h[pi]).fold(f64::INFINITY, f64::min))
                .unwrap_or(f64::INFINITY)
        };
        let long: Vec<(usize, usize)> = creases
            .iter()
            .copied()
            .filter(|key| {
                let h = crease_h(key)
                    .min(point_h[key.0])
                    .min(point_h[key.1]);
                h.is_finite()
                    && dist2(points[key.0], points[key.1])
                        > (OVERSIZE_FACTOR * h) * (OVERSIZE_FACTOR * h)
            })
            .collect();
        if !long.is_empty() {
            let mut n = 0;
            for key in long {
                if split_crease_midpoint(
                    key,
                    points,
                    point_h,
                    params.grading,
                    builder,
                    point_index,
                    on_patch,
                    creases,
                    crease_marks,
                ) {
                    n += 1;
                }
            }
            if n > 0 {
                return n;
            }
        }
    }

    // Encroachment helper: the crease sub-edge whose diametral ball contains
    // x, if any. Grid-indexed over the creases at entry (linear scans are
    // quadratic on surface models); creases split DURING this step are the
    // small overlay tail of the vec, scanned linearly, and dead snapshot
    // entries are filtered via crease_marks (the canonical live set). Vec
    // order semantics (first live hit) are preserved: retain keeps survivor
    // order and children append.
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

    // Phase 2: oversized patch tiles. Batched: all oversized tile
    // circumcenters in one step, thinned by a spacing guard (insertions
    // computed against the stale DT are only redundant when they cluster,
    // and clustered points would create short edges anyway).
    if sized {
        let mut cands: Vec<([f64; 3], f64, usize, f64, [usize; 3])> = Vec::new();
        for (pi, tiles) in tilings.iter().enumerate() {
            for f in tiles {
                let h_t = patch_h[pi]
                    .min(point_h[f[0]])
                    .min(point_h[f[1]])
                    .min(point_h[f[2]]);
                if !h_t.is_finite() {
                    continue;
                }
                if let Some((cc, r)) = tri_circumcenter(points[f[0]], points[f[1]], points[f[2]])
                {
                    if r > 0.5 * OVERSIZE_FACTOR * h_t {
                        cands.push((cc, r, pi, h_t, *f));
                    }
                }
            }
        }
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut n = 0;
        let mut placed: Vec<[f64; 3]> = Vec::new();
        for (cc, _r, pi, h_t, f) in cands {
            let spacing = 0.5 * h_t;
            if placed.iter().any(|q| dist2(*q, cc) < spacing * spacing) {
                continue;
            }
            if let Some(key) = encroached_crease(cc, points, creases, crease_marks) {
                if split_crease_midpoint(
                    key,
                    points,
                    point_h,
                    params.grading,
                    builder,
                    point_index,
                    on_patch,
                    creases,
                    crease_marks,
                ) {
                    n += 1;
                    placed.push(cc);
                }
                continue;
            }
            if point_index.contains_key(&cc.map(|x| (x + 0.0).to_bits())) {
                continue;
            }
            // A rim tile's circumcenter can land just OUTSIDE its patch
            // (e.g. behind the neighboring barrel facet of a tessellated
            // cylinder). Inserting it with this patch's mark poisons the
            // neighbor's tiling bookkeeping (uncountable tiles, permanent
            // deficit), so off-patch centers are skipped; the size target
            // is still reached through phase 3's longest-edge fallback.
            if !patch_inside_closed(&patch_grids[pi], &patches[pi], points, cc) {
                continue;
            }
            let g = points.len();
            points.push(cc);
            point_h.push(child_h(cc, &f, points, point_h, params.grading, patch_h[pi]));
            builder.insert(cc);
            point_index.insert(cc.map(|x| (x + 0.0).to_bits()), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            n += 1;
            placed.push(cc);
        }
        if n > 0 {
            return n;
        }
    }

    // Phase 3: tets, oversized or poor radius-edge quality. Insert
    // circumcenters unless they encroach the boundary (then split that
    // instead). Batched with a spacing guard against near-duplicate
    // circumcenters from neighboring tets.
    // Constraint sets for guarded quality insertions: a quality insertion
    // must never remove a recovered tile face or crease edge (otherwise
    // recovery re-splits the boundary ever finer and refinement spirals,
    // which is what made organic surface imports non-terminating).
    let crease_set: DSet<(usize, usize)> = creases
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let tile_set: DSet<[usize; 3]> = tilings
        .iter()
        .flatten()
        .map(|f| {
            let mut t = *f;
            t.sort_unstable();
            t
        })
        .collect();
    // Tile circumcircles, precomputed once for the per-candidate
    // encroachment checks.
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
    // (cc, circumradius, spacing, lmin2, tet verts, longest edge if oversized)
    #[allow(clippy::type_complexity)]
    let mut candidates: Vec<([f64; 3], f64, f64, f64, [usize; 4], Option<(usize, usize)>)> =
        Vec::new();
    for (ti, t) in dt_tets.iter().enumerate() {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| points[t[k]]);
        let mut lmin2 = f64::MAX;
        let mut lmax2 = 0.0_f64;
        let mut longest = (t[0], t[1]);
        for i in 0..4 {
            for j in i + 1..4 {
                let d = dist2(p[i], p[j]);
                lmin2 = lmin2.min(d);
                if d > lmax2 {
                    lmax2 = d;
                    longest = (t[i], t[j]);
                }
            }
        }
        // Near-degenerate (sliver) tets have no usable circumcenter; their
        // long edges are tolerated (sizing is a target, and forcing midpoint
        // splits into slivers spawns new slivers without converging).
        let Some((cc, r)) = tet_circumcenter(p) else {
            continue;
        };
        let bad_quality = params.radius_edge_bound.is_finite()
            && r > params.radius_edge_bound * lmin2.sqrt();
        let maybe_oversized = sized && lmax2 > 0.0;
        if !maybe_oversized && !bad_quality {
            continue;
        }
        // Only refine tets that belong to a region (the DT also fills the
        // gap between the convex hull and the domain); the region also
        // selects the local size target.
        let region = tet_region[ti];
        if region == 0 {
            continue;
        }
        // Graded local target: the region cap tightened by the per-point
        // size field (a tet touching a fine feature's envelope refines even
        // deep inside a coarse region; sizes then grow with distance).
        let h = h_of_region(region)
            .min(point_h[t[0]])
            .min(point_h[t[1]])
            .min(point_h[t[2]])
            .min(point_h[t[3]]);
        let oversized = h.is_finite() && lmax2 > (OVERSIZE_FACTOR * h) * (OVERSIZE_FACTOR * h);
        // Quality splits stop once the local mesh is at (or below) target
        // size: boundary slivers have huge circumradii whose centers all
        // land far away (e.g. a dense blob in a sphere's middle) and are the
        // optimizer's job, not insertion's.
        let quality_allowed = bad_quality
            && (!h.is_finite() || lmin2.sqrt() > 0.2 * h);
        if !oversized && !quality_allowed {
            continue;
        }
        // Quality-only candidates do not blindly retry every round (most
        // would re-run their veto verbatim), but they DO retry once their
        // neighborhood changed: a point inserted since the last attempt
        // within twice the circumradius of the candidate's center can change
        // the insertion cavity and thus the verdict. (Spiraling through
        // retries is structurally impossible since guarded insertion:
        // a successful insert always destroys its candidate tet, and a veto
        // inserts nothing.) Oversized tets must shrink, so they always
        // retry.
        if !oversized {
            let mut key = *t;
            key.sort_unstable();
            let attempts = tried.entry(key).or_insert(0);
            if *attempts > 0 {
                // Retry only when the neighborhood changed, and only a few
                // times: candidates with huge circumradii match every new
                // point and would otherwise re-pay their (vetoed) insertion
                // cavity every single round.
                let changed = *attempts <= QUALITY_RETRY_LIMIT
                    && points[new_since.min(points.len())..]
                        .iter()
                        .any(|q| dist2(cc, *q) < 4.0 * r * r);
                if !changed {
                    continue;
                }
            }
            *attempts = attempts.saturating_add(1);
        }
        // Batch spacing scales with the LOCAL TARGET SIZE: an insertion
        // fixes its whole neighborhood, so stale-DT candidates closer than
        // about half the target are redundant and would over-refine.
        let spacing = if h.is_finite() {
            (0.5 * lmin2.sqrt()).max(0.5 * h)
        } else {
            0.5 * lmin2.sqrt()
        };
        candidates.push((cc, r, spacing, lmin2, *t, oversized.then_some(longest)));
    }
    if std::env::var_os("RAPIDMESH_CAND_TRACE").is_some() && !candidates.is_empty() {
        let mut ls: Vec<f64> = candidates.iter().map(|c| c.2).collect();
        ls.sort_by(f64::total_cmp);
        eprintln!(
            "  cands {}: spacing min {:.2e} med {:.2e} max {:.2e}",
            ls.len(),
            ls[0],
            ls[ls.len() / 2],
            ls[ls.len() - 1]
        );
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let cand_trace = std::env::var_os("RAPIDMESH_CAND_TRACE").is_some();
    let t_cands = std::time::Instant::now();
    let n_cands = candidates.len();
    let mut guarded_ok = 0usize;
    let mut guarded_veto = 0usize;
    let mut t_guarded = std::time::Duration::ZERO;
    let mut t_encroach = std::time::Duration::ZERO;
    let mut n = 0;
    let mut placed: Vec<[f64; 3]> = Vec::new();
    // Longest-edge midpoint fallback for oversized tets whose circumcenter
    // is rejected (outside the domain, duplicate): crease edges split their
    // chain, surface edges inherit the shared patch marks, interior edges
    // insert plainly.
    macro_rules! split_longest_edge {
        ($a:expr, $b:expr) => {{
            let (a, b) = ($a, $b);
            let key = (a.min(b), a.max(b));
            if crease_marks.contains_key(&key) {
                if split_crease_midpoint(
                    key,
                    points,
                    point_h,
                    params.grading,
                    builder,
                    point_index,
                    on_patch,
                    creases,
                    crease_marks,
                ) {
                    n += 1;
                }
            } else {
                let m: [f64; 3] =
                    std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                if !point_index.contains_key(&m.map(|x| (x + 0.0).to_bits())) {
                    let g = points.len();
                    points.push(m);
                    point_h.push(child_h(
                        m,
                        &[a, b],
                        points,
                        point_h,
                        params.grading,
                        f64::INFINITY,
                    ));
                    builder.insert(m);
                    point_index.insert(m.map(|x| (x + 0.0).to_bits()), g);
                    let marks: DSet<usize> = on_patch[a]
                        .intersection(&on_patch[b])
                        .copied()
                        .collect();
                    on_patch.push(marks);
                    n += 1;
                    placed.push(m);
                }
            }
        }};
    }
    for (cc, _r, spacing, lmin2, tverts, longest) in candidates {
        if n >= 512 {
            break;
        }
        if placed.iter().any(|q| dist2(*q, cc) < spacing * spacing) {
            continue;
        }
        let before = n;
        let quality_only = longest.is_none();
        'attempt: {
            if quality_only {
                // Pure Delaunay-refinement step: only insert circumcenters
                // that encroach nothing (boundary slivers are the optimize
                // pass's job, not insertion's).
                let te = std::time::Instant::now();
                let hit = encroached_crease(cc, points, creases, crease_marks).is_some();
                t_encroach += te.elapsed();
                if hit {
                    break 'attempt;
                }
            }
            if let Some(key) = encroached_crease(cc, points, creases, crease_marks) {
                if split_crease_midpoint(
                    key,
                    points,
                    point_h,
                    params.grading,
                    builder,
                    point_index,
                    on_patch,
                    creases,
                    crease_marks,
                ) {
                    n += 1;
                    placed.push(cc);
                }
                break 'attempt;
            }
            // Encroached patch tile: insert that tile's circumcenter instead.
            let tile_hit: Option<(usize, [f64; 3], [usize; 3])> = tile_grid
                .first_containing(cc, |_| true)
                .map(|i| (tile_balls[i].1, tile_balls[i].0, tile_balls[i].2));
            if let Some((pi, tc, tile_f)) = tile_hit {
                if quality_only {
                    break 'attempt; // no boundary interaction for quality steps
                }
                if let Some(key) = encroached_crease(tc, points, creases, crease_marks) {
                    if split_crease_midpoint(
                        key,
                        points,
                        point_h,
                        params.grading,
                        builder,
                        point_index,
                        on_patch,
                        creases,
                        crease_marks,
                    ) {
                        n += 1;
                        placed.push(tc);
                    }
                    break 'attempt;
                }
                if point_index.contains_key(&tc.map(|x| (x + 0.0).to_bits())) {
                    break 'attempt;
                }
                let g = points.len();
                points.push(tc);
                point_h.push(child_h(
                    tc,
                    &tile_f,
                    points,
                    point_h,
                    params.grading,
                    patch_h[pi],
                ));
                builder.insert(tc);
                point_index.insert(tc.map(|x| (x + 0.0).to_bits()), g);
                let mut marks = DSet::default();
                marks.insert(pi);
                on_patch.push(marks);
                n += 1;
                placed.push(tc);
                break 'attempt;
            }
            // Free interior point. No inside-the-domain test needed: the
            // boundary is tiled by Delaunay faces, so a circumcenter that
            // left its tet's region would encroach the diametral ball of a
            // tile it crossed (Shewchuk's containment argument) and was
            // redirected above.
            if point_index.contains_key(&cc.map(|x| (x + 0.0).to_bits())) {
                break 'attempt;
            }
            if quality_only {
                // Guarded: never remove a constraint (see crease_set above),
                // and never create an edge shorter than the candidate's own
                // shortest edge (an exact circumcenter lands at distance
                // >= 2x that; closer means the float circumcenter of a
                // near-degenerate tet is numerically corrupted).
                let tg = std::time::Instant::now();
                let admitted = builder.insert_guarded(cc, lmin2, |rem| match rem {
                    crate::delaunay::Removal::Face(f) => !tile_set.contains(&f),
                    crate::delaunay::Removal::Edge(a, b) => !crease_set.contains(&(a, b)),
                });
                t_guarded += tg.elapsed();
                if admitted.is_none() {
                    guarded_veto += 1;
                    break 'attempt;
                }
                guarded_ok += 1;
            } else {
                builder.insert(cc);
            }
            let g = points.len();
            points.push(cc);
            point_h.push(child_h(
                cc,
                &tverts,
                points,
                point_h,
                params.grading,
                f64::INFINITY,
            ));
            point_index.insert(cc.map(|x| (x + 0.0).to_bits()), g);
            on_patch.push(DSet::default());
            n += 1;
            placed.push(cc);
        }
        // No progress on an oversized tet via the circumcenter routes:
        // longest-edge midpoint fallback guarantees the size target.
        if n == before {
            if let Some((a, b)) = longest {
                split_longest_edge!(a, b);
            }
        }
    }
    if cand_trace && n_cands > 0 {
        use std::sync::atomic::Ordering;
        eprintln!(
            "  phase3: {n_cands} cands, {guarded_ok} ok, {guarded_veto} veto (nn {}, keep {}), scans {}, insert loop {:.1?} (guarded {:.1?}, encroach {:.1?})",
            crate::delaunay::GUARDED_NN_BAILS.swap(0, Ordering::Relaxed),
            crate::delaunay::GUARDED_KEEP_VETOES.swap(0, Ordering::Relaxed),
            crate::delaunay::LOCATE_SCANS.swap(0, Ordering::Relaxed),
            t_cands.elapsed(),
            t_guarded,
            t_encroach
        );
    }
    n
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
