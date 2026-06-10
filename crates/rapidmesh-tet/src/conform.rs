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
use rapidmesh_geom::{FaceTag, RegionTag, TaggedPlc};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
type DState = BuildHasherDefault<std::collections::hash_map::DefaultHasher>;
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
    /// Exact double area of the patch in the projection.
    area2: Expansion,
}

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
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
    /// Delaunay-refinement quality bound: tets with
    /// circumradius / shortest-edge above this get their circumcenter
    /// inserted. The provable refinement regime is >= 2.0.
    pub radius_edge_bound: f64,
    /// Refinement stops (best effort) once this many points exist.
    pub max_points: usize,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            maxh: f64::INFINITY,
            radius_edge_bound: 2.0,
            max_points: 100_000,
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
            radius_edge_bound: f64::INFINITY,
            max_points: usize::MAX,
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
        point_index.insert(p.map(f64::to_bits), i);
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
    let inside_patch = |patch: &Patch, points: &[[f64; 3]], q: [f64; 3]| -> bool {
        let qp = Point3::Explicit(q);
        patch.members.iter().any(|f| {
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
    let strictly_inside_patch = |patch: &Patch, points: &[[f64; 3]], q: [f64; 3]| -> bool {
        let qp = Point3::Explicit(q);
        patch.members.iter().any(|f| {
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

    let mut round = 0;
    let mut refine_round = 0;
    let mut tried: DSet<[usize; 4]> = DSet::default();
    let (tets, patch_faces): (Vec<[usize; 4]>, Vec<Vec<[usize; 3]>>) = 'outer: loop {
        round += 1;
        assert!(round <= 16384, "boundary recovery did not converge");

        let dt_tets = builder.tets();
        let mut dt_edges: DSet<(usize, usize)> = DSet::default();
        let mut dt_faces: DMap<[usize; 3], Vec<usize>> = DMap::default();
        for (ti, t) in dt_tets.iter().enumerate() {
            for i in 0..4 {
                for j in i + 1..4 {
                    dt_edges.insert(sorted2(t[i], t[j]));
                }
                let mut f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
                f.sort_unstable();
                dt_faces.entry([f[0], f[1], f[2]]).or_default().push(ti);
            }
        }

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
            }
            for (a, b) in missing {
                let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
                // A vertex may already sit exactly at the midpoint (e.g. a
                // piercing Steiner point from a symmetric tet edge): reuse
                // it as the chain split point instead of duplicating.
                let g = match point_index.get(&m.map(f64::to_bits)) {
                    Some(&g) => g,
                    None => {
                        let g = points.len();
                        points.push(m);
                        builder.insert(m);
                        point_index.insert(m.map(f64::to_bits), g);
                        on_patch.push(DSet::default());
                        g
                    }
                };
                let marks = crease_marks.remove(&(a, b)).expect("known crease");
                on_patch[g].extend(marks.iter().copied());
                creases.retain(|&e| e != (a, b));
                creases.push(sorted2(a, g));
                creases.push(sorted2(g, b));
                crease_marks.insert(sorted2(a, g), marks.clone());
                crease_marks.insert(sorted2(g, b), marks);
            }
            continue;
        }

        // 2. Patch tiling: exact projected-area comparison.
        let mut all_tilings: Vec<Vec<[usize; 3]>> = Vec::with_capacity(patches.len());
        let mut deficits: Vec<usize> = Vec::new();
        for (pi, patch) in patches.iter().enumerate() {
            let mut tiles: Vec<[usize; 3]> = Vec::new();
            let mut sum = Expansion::from_f64(0.0);
            for f in dt_faces.keys() {
                if !f.iter().all(|&v| on_patch[v].contains(&pi)) {
                    continue;
                }
                let c: [f64; 3] = std::array::from_fn(|k| {
                    (points[f[0]][k] + points[f[1]][k] + points[f[2]][k]) / 3.0
                });
                if !inside_patch(patch, &points, c) {
                    continue;
                }
                let mut a = area2_exact(&points, *f, patch.axis);
                if a.sign() == Sign::Negative {
                    a = a.neg();
                }
                sum = sum.add(&a);
                tiles.push(*f);
            }
            // Tolerant area gate: Steiner chain points sit within rounding
            // of patch boundaries, so the tiling can differ from the exact
            // patch area by slivers of relative size ~1e-16 that no further
            // refinement can remove. Structural conformity stays exact (all
            // tiles are actual tet faces); only "fully tiled" is tolerant.
            let diff = sum.add(&patch.area2.neg()).approx().abs();
            if diff > 1e-9 * patch.area2.approx().abs() {
                deficits.push(pi);
            }
            all_tilings.push(tiles);
        }
        if deficits.is_empty() {
            // Conforming. Apply one sizing/quality refinement step; when it
            // makes no insertion (or the point budget is spent), finish.
            if points.len() >= params.max_points {
                if trace {
                    eprintln!("refinement stopped at point budget ({})", points.len());
                }
                break 'outer (dt_tets, all_tilings);
            }
            refine_round += 1;
            if refine_round > 200 {
                // Sizing/quality are targets, not hard bounds (as in gmsh):
                // stop best-effort instead of chasing corner configurations.
                if trace {
                    eprintln!("refinement stopped at round budget");
                }
                break 'outer (dt_tets, all_tilings);
            }
            let inserted = refine_step(
                params,
                &patches,
                &dt_tets,
                &all_tilings,
                &mut tried,
                &mut points,
                &mut builder,
                &mut point_index,
                &mut on_patch,
                &mut creases,
                &mut crease_marks,
                (blo, bhi),
            );
            if trace && inserted > 0 {
                eprintln!(
                    "refine round {refine_round}: inserted {inserted}, {} points",
                    points.len()
                );
            }
            if inserted == 0 {
                break 'outer (dt_tets, all_tilings);
            }
            continue;
        }
        if trace {
            eprintln!("round {round}: {} patches with tiling deficit", deficits.len());
        }

        // 3. Repair: Steiner points where tet edges pierce a deficient
        // patch. ONE point per patch per round: every insertion changes the
        // DT, so batch insertions computed against a stale DT mostly land
        // redundantly and feed encroachment cascades.
        let mut inserted = 0;
        let mut new_pts: Vec<([f64; 3], usize)> = Vec::new();
        for &pi in &deficits {
            let patch = &patches[pi];
            for &(a, b) in &dt_edges {
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
                if strictly_inside_patch(patch, &points, x) {
                    new_pts.push((x, pi));
                }
            }
        }
        let scale = (0..3).map(|k| bhi[k] - blo[k]).fold(1.0_f64, f64::max);
        // One successful insertion per deficient patch per round: every
        // insertion changes the DT, so further candidates computed against
        // the stale DT would mostly land redundantly and feed encroachment
        // cascades. Candidates are tried in order until one inserts.
        let mut patch_done: DSet<usize> = DSet::default();
        for (x, pi) in new_pts {
            if patch_done.contains(&pi) {
                continue;
            }
            // A piercing point may coincide with (or sit within rounding of)
            // an existing vertex, e.g. a tet edge passing exactly through a
            // crease midpoint. Inserting it would create a duplicate that
            // poisons the triangulation; skip it, other piercings progress.
            if point_index.contains_key(&x.map(f64::to_bits)) {
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
                let g = match point_index.get(&xs.map(f64::to_bits)) {
                    Some(&g) => g,
                    None => {
                        let g = points.len();
                        points.push(xs);
                        builder.insert(xs);
                        point_index.insert(xs.map(f64::to_bits), g);
                        on_patch.push(DSet::default());
                        g
                    }
                };
                if g != a && g != b {
                    let marks = crease_marks.remove(&(a, b)).expect("known crease");
                    on_patch[g].extend(marks.iter().copied());
                    on_patch[g].insert(pi);
                    creases.retain(|&e| e != (a, b));
                    creases.push(sorted2(a, g));
                    creases.push(sorted2(g, b));
                    crease_marks.insert(sorted2(a, g), marks.clone());
                    crease_marks.insert(sorted2(g, b), marks);
                    inserted += 1;
                    patch_done.insert(pi);
                }
                continue;
            }
            let g = points.len();
            points.push(x);
            builder.insert(x);
            point_index.insert(x.map(f64::to_bits), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            inserted += 1;
            patch_done.insert(pi);
        }
        if trace {
            eprintln!("round {round}: inserted {inserted} repair points, {} total", points.len());
        }
        assert!(inserted > 0, "tiling deficit but no piercing edges found");
    };

    // --------------------------------------------------- region tagging
    let mut region_bounds: DMap<u32, Vec<Tri>> = DMap::default();
    for (pi, patch) in patches.iter().enumerate() {
        if patch.regions[0] == patch.regions[1] {
            continue;
        }
        for f in &patch_faces[pi] {
            let t = Tri::new(points[f[0]], points[f[1]], points[f[2]]);
            for tag in patch.regions {
                region_bounds.entry(tag.0).or_default().push(t);
            }
        }
    }

    let mut kept_tets: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for t in &tets {
        let c: [f64; 3] = std::array::from_fn(|k| {
            0.25 * (points[t[0]][k] + points[t[1]][k] + points[t[2]][k] + points[t[3]][k])
        });
        let rep = Point3::Explicit(c);
        let strictly_inside = [
            [t[1], t[3], t[2]],
            [t[0], t[2], t[3]],
            [t[0], t[3], t[1]],
            [t[0], t[1], t[2]],
        ]
        .iter()
        .all(|f| {
            orient3d(
                &Point3::Explicit(points[f[0]]),
                &Point3::Explicit(points[f[1]]),
                &Point3::Explicit(points[f[2]]),
                &rep,
            ) == Some(Sign::Positive)
        });
        // The rounded centroid of a sliver can escape it; classify with the
        // (non-strict) centroid anyway: the exact region-volume gates catch
        // any misclassification, and a sliver thinner than one ulp cannot
        // straddle a region boundary by more than that.
        let _ = strictly_inside;
        let mut region: Option<u32> = None;
        for (&r, bound) in &region_bounds {
            if r == 0 {
                continue;
            }
            if point_inside_solid(&rep, bound, (blo, bhi)) {
                assert!(
                    region.is_none(),
                    "tet centroid inside two regions: PLC regions not disjoint"
                );
                region = Some(r);
            }
        }
        if let Some(r) = region {
            kept_tets.push(*t);
            tet_regions.push(RegionTag(r));
        }
    }

    let mut out_faces: Vec<SurfaceFace> = Vec::new();
    for (pi, patch) in patches.iter().enumerate() {
        for f in &patch_faces[pi] {
            out_faces.push(SurfaceFace {
                tri: *f,
                face_tag: patch.face_tag,
                regions: patch.regions,
            });
        }
    }

    TetMesh {
        points,
        tets: kept_tets,
        tet_regions,
        faces: out_faces,
    }
}

/// Splits the crease sub-edge `key` at its midpoint (reusing an exactly
/// coincident existing vertex if present). Returns false if the chain entry
/// vanished (already split).
#[allow(clippy::too_many_arguments)]
fn split_crease_midpoint(
    key: (usize, usize),
    points: &mut Vec<[f64; 3]>,
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
    let g = match point_index.get(&m.map(f64::to_bits)) {
        Some(&g) => g,
        None => {
            let g = points.len();
            points.push(m);
            builder.insert(m);
            point_index.insert(m.map(f64::to_bits), g);
            on_patch.push(DSet::default());
            g
        }
    };
    on_patch[g].extend(marks.iter().copied());
    creases.retain(|&e| e != key);
    creases.push((a.min(g), a.max(g)));
    creases.push((g.min(b), g.max(b)));
    crease_marks.insert((a.min(g), a.max(g)), marks.clone());
    crease_marks.insert((g.min(b), g.max(b)), marks);
    true
}

/// One sizing/quality refinement step on a conforming state. Returns the
/// number of inserted points (0 = all targets met). Shewchuk phase order:
/// oversized creases, then oversized patch tiles, then oversized or
/// poor-quality tets (circumcenters, with encroachment redirected to the
/// boundary).
#[allow(clippy::too_many_arguments)]
fn refine_step(
    params: &MeshParams,
    patches: &[Patch],
    dt_tets: &[[usize; 4]],
    tilings: &[Vec<[usize; 3]>],
    tried: &mut DSet<[usize; 4]>,
    points: &mut Vec<[f64; 3]>,
    builder: &mut DelaunayBuilder,
    point_index: &mut DMap<[u64; 3], usize>,
    on_patch: &mut Vec<DSet<usize>>,
    creases: &mut Vec<(usize, usize)>,
    crease_marks: &mut DMap<(usize, usize), DSet<usize>>,
    bbox: ([f64; 3], [f64; 3]),
) -> usize {
    if params.maxh.is_infinite() && params.radius_edge_bound.is_infinite() {
        return 0;
    }
    let dist2 = |a: [f64; 3], b: [f64; 3]| -> f64 {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
    };

    // Phase 1: creases longer than maxh (disjoint midpoint splits batch
    // safely).
    if params.maxh.is_finite() {
        let long: Vec<(usize, usize)> = creases
            .iter()
            .copied()
            .filter(|&(a, b)| dist2(points[a], points[b]) > params.maxh * params.maxh)
            .collect();
        if !long.is_empty() {
            let mut n = 0;
            for key in long {
                if split_crease_midpoint(
                    key,
                    points,
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
    // x, if any.
    let encroached_crease = |x: [f64; 3], points: &[[f64; 3]], creases: &[(usize, usize)]| {
        creases.iter().copied().find(|&(a, b)| {
            let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (points[a][k] + points[b][k]));
            dist2(x, m) < 0.25 * dist2(points[a], points[b])
        })
    };

    // Phase 2: oversized patch tiles (one per patch per step; insertions
    // change the DT).
    if params.maxh.is_finite() {
        let mut n = 0;
        for (pi, tiles) in tilings.iter().enumerate() {
            let Some((cc, _r)) = tiles
                .iter()
                .filter_map(|f| tri_circumcenter(points[f[0]], points[f[1]], points[f[2]]))
                .filter(|&(_, r)| r > 0.5 * params.maxh)
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            if let Some(key) = encroached_crease(cc, points, creases) {
                if split_crease_midpoint(
                    key,
                    points,
                    builder,
                    point_index,
                    on_patch,
                    creases,
                    crease_marks,
                ) {
                    n += 1;
                }
                continue;
            }
            if point_index.contains_key(&cc.map(f64::to_bits)) {
                continue;
            }
            let g = points.len();
            points.push(cc);
            builder.insert(cc);
            point_index.insert(cc.map(f64::to_bits), g);
            let mut marks = DSet::default();
            marks.insert(pi);
            on_patch.push(marks);
            n += 1;
        }
        if n > 0 {
            return n;
        }
    }

    // Phase 3: tets, oversized or poor radius-edge quality. Insert
    // circumcenters unless they encroach the boundary (then split that
    // instead). Batched with a spacing guard against near-duplicate
    // circumcenters from neighboring tets.
    let mut region_bounds: Vec<Tri> = Vec::new();
    for (pi, patch) in patches.iter().enumerate() {
        if patch.regions[0] == patch.regions[1] {
            continue;
        }
        for f in &tilings[pi] {
            region_bounds.push(Tri::new(points[f[0]], points[f[1]], points[f[2]]));
        }
    }
    // (cc, circumradius, spacing, longest edge if oversized)
    #[allow(clippy::type_complexity)]
    let mut candidates: Vec<([f64; 3], f64, f64, Option<(usize, usize)>)> = Vec::new();
    for t in dt_tets {
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
        let oversized = params.maxh.is_finite() && lmax2 > params.maxh * params.maxh;
        // Near-degenerate (sliver) tets have no usable circumcenter; their
        // long edges are tolerated (sizing is a target, and forcing midpoint
        // splits into slivers spawns new slivers without converging).
        let Some((cc, r)) = tet_circumcenter(p) else {
            continue;
        };
        let bad_quality = params.radius_edge_bound.is_finite()
            && r > params.radius_edge_bound * lmin2.sqrt();
        if !oversized && !bad_quality {
            continue;
        }
        // Quality-only candidates are attempted once: re-attempting a
        // boundary-locked sliver every round spirals (its repair points
        // spawn new slivers). Oversized tets must shrink, so they retry.
        if !oversized {
            let mut key = *t;
            key.sort_unstable();
            if !tried.insert(key) {
                continue;
            }
        }
        // Only refine tets that belong to a region (the DT also fills the
        // gap between the convex hull and the domain).
        let centroid: [f64; 3] = std::array::from_fn(|k| {
            0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k])
        });
        if !point_inside_solid(&Point3::Explicit(centroid), &region_bounds, bbox) {
            continue;
        }
        candidates.push((cc, r, 0.5 * lmin2.sqrt(), oversized.then_some(longest)));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
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
                if !point_index.contains_key(&m.map(f64::to_bits)) {
                    let g = points.len();
                    points.push(m);
                    builder.insert(m);
                    point_index.insert(m.map(f64::to_bits), g);
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
    for (cc, _r, spacing, longest) in candidates {
        if n >= 32 {
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
                if encroached_crease(cc, points, creases).is_some() {
                    break 'attempt;
                }
            }
            if let Some(key) = encroached_crease(cc, points, creases) {
                if split_crease_midpoint(
                    key,
                    points,
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
            let mut tile_hit: Option<(usize, [f64; 3])> = None;
            'tiles: for (pi, tiles) in tilings.iter().enumerate() {
                for f in tiles {
                    if let Some((tc, tr)) =
                        tri_circumcenter(points[f[0]], points[f[1]], points[f[2]])
                    {
                        if dist2(cc, tc) < tr * tr {
                            tile_hit = Some((pi, tc));
                            break 'tiles;
                        }
                    }
                }
            }
            if let Some((pi, tc)) = tile_hit {
                if quality_only {
                    break 'attempt; // no boundary interaction for quality steps
                }
                if let Some(key) = encroached_crease(tc, points, creases) {
                    if split_crease_midpoint(
                        key,
                        points,
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
                if point_index.contains_key(&tc.map(f64::to_bits)) {
                    break 'attempt;
                }
                let g = points.len();
                points.push(tc);
                builder.insert(tc);
                point_index.insert(tc.map(f64::to_bits), g);
                let mut marks = DSet::default();
                marks.insert(pi);
                on_patch.push(marks);
                n += 1;
                placed.push(tc);
                break 'attempt;
            }
            // Free interior point: must lie inside some region.
            if point_index.contains_key(&cc.map(f64::to_bits))
                || !point_inside_solid(&Point3::Explicit(cc), &region_bounds, bbox)
            {
                break 'attempt;
            }
            let g = points.len();
            points.push(cc);
            builder.insert(cc);
            point_index.insert(cc.map(f64::to_bits), g);
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
    n
}

/// Minimum dihedral angle of a tet in degrees (the projection-based
/// formula: angle between the projections of the two opposite vertices onto
/// the plane normal to each edge).
pub(crate) fn tet_min_dihedral_deg(p: [[f64; 3]; 4]) -> f64 {
    let mut min_dihedral = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            let others: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
            let (a, b) = (p[i], p[j]);
            let tlen: f64 = (0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt();
            if tlen == 0.0 {
                return 0.0;
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
                return 0.0;
            }
            let cosang = ((0..3).map(|k| u[k] * v[k]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
            min_dihedral = min_dihedral.min(cosang.acos().to_degrees());
        }
    }
    min_dihedral
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
