//! Boundary recovery and region tagging: TaggedPlc to a conforming tet mesh.
//!
//! The PLC's true constraints are planar tagged PATCHES (maximal connected
//! coplanar groups of facets with equal tags) and their boundary CREASE
//! edges — not the individual facet triangles: requiring every assembler
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
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
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

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    groups
        .into_values()
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

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh.
/// Background (region 0) tets are dropped.
pub fn mesh_plc(plc: &TaggedPlc) -> TetMesh {
    let trace = std::env::var_os("RAPIDMESH_TRACE").is_some();
    let mut points: Vec<[f64; 3]> = plc.vertices.clone();
    let patches = build_patches(plc);

    // Combinatorial point-on-patch membership.
    let mut on_patch: Vec<HashSet<usize>> = vec![HashSet::new(); points.len()];
    for (pi, p) in patches.iter().enumerate() {
        for f in &p.members {
            for &v in f {
                on_patch[v].insert(pi);
            }
        }
    }

    // Crease sub-edges: PLC edges that are not interior to a single patch.
    let mut edge_owner_patches: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
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
    let crease_patches: HashMap<(usize, usize), HashSet<usize>> = creases
        .iter()
        .map(|&key| (key, edge_owner_patches[&key].clone()))
        .collect();
    let mut crease_marks: HashMap<(usize, usize), HashSet<usize>> = crease_patches.clone();

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
    let mut point_index: HashMap<[u64; 3], usize> = HashMap::new();
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
    let (tets, patch_faces): (Vec<[usize; 4]>, Vec<Vec<[usize; 3]>>) = loop {
        round += 1;
        assert!(round <= 512, "boundary recovery did not converge");

        let dt_tets = builder.tets();
        let mut dt_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut dt_faces: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
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
                        on_patch.push(HashSet::new());
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
            break (dt_tets, all_tilings);
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
        let mut patch_done: HashSet<usize> = HashSet::new();
        for (x, pi) in new_pts {
            if patch_done.contains(&pi) {
                continue;
            }
            // A piercing point may coincide with (or sit within rounding of)
            // an existing vertex — e.g. a tet edge passing exactly through a
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
                        on_patch.push(HashSet::new());
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
            let mut marks = HashSet::new();
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
    let mut region_bounds: HashMap<u32, Vec<Tri>> = HashMap::new();
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
        // The rounded centroid of a tet can only escape it for
        // pathologically flat tets; fail loudly rather than misclassify.
        assert!(strictly_inside, "tet centroid not strictly inside its tet");
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
