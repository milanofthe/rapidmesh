//! CVT (centroidal Voronoi) tetrahedral meshing of a tagged PLC.
//!
//! Replaces the constrained-Delaunay + Steiner boundary recovery of the old
//! pipeline. The exact CSG arrangement (the `TaggedPlc`) is untouched; this
//! stage fills it with a variational (Lloyd-relaxed) tetrahedralization.
//!
//! Staged build (see the CVT-rewrite plan):
//!   WP3 (this commit): single-region solids whose boundary the convex hull of
//!     the seeded sites recovers. Boundary sites are the PLC vertices (pinned);
//!     interior sites are a BCC lattice filtered to strictly inside the domain;
//!     a Lloyd pass relaxes the interior toward a centroidal layout.
//!   WP4: restricted-Voronoi boundary conformity for non-convex / planar
//!     interfaces. WP5: multi-region. WP6: features + curved projection.
//!   WP7: grading. WP8: quality post-pass. WP9: octree/parallel perf.

use crate::conform::{build_patches, quality_stats, MeshParams, SurfaceFace, TetMesh};
use crate::delaunay::DelaunayBuilder;
use crate::seed::{bcc_lattice, SizingField};
use crate::spatial::Octree;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{RegionTag, TaggedPlc};

type V3 = [f64; 3];

/// Lloyd relaxation passes over the interior sites.
const LLOYD_ITERS: usize = 10;
/// Bounding-box subdivisions for the default spacing when no `maxh` is given
/// (an unsized `mesh_plc` still gets a coarse but valid interior fill).
const DEFAULT_SUBDIV: f64 = 8.0;
/// Per-triangle bounding-box pad for the inside test, as a fraction of the
/// scene diagonal (absorbs f64 round-off of query points).
const BOX_PAD_FRAC: f64 = 1e-6;
/// Minimum separation between a moved/seeded site and any other, as a fraction
/// of the local spacing (guards the Delaunay against near-duplicate inserts).
const SEPARATION_FRAC: f64 = 0.35;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

/// Signed 6x-volume determinant of a tet.
fn tet_det(p: [V3; 4]) -> f64 {
    dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])))
}
fn centroid4(p: [V3; 4]) -> V3 {
    std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]))
}

/// Builds the closed Tri soup of the PLC boundary (every PLC triangle), for the
/// inside test. Valid as a single closed surface only when the PLC bounds one
/// region (the WP3 scope); multi-region uses flood-fill classification (WP5).
fn boundary_soup(plc: &TaggedPlc) -> Vec<Tri> {
    plc.triangles
        .iter()
        .map(|t| {
            Tri::new(
                plc.vertices[t[0] as usize],
                plc.vertices[t[1] as usize],
                plc.vertices[t[2] as usize],
            )
        })
        .collect()
}

/// The non-background regions present in the PLC.
fn regions_of(plc: &TaggedPlc) -> Vec<u32> {
    let mut rs: Vec<u32> = Vec::new();
    for pair in &plc.region_tags {
        for r in pair {
            if r.0 != 0 && !rs.contains(&r.0) {
                rs.push(r.0);
            }
        }
    }
    rs.sort_unstable();
    rs
}

/// Meshes a tagged PLC into a region-tagged tet mesh by CVT. WP3 scope: single
/// non-background region with a hull-recoverable (convex) boundary.
pub fn mesh(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    let t_start = std::time::Instant::now();

    // Bounding box + diagonal.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max);

    let regions = regions_of(plc);
    let primary = regions.first().copied().unwrap_or(0);
    let field = SizingField::new(params);
    let cap = field.finest_cap(&regions);
    let spacing = if cap.is_finite() { cap } else { diag / DEFAULT_SUBDIV };

    // Inside-test acceleration over the boundary soup.
    let soup = boundary_soup(plc);
    let boxes = TriBoxes::build(&soup, BOX_PAD_FRAC * diag.max(1.0));
    let inside = |p: V3| point_inside_solid(&Point3::Explicit(p), p, &soup, &boxes, (lo, hi));

    // --- seeding ----------------------------------------------------------
    let t_seed = std::time::Instant::now();
    // Boundary sites: the PLC vertices, in order (pinned). Their indices match
    // PLC vertex indices, which the boundary-face tagging below relies on.
    let nb = plc.vertices.len();
    let mut sites: Vec<V3> = plc.vertices.clone();
    // Interior sites: BCC lattice strictly inside the domain, kept away from the
    // boundary sites so the Delaunay never sees a near-duplicate.
    if spacing.is_finite() && spacing > 0.0 {
        let btree = Octree::build(&plc.vertices);
        for p in bcc_lattice(lo, hi, spacing) {
            if !inside(p) {
                continue;
            }
            // Keep clear of the pinned boundary sites (near-duplicate guard).
            if let Some(j) = btree.nearest(p) {
                if dist(p, plc.vertices[j as usize]) < SEPARATION_FRAC * spacing {
                    continue;
                }
            }
            sites.push(p);
        }
    }
    let n_interior_seed = sites.len() - nb;
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // --- Lloyd relaxation of the interior sites ---------------------------
    let t_lloyd = std::time::Instant::now();
    for _ in 0..LLOYD_ITERS {
        if sites.len() == nb {
            break; // no interior sites to move
        }
        let tets = delaunay_of(&sites, lo, hi);
        // Volume-weighted incident-tet centroid per site (an ODT-flavored Lloyd
        // move that does not shrink the boundary).
        let mut num = vec![[0.0f64; 3]; sites.len()];
        let mut den = vec![0.0f64; sites.len()];
        for t in &tets {
            let p = [sites[t[0]], sites[t[1]], sites[t[2]], sites[t[3]]];
            let w = tet_det(p).abs();
            let c = centroid4(p);
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * c[k];
                }
                den[i] += w;
            }
        }
        let tree = Octree::build(&sites);
        for i in nb..sites.len() {
            if den[i] == 0.0 {
                continue;
            }
            let tgt: V3 = std::array::from_fn(|k| num[i][k] / den[i]);
            // Reject a move that leaves the domain or crowds a neighbor.
            if !inside(tgt) {
                continue;
            }
            let crowded = tree
                .within_radius(tgt, SEPARATION_FRAC * spacing)
                .into_iter()
                .any(|j| j as usize != i);
            if crowded {
                continue;
            }
            sites[i] = tgt;
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());

    // --- final triangulation + region classification ----------------------
    let t_classify = std::time::Instant::now();
    let all_tets = delaunay_of(&sites, lo, hi);
    // Single-region: a tet belongs to `primary` iff its centroid is inside the
    // domain; convex-hull overshoot tets (centroid outside) are dropped.
    let mut kept: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for t in &all_tets {
        let c = centroid4([sites[t[0]], sites[t[1]], sites[t[2]], sites[t[3]]]);
        if inside(c) {
            kept.push(*t);
            tet_regions.push(RegionTag(primary));
        }
    }
    rmlog::stage("mesh.classify", t_classify.elapsed().as_secs_f64());

    // --- boundary faces tagged to their PLC patch -------------------------
    let patches = build_patches(plc);
    let faces = tag_boundary_faces(plc, &patches, &sites, &kept, nb);

    let mesh = TetMesh {
        points: sites,
        tets: kept,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points: nb,
    };

    // Stats (Python contract).
    let q = quality_stats(&mesh);
    rmlog::stat("mesh.points", mesh.points.len() as f64);
    rmlog::stat("mesh.tets", mesh.tets.len() as f64);
    rmlog::stat("mesh.interior_seeds", n_interior_seed as f64);
    rmlog::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    rmlog::stage("mesh.total", t_start.elapsed().as_secs_f64());
    mesh
}

/// A fresh Delaunay over the current sites; returns real tets in site indices.
fn delaunay_of(sites: &[V3], lo: V3, hi: V3) -> Vec<[usize; 4]> {
    let mut db = DelaunayBuilder::enclosing(lo, hi);
    for &p in sites {
        db.insert(p);
    }
    db.tets()
}

/// Boundary faces (shared by exactly one kept tet) tagged to the PLC patch
/// whose plane contains them. Face vertices are boundary sites (indices < nb),
/// i.e. PLC vertices, so coplanarity is tested against the patch facets exactly.
fn tag_boundary_faces(
    plc: &TaggedPlc,
    patches: &[crate::conform::Patch],
    sites: &[V3],
    kept: &[[usize; 4]],
    nb: usize,
) -> Vec<SurfaceFace> {
    use std::collections::HashMap;
    const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    // Count incident kept tets per face.
    let mut owners: HashMap<[usize; 3], usize> = HashMap::new();
    for t in kept {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            *owners.entry(f).or_default() += 1;
        }
    }
    // Patch plane representative (first member facet's three PLC vertices).
    let patch_tri = |p: &crate::conform::Patch| -> [Point3; 3] {
        let t = plc.triangles[p.member_indices[0]];
        [
            Point3::Explicit(plc.vertices[t[0] as usize]),
            Point3::Explicit(plc.vertices[t[1] as usize]),
            Point3::Explicit(plc.vertices[t[2] as usize]),
        ]
    };
    let mut out = Vec::new();
    for (f, &c) in &owners {
        if c != 1 {
            continue; // interior face (2 owners) or non-manifold
        }
        // All three vertices must be boundary sites (PLC vertices).
        if f.iter().any(|&v| v >= nb) {
            continue;
        }
        // Find the patch whose plane contains the face.
        let mut chosen: Option<usize> = None;
        for (pi, p) in patches.iter().enumerate() {
            let [a, b, cc] = patch_tri(p);
            let coplanar = f.iter().all(|&v| {
                orient3d(&a, &b, &cc, &Point3::Explicit(sites[v])) == Some(Sign::Zero)
            });
            if coplanar {
                chosen = Some(pi);
                break;
            }
        }
        if let Some(pi) = chosen {
            let p = &patches[pi];
            out.push(SurfaceFace {
                tri: *f,
                face_tag: p.face_tag,
                regions: p.regions,
                patch: pi as u32,
                surface: p.surface,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::Zero;
    use rapidmesh_geom::{solid_box, Scene};
    use rapidmesh_testutil::rat;

    fn region_volume6(m: &TetMesh, r: u32) -> BigRational {
        let mut acc = BigRational::zero();
        for (t, tr) in m.tets.iter().zip(&m.tet_regions) {
            if tr.0 != r {
                continue;
            }
            let p: Vec<[BigRational; 3]> = t.iter().map(|&i| m.points[i].map(rat)).collect();
            let row: Vec<[BigRational; 3]> =
                (0..3).map(|k| std::array::from_fn(|j| &p[k][j] - &p[3][j])).collect();
            let det = &row[0][0] * (&row[1][1] * &row[2][2] - &row[1][2] * &row[2][1])
                - &row[0][1] * (&row[1][0] * &row[2][2] - &row[1][2] * &row[2][0])
                + &row[0][2] * (&row[1][0] * &row[2][1] - &row[1][1] * &row[2][0]);
            acc += det;
        }
        acc
    }

    fn watertight(m: &TetMesh) -> bool {
        use std::collections::HashMap;
        const TF: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
        let mut faces: HashMap<[usize; 3], usize> = HashMap::new();
        for t in &m.tets {
            for fv in &TF {
                let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
                f.sort_unstable();
                *faces.entry(f).or_default() += 1;
            }
        }
        faces.values().all(|&c| c <= 2)
    }

    #[test]
    fn box_meshes_exactly_and_with_quality() {
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let params = MeshParams { maxh: 0.8, ..Default::default() };
        let mesh = mesh(&plc, &params);

        // Exact volume: 2*3*4 = 24, times 6 = 144.
        assert_eq!(region_volume6(&mesh, r.0), rat(144.0), "exact box volume");
        assert!(watertight(&mesh), "watertight boundary");
        assert!(!mesh.tets.is_empty());
        let q = quality_stats(&mesh);
        assert!(q.min_dihedral_deg > 1.0, "non-degenerate tets, got {}", q.min_dihedral_deg);
        // Real interior refinement happened (more than just the 8 corners).
        assert!(mesh.points.len() > 8, "interior seeded: {} points", mesh.points.len());
    }

    #[test]
    fn unsized_box_is_valid() {
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams::default());
        assert_eq!(region_volume6(&mesh, r.0), rat(6.0), "exact unit cube");
        assert!(watertight(&mesh));
    }
}
