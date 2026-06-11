//! End-to-end: scene -> TaggedPlc -> conforming region-tagged tet mesh.
//! Gates: every constraint face is a tet face, per-region tet volumes match
//! the PLC's polyhedral region volumes exactly, orientation and conformity.

use num_rational::BigRational;
use num_traits::Zero;
use rapidmesh_geom::{cylinder, sheet_rect, solid_box, FaceTag, RegionTag, Scene, TaggedPlc};
use rapidmesh_tet::{
    mesh_plc, mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams, TetMesh,
};
use rapidmesh_testutil::rat;

/// Exact 6x volume of a PLC region from its interface facets.
fn plc_region_volume6(plc: &TaggedPlc, r: RegionTag) -> BigRational {
    let mut acc = BigRational::zero();
    for (t, tags) in plc.triangles.iter().zip(&plc.region_tags) {
        let sign = if tags[0] == tags[1] {
            continue;
        } else if tags[1] == r {
            1
        } else if tags[0] == r {
            -1
        } else {
            continue;
        };
        let (a, b, c) = (
            plc.vertices[t[0] as usize].map(rat),
            plc.vertices[t[1] as usize].map(rat),
            plc.vertices[t[2] as usize].map(rat),
        );
        let det = &a[0] * (&b[1] * &c[2] - &b[2] * &c[1])
            - &a[1] * (&b[0] * &c[2] - &b[2] * &c[0])
            + &a[2] * (&b[0] * &c[1] - &b[1] * &c[0]);
        acc += if sign > 0 { det } else { -det };
    }
    acc
}

/// Exact 6x total volume of all tets in one region.
fn mesh_region_volume6(m: &TetMesh, r: RegionTag) -> BigRational {
    let mut acc = BigRational::zero();
    for (t, &tr) in m.tets.iter().zip(&m.tet_regions) {
        if tr != r {
            continue;
        }
        let p: Vec<[BigRational; 3]> = t.iter().map(|&i| m.points[i].map(rat)).collect();
        let rrow: Vec<[BigRational; 3]> = (0..3)
            .map(|k| std::array::from_fn(|j| &p[k][j] - &p[3][j]))
            .collect();
        let det = &rrow[0][0] * (&rrow[1][1] * &rrow[2][2] - &rrow[1][2] * &rrow[2][1])
            - &rrow[0][1] * (&rrow[1][0] * &rrow[2][2] - &rrow[1][2] * &rrow[2][0])
            + &rrow[0][2] * (&rrow[1][0] * &rrow[2][1] - &rrow[1][1] * &rrow[2][0]);
        acc += det;
    }
    acc
}

/// Structural gates: conformity of constraint faces, tet-face matching of
/// region interfaces, manifold tet connectivity.
fn check_structure(m: &TetMesh) {
    // Tet face multiset.
    let mut tet_faces: std::collections::HashMap<[usize; 3], Vec<usize>> =
        std::collections::HashMap::new();
    for (ti, t) in m.tets.iter().enumerate() {
        for i in 0..4 {
            let mut f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
            f.sort_unstable();
            tet_faces.entry([f[0], f[1], f[2]]).or_default().push(ti);
        }
    }
    // Conformity: every constraint face is a tet face; region interfaces
    // separate tets of exactly the right regions.
    for sf in &m.faces {
        let mut k = sf.tri;
        k.sort_unstable();
        let owners = tet_faces
            .get(&k)
            .unwrap_or_else(|| panic!("constraint face {k:?} is not a tet face"));
        if sf.regions[0] != sf.regions[1] {
            let mut have: Vec<u32> = owners.iter().map(|&ti| m.tet_regions[ti].0).collect();
            have.sort_unstable();
            let mut want: Vec<u32> = sf
                .regions
                .iter()
                .map(|r| r.0)
                .filter(|&r| r != 0)
                .collect();
            want.sort_unstable();
            assert_eq!(have, want, "interface face has wrong adjacent regions");
        } else {
            // Embedded sheet: both sides exist and share the region.
            assert_eq!(owners.len(), 2, "embedded sheet face must have 2 tets");
            assert_eq!(m.tet_regions[owners[0]], m.tet_regions[owners[1]]);
        }
    }
    // Every face is shared by at most 2 tets.
    assert!(tet_faces.values().all(|v| v.len() <= 2), "non-manifold face");
}

#[test]
fn air_dielectric_pec_scene_meshes_exactly() {
    let mut scene = Scene::new();
    let air = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    scene.add_sheet(
        sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(7),
    );
    scene.add_sheet(
        sheet_rect([0.5, 0.5, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(7),
    );
    let plc = scene.assemble();
    let mesh = mesh_plc(&plc);

    assert_eq!(mesh_region_volume6(&mesh, air), rat(360.0));
    assert_eq!(mesh_region_volume6(&mesh, diel), rat(24.0));
    check_structure(&mesh);

    // PEC faces made it into the mesh as tet faces (checked in
    // check_structure) and exist in nonzero number.
    let pec = mesh.faces.iter().filter(|f| f.face_tag == FaceTag(7)).count();
    assert!(pec >= 4, "expected PEC faces in the mesh, got {pec}");
}

#[test]
fn cylinder_via_in_box_meshes_exactly() {
    // A polyhedral via through a dielectric block: curved-ish geometry,
    // non-grid points, region priority.
    let mut scene = Scene::new();
    let block = scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    let via = scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 12));
    let plc = scene.assemble();
    let mesh = mesh_plc(&plc);

    // Region volumes agree with the PLC's polyhedral volumes up to the
    // epsilon slivers of rounded Steiner boundary points (relative ~1e-16;
    // gate at 1e-9). Grid-aligned scenes (other tests) stay bit-exact.
    let close = |have: BigRational, want: BigRational| {
        let tol = want.clone() * rat(1e-9);
        let diff = if have > want {
            have - want.clone()
        } else {
            want.clone() - have
        };
        assert!(diff <= tol, "volume off by more than 1e-9 relative");
    };
    close(
        mesh_region_volume6(&mesh, block),
        plc_region_volume6(&plc, block),
    );
    close(mesh_region_volume6(&mesh, via), plc_region_volume6(&plc, via));
    check_structure(&mesh);
}

#[test]
fn sized_box_respects_maxh_and_quality() {
    let mut scene = Scene::new();
    let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]));
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: 0.6,
        region_maxh: Vec::new(),
        radius_edge_bound: 2.0,
        max_points: 20_000,
        grading: 0.5,
        face_maxh: Vec::new(),
        size_points: Vec::new(),
    };
    let mut mesh = mesh_plc_with(&plc, &params);
    let before = quality_stats(&mesh);
    let ops = optimize(
        &mut mesh,
        &OptimizeParams {
            maxh: params.maxh,
            region_maxh: params.region_maxh.clone(),
            ..OptimizeParams::default()
        },
    );
    let q = quality_stats(&mesh);
    eprintln!("sized box quality: {before:?} -> {q:?} ({ops} ops)");
    // Volume stays exact under interior refinement and optimization;
    // structure intact.
    assert_eq!(mesh_region_volume6(&mesh, r), rat(36.0));
    check_structure(&mesh);
    assert!(
        q.max_edge <= 1.5 * params.maxh,
        "edge {} too long",
        q.max_edge
    );
    assert!(
        q.min_dihedral_deg >= before.min_dihedral_deg,
        "optimization must not worsen the worst dihedral"
    );
    assert!(
        q.min_dihedral_deg >= 5.0,
        "min dihedral {} too small after optimization",
        q.min_dihedral_deg
    );
    assert!(q.n_tets > 100, "expected real refinement, got {} tets", q.n_tets);
}

#[test]
fn sized_em_scene_stays_exact_and_conforming() {
    let mut scene = Scene::new();
    let air = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    scene.add_sheet(
        sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(7),
    );
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: 1.1,
        region_maxh: Vec::new(),
        radius_edge_bound: 2.0,
        max_points: 20_000,
        grading: 0.5,
        face_maxh: Vec::new(),
        size_points: Vec::new(),
    };
    let mut mesh = mesh_plc_with(&plc, &params);
    let before = quality_stats(&mesh);
    let ops = optimize(
        &mut mesh,
        &OptimizeParams {
            maxh: params.maxh,
            region_maxh: params.region_maxh.clone(),
            ..OptimizeParams::default()
        },
    );
    let q = quality_stats(&mesh);
    eprintln!("sized EM scene quality: {before:?} -> {q:?} ({ops} ops)");
    assert_eq!(mesh_region_volume6(&mesh, air), rat(360.0));
    assert_eq!(mesh_region_volume6(&mesh, diel), rat(24.0));
    check_structure(&mesh);
    // Sizing is a target (like gmsh's mesh size), not a hard bound; allow
    // slack for corner configurations.
    assert!(q.max_edge <= 1.5 * params.maxh, "edge {} too long", q.max_edge);
    assert!(
        q.min_dihedral_deg >= 5.0,
        "min dihedral {} too small after optimization",
        q.min_dihedral_deg
    );
    assert!(
        mesh.faces.iter().filter(|f| f.face_tag == FaceTag(7)).count() >= 2,
        "PEC faces must survive refinement"
    );
}

#[test]
fn per_region_sizing_creates_density_transition() {
    let mut scene = Scene::new();
    let air = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: 1.4,
        region_maxh: vec![(diel.0, 0.5)],
        radius_edge_bound: 2.0,
        max_points: 40_000,
        grading: 0.5,
        face_maxh: Vec::new(),
        size_points: Vec::new(),
    };
    let mut mesh = mesh_plc_with(&plc, &params);
    optimize(
        &mut mesh,
        &OptimizeParams {
            maxh: params.maxh,
            region_maxh: params.region_maxh.clone(),
            ..OptimizeParams::default()
        },
    );
    assert_eq!(mesh_region_volume6(&mesh, air), rat(360.0));
    assert_eq!(mesh_region_volume6(&mesh, diel), rat(24.0));
    check_structure(&mesh);
    // Per-region max edge respects each region's target (with slack).
    let max_edge_in = |r: RegionTag| -> f64 {
        let mut m: f64 = 0.0;
        for (t, &tr) in mesh.tets.iter().zip(&mesh.tet_regions) {
            if tr != r {
                continue;
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    let d2: f64 = (0..3)
                        .map(|k| (mesh.points[t[i]][k] - mesh.points[t[j]][k]).powi(2))
                        .sum();
                    m = m.max(d2.sqrt());
                }
            }
        }
        m
    };
    let (e_diel, e_air) = (max_edge_in(diel), max_edge_in(air));
    eprintln!("density transition: diel max edge {e_diel:.3}, air max edge {e_air:.3}");
    assert!(e_diel <= 1.5 * 0.5, "dielectric too coarse: {e_diel}");
    assert!(e_air <= 1.5 * 1.4, "air too coarse: {e_air}");
    // The transition exists: air really is coarser than the dielectric.
    assert!(e_air > 1.5 * e_diel, "expected a density transition");
}

#[test]
fn single_box_meshes_exactly() {
    let mut scene = Scene::new();
    let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]));
    let plc = scene.assemble();
    let mesh = mesh_plc(&plc);
    assert_eq!(mesh_region_volume6(&mesh, r), rat(36.0));
    check_structure(&mesh);
    assert!(mesh.tets.iter().all(|t| t.iter().all(|&v| v < mesh.points.len())));
}

/// Regression for the cylinder-tessellation recovery lottery: stacked
/// cylinders in a box (the dielectric-resonator example) with per-region
/// sizing used to break for specific segment counts. Refinement points
/// hovering an ulp off non-axis-aligned facet planes (or an ulp outside the
/// hull) double-covered or holed the patch tilings, abandoning patches and
/// leaking regions. The side-of-plane tile rule, guarded interior inserts,
/// and uncovered-first repair keep every tessellation conforming; region
/// volumes match the PLC polyhedra.
#[test]
fn resonator_cylinder_tessellations_stay_conforming() {
    let close = |have: BigRational, want: BigRational| {
        let tol = want.clone() * rat(1e-9);
        let diff = if have > want {
            have - want.clone()
        } else {
            want.clone() - have
        };
        assert!(diff <= tol, "volume off by more than 1e-9 relative");
    };
    let mm = 1e-3;
    let inch = 25.4 * mm;
    let w = 2.0 * inch;
    let s = 2.03 * inch;
    let (d_sup, l_sup) = (0.56 * inch, 0.80 * inch);
    let (d_res, l_res) = (1.176 * inch, 0.481 * inch);
    // 20 fanned hull faces below the box bottom, 24 holed a lateral facet
    // through an unmarked chord midpoint, 28 through an unguarded insert.
    for segments in [20, 24, 28] {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-w / 2.0, -w / 2.0, 0.0], [w / 2.0, w / 2.0, s]));
        let sup = scene.add_solid(cylinder(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, l_sup],
            d_sup / 2.0,
            segments,
        ));
        let res = scene.add_solid(cylinder(
            [0.0, 0.0, l_sup],
            [0.0, 0.0, l_res],
            d_res / 2.0,
            segments,
        ));
        let plc = scene.assemble();
        let params = MeshParams {
            maxh: 9.993e-3,
            region_maxh: vec![(sup.0, 3.160e-3), (res.0, 1.713e-3)],
            ..MeshParams::default()
        };
        let mesh = mesh_plc_with(&plc, &params);
        close(
            mesh_region_volume6(&mesh, sup),
            plc_region_volume6(&plc, sup),
        );
        close(
            mesh_region_volume6(&mesh, res),
            plc_region_volume6(&plc, res),
        );
        check_structure(&mesh);
    }
}

/// The cut boolean: a void carves its volume out of the mesh; the remaining
/// region volume is exact and the void walls survive as boundary patches.
#[test]
fn void_carves_exact_volume() {
    let mut scene = Scene::new();
    let block = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
    scene.add_void(solid_box([1.0, 1.0, 0.5], [3.0, 3.0, 1.5]));
    let plc = scene.assemble();
    let mesh = mesh_plc(&plc);
    // 4*4*2 - 2*2*1 = 28; times 6 = 168.
    assert_eq!(mesh_region_volume6(&mesh, block), rat(168.0));
    check_structure(&mesh);
    // The void walls are boundary faces (block on one side, background on
    // the other) that boundary conditions can target.
    let wall_faces = mesh
        .faces
        .iter()
        .filter(|f| {
            let r = [f.regions[0].0, f.regions[1].0];
            r.contains(&0) && r.contains(&block.0) && {
                let c: [f64; 3] = std::array::from_fn(|k| {
                    (mesh.points[f.tri[0]][k]
                        + mesh.points[f.tri[1]][k]
                        + mesh.points[f.tri[2]][k])
                        / 3.0
                });
                c[0] > 0.9 && c[0] < 3.1 && c[1] > 0.9 && c[1] < 3.1 && c[2] > 0.4 && c[2] < 1.6
            }
        })
        .count();
    assert!(wall_faces >= 12, "expected void wall faces, got {wall_faces}");
}

/// Per-face-tag sizing: a tagged sheet refines to its own target inside a
/// coarse region, and the optimizer's face budget keeps it there.
#[test]
fn face_maxh_refines_tagged_sheet() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_sheet(
        sheet_rect([1.0, 1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
        FaceTag(7),
    );
    let params = MeshParams {
        maxh: 1.5,
        face_maxh: vec![(7, 0.4)],
        ..MeshParams::default()
    };
    let mut mesh = mesh_plc_with(&plc_of(&scene), &params);
    optimize(
        &mut mesh,
        &OptimizeParams {
            maxh: params.maxh,
            face_maxh: params.face_maxh.clone(),
            ..OptimizeParams::default()
        },
    );
    let mut sheet_lmax2 = 0.0f64;
    let mut n_sheet = 0;
    for f in &mesh.faces {
        if f.face_tag != FaceTag(7) {
            continue;
        }
        n_sheet += 1;
        for e in 0..3 {
            let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
            let d2: f64 = (0..3)
                .map(|k| (mesh.points[a][k] - mesh.points[b][k]).powi(2))
                .sum();
            sheet_lmax2 = sheet_lmax2.max(d2);
        }
    }
    assert!(n_sheet > 20, "sheet should refine, got {n_sheet} faces");
    assert!(
        sheet_lmax2.sqrt() <= 1.5 * 0.4 + 1e-9,
        "sheet edge {} exceeds the face budget",
        sheet_lmax2.sqrt()
    );
    check_structure(&mesh);
}

/// Point size sources: the target shrinks near the source and recovers
/// along the grading away from it.
#[test]
fn size_points_refine_locally() {
    let mut scene = Scene::new();
    let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    let params = MeshParams {
        maxh: 1.5,
        size_points: vec![([2.0, 2.0, 2.0], 0.2)],
        grading: 0.5,
        ..MeshParams::default()
    };
    let mesh = mesh_plc_with(&plc_of(&scene), &params);
    assert_eq!(mesh_region_volume6(&mesh, r), rat(384.0));
    // Edges fully inside a ball around the source obey the graded target;
    // edges far away stay coarse.
    let mut near_lmax = 0.0f64;
    let mut far_lmax = 0.0f64;
    for t in &mesh.tets {
        for i in 0..4 {
            for j in i + 1..4 {
                let (a, b) = (mesh.points[t[i]], mesh.points[t[j]]);
                let d: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
                let mid_r: f64 = (0..3)
                    .map(|k| (0.5 * (a[k] + b[k]) - 2.0).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if mid_r < 0.4 {
                    near_lmax = near_lmax.max(d);
                } else if mid_r > 1.6 {
                    far_lmax = far_lmax.max(d);
                }
            }
        }
    }
    // Near the source the graded target is ~0.2 + 0.5 * 0.4 = 0.4; allow the
    // 1.5x contract plus the oversize trigger slack.
    assert!(
        near_lmax <= 1.5 * 0.45,
        "near-source edge {near_lmax} not refined"
    );
    assert!(far_lmax > 2.0 * near_lmax, "expected a graded transition");
}

/// Scene helper for the new tests.
fn plc_of(scene: &Scene) -> TaggedPlc {
    scene.assemble()
}

/// A CURVED void (cylindrical bore as a void, not a region) stays exact
/// through meshing and within fidelity tolerance through optimization.
#[test]
fn cylinder_void_volume_through_optimize() {
    let close = |have: BigRational, want: BigRational, tol_rel: f64, what: &str| {
        let tol = want.clone() * rat(tol_rel);
        let diff = if have > want.clone() {
            have - want.clone()
        } else {
            want - have
        };
        assert!(diff <= tol, "{what}: volume off by more than {tol_rel:e} relative");
    };
    let mut scene = Scene::new();
    let block = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
    scene.add_void(cylinder([2.0, 2.0, 0.0], [0.0, 0.0, 2.0], 0.8, 24));
    let plc = scene.assemble();
    let want = plc_region_volume6(&plc, block);
    let params = MeshParams {
        maxh: 0.6,
        ..MeshParams::default()
    };
    let mut mesh = mesh_plc_with(&plc, &params);
    close(
        mesh_region_volume6(&mesh, block),
        want.clone(),
        1e-9,
        "after meshing",
    );
    optimize(
        &mut mesh,
        &OptimizeParams {
            maxh: params.maxh,
            ..OptimizeParams::default()
        },
    );
    // Fidelity snapping moves barrel vertices from the 24-gon chords onto
    // the true circle, which can only SHRINK the material volume (the bore
    // grows toward pi r^2); growth beyond rounding is a conformity bug.
    let after = mesh_region_volume6(&mesh, block);
    assert!(
        after <= want.clone() + want.clone() * rat(1e-9),
        "void shrank: material grew into the bore"
    );
    check_structure(&mesh);
}

/// CONCAVE geometry: the torus bore's big void tets sit wholesale on either
/// side of any facet plane, which the original centroid-side tile rule
/// misread as same-side (mass patch abandonment). The side reject is now
/// confined to the flat all-marked sliver complex; the torus must mesh with
/// exact region volume.
#[test]
fn torus_meshes_exactly() {
    let mut scene = Scene::new();
    let r = scene.add_solid(rapidmesh_geom::torus(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        2.0,
        0.5,
        16,
        8,
    ));
    let plc = scene.assemble();
    let want = plc_region_volume6(&plc, r);
    let params = MeshParams {
        maxh: 0.4,
        ..MeshParams::default()
    };
    let mesh = mesh_plc_with(&plc, &params);
    let have = mesh_region_volume6(&mesh, r);
    let tol = want.clone() * rat(1e-9);
    let diff = if have > want.clone() { have - want.clone() } else { want - have };
    assert!(diff <= tol, "torus volume off by more than 1e-9 relative");
    check_structure(&mesh);
}
