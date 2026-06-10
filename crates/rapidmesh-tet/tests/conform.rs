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
        mesh.faces.iter().filter(|f| f.face_tag == FaceTag(7)).count() >= 4,
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
