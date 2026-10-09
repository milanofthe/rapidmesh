//! The Rust API end to end: building, sizing, naming, periodic pairs,
//! meshing and what a solver reads off the result.

use rapidmesh::shapes::{Cuboid, Cylinder, Prism, Sheet, Sphere};
use rapidmesh::{EdgeFilter, FaceFilter, Geometry, MeshOptions, Scope, SurfaceOptions};

fn side(n: [f64; 3]) -> Scope {
    Scope::surf(Some(FaceFilter::normal(n)))
}

/// A unit cell: air over a substrate, a patch between them.
fn cell() -> Geometry {
    let mut g = Geometry::new(Some(0.6));
    let air = g.add(Cuboid::new([2.0, 2.0, 3.0])).unwrap();
    let sub = g
        .add_solid(Cuboid::new([2.0, 2.0, 0.5]), Some(0.3), false)
        .unwrap();
    g.label_solid(air, "air");
    g.label_solid(sub, "substrate");
    g.add_sheet(&Sheet::xy(1.0, 1.0, [0.5, 0.5, 0.5]), 3, None)
        .unwrap();
    g.label_tag(3, "patch");
    g
}

#[test]
fn periodic_sides_carry_the_same_points() {
    let mut g = cell();
    let sx = g
        .periodic(&side([-1.0, 0.0, 0.0]), &side([1.0, 0.0, 0.0]), None)
        .unwrap();
    let sy = g
        .periodic(&side([0.0, -1.0, 0.0]), &side([0.0, 1.0, 0.0]), None)
        .unwrap();
    assert_eq!(sx, [2.0, 0.0, 0.0]);
    assert_eq!(sy, [0.0, 2.0, 0.0]);
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!(!m.periodic_points.is_empty());
    for &[a, b] in &m.periodic_points {
        let (p, q) = (m.points[a], m.points[b]);
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let fits = |s: [f64; 3]| (0..3).all(|k| (d[k] - s[k]).abs() < 1e-9);
        assert!(fits(sx) || fits(sy), "pair {a} {b} is {d:?} apart");
    }
    let d = m.diagnostics();
    assert!(d.mesh.watertight);
    assert_eq!(d.defects().count(), 0);
}

#[test]
fn sets_and_msh_groups_follow_the_labels() {
    let mut g = cell();
    let top = Scope::surf(Some(FaceFilter::near([1.0, 1.0, 3.0])));
    g.name(&top, "port").unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let sets = m.sets();
    let cells: usize = sets.cells.values().map(Vec::len).sum();
    assert_eq!(cells, m.tets.len());
    assert!(!sets.cells["substrate"].is_empty());
    assert!(!sets.faces["patch"].is_empty());
    assert!(!sets.faces["port"].is_empty());
    // the port is on the outer boundary
    let boundary = &sets.faces["boundary"];
    assert!(sets.faces["port"].iter().all(|f| boundary.contains(f)));
    let mut msh = Vec::new();
    m.write_msh_to(&mut msh).unwrap();
    let msh = String::from_utf8(msh).unwrap();
    for name in ["\"air\"", "\"substrate\"", "\"patch\"", "\"port\""] {
        assert!(msh.contains(name), "no group {name}");
    }
    assert!(m.report().starts_with("Mesh("));
    assert!(m
        .viewer_json("cell", rapidmesh::Order::Linear)
        .contains("\"mesher\":\"rapidmesh\""));
}

#[test]
fn scopes_size_and_refuse_what_does_not_fit() {
    let coarse = cell().mesh(&MeshOptions::default()).unwrap();
    let mut g = cell();
    g.set_maxh_on(&Scope::region(Some(1)), 0.3).unwrap();
    let fine = g.mesh(&MeshOptions::default()).unwrap();
    assert!(fine.tets.len() > coarse.tets.len());
    assert!(g.set_tol_on(&Scope::region(None), 1e-3).is_err());
    let nowhere = Scope::surf(Some(FaceFilter::tag(99)));
    assert!(g.name(&nowhere, "none").is_err());
    let mismatch = g.periodic(&side([-1.0, 0.0, 0.0]), &side([0.0, 0.0, 1.0]), None);
    assert!(mismatch.is_err());
    let edges = Scope::region(Some(2))
        .surfs(None)
        .edges(Some(EdgeFilter::near([0.0, 0.0, 0.5])));
    assert_eq!(g.resolve(&edges).unwrap().len(), 1);
}

#[test]
fn unions_and_voids() {
    let mut g = Geometry::new(Some(0.5));
    g.add(Cuboid::new([3.0, 3.0, 3.0])).unwrap();
    let a = g
        .add(Cuboid::new([1.0, 1.0, 1.0]).at([0.5, 0.5, 0.5]))
        .unwrap();
    let b = g.add(Cylinder::new(0.4, 1.0).at([1.5, 1.0, 0.5])).unwrap();
    let u = g.union(&[a, b]).unwrap();
    g.label_solid(u, "metal");
    let hole = g.cut(Sphere::new(0.4).at([2.2, 2.2, 2.2])).unwrap();
    assert_eq!(hole.region, 0);
    assert!(g.add(Cylinder::new(0.4, 1.0).along([0.0; 3])).is_err());
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let groups = m.labels.region_groups();
    assert!(groups
        .iter()
        .any(|(n, rs)| n == "metal" && rs == &vec![u.region]));
    assert!(m
        .tet_regions
        .iter()
        .all(|r| r.0 != b.region || b.region == u.region));
}

/// Conductors meshed as regions of their own to place their walls, then
/// left out: a via on a strip, touching it, in air. The air keeps its tets
/// and points; the walls stay as boundary faces, region 0 on the conductor
/// side; the face between via and strip goes; the sets and the fidelity
/// check know the conductors are gone.
#[test]
fn regions_left_out_leave_their_walls() {
    let mut g = Geometry::new(Some(0.5));
    let air = g.add(Cuboid::new([4.0, 4.0, 3.0])).unwrap();
    let strip = g
        .add(Cuboid::new([3.0, 1.0, 0.5]).at([0.5, 1.5, 1.0]))
        .unwrap();
    let via = g.add(Cylinder::new(0.3, 1.0).at([2.0, 2.0, 1.5])).unwrap();
    g.label_solid(air, "air");
    g.label_solid(strip, "strip");
    g.label_solid(via, "via");
    let m = g.mesh(&MeshOptions::default()).unwrap();
    clean(&m);
    let w = m.without_regions(&[strip.region, via.region]);
    clean(&w);
    assert!(w.tet_regions.iter().all(|r| r.0 == air.region));
    assert_eq!(
        w.tets.len(),
        m.tet_regions.iter().filter(|r| r.0 == air.region).count()
    );
    let mut used = vec![false; w.points.len()];
    for t in &w.tets {
        for &v in t {
            used[v] = true;
        }
    }
    assert!(used.iter().all(|&u| u));
    assert!(w.plc_points <= w.points.len());
    assert!(w
        .faces
        .iter()
        .all(|f| f.regions.iter().any(|r| r.0 == air.region)));
    let walls = w
        .faces
        .iter()
        .filter(|f| f.regions.iter().any(|r| r.0 == 0))
        .count();
    assert!(walls > 0);
    let names: Vec<String> = w
        .labels
        .region_groups()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, ["air"]);
    // One at a time comes to the same.
    let one = m
        .without_regions(&[via.region])
        .without_regions(&[strip.region]);
    assert_eq!(one.tets, w.tets);
    clean(&one);
}

#[test]
fn surface_mesh_carries_the_names() {
    let mut g = cell();
    g.name(&side([0.0, 0.0, -1.0]), "ground").unwrap();
    let s = g.surface_mesh(&SurfaceOptions::default()).unwrap();
    let sets = s.sets();
    assert!(sets.cells.is_empty());
    assert!(!sets.faces["patch"].is_empty());
    assert!(!sets.faces["ground"].is_empty());
    assert!(!s.rwg_edges(false).is_empty());
    let mut msh = Vec::new();
    s.write_msh_to(&mut msh).unwrap();
    assert!(String::from_utf8(msh).unwrap().contains("\"ground\""));
}

/// The corners of polygon outlines are mesh points: a cross on the
/// coplanar top of a substrate (its rim lies between two parallel planes)
/// and an octagon (45 degree turns), in the volume and the surface mesh.
#[test]
fn polygon_corners_are_mesh_points() {
    let (c, h, w) = (1.0, 0.8, 0.2);
    let cross = vec![
        [c - w / 2.0, c - h],
        [c + w / 2.0, c - h],
        [c + w / 2.0, c - w / 2.0],
        [c + h, c - w / 2.0],
        [c + h, c + w / 2.0],
        [c + w / 2.0, c + w / 2.0],
        [c + w / 2.0, c + h],
        [c - w / 2.0, c + h],
        [c - w / 2.0, c + w / 2.0],
        [c - h, c + w / 2.0],
        [c - h, c - w / 2.0],
        [c - w / 2.0, c - w / 2.0],
    ];
    let octagon: Vec<[f64; 2]> = (0..8)
        .map(|k| {
            let a = std::f64::consts::PI / 8.0 * (1.0 + 2.0 * k as f64);
            [3.0 + 0.8 * a.cos(), 1.0 + 0.8 * a.sin()]
        })
        .collect();
    let mut g = Geometry::new(Some(0.3));
    g.add(Cuboid::new([4.0, 2.0, 2.0])).unwrap();
    g.add(Cuboid::new([4.0, 2.0, 0.5])).unwrap();
    g.add_sheet(&Sheet::polygon(cross.clone(), [0.0, 0.0, 0.5]), 1, None)
        .unwrap();
    g.add_sheet(&Sheet::polygon(octagon.clone(), [0.0, 0.0, 0.5]), 2, None)
        .unwrap();
    let corners: Vec<[f64; 3]> = cross
        .iter()
        .chain(&octagon)
        .map(|p| [p[0], p[1], 0.5])
        .collect();
    let has = |pts: &[[f64; 3]], q: [f64; 3]| {
        pts.iter()
            .any(|p| (0..3).all(|k| (p[k] - q[k]).abs() < 1e-9))
    };
    let vol = g.mesh(&MeshOptions::default()).unwrap();
    let surf = g.surface_mesh(&SurfaceOptions::default()).unwrap();
    for q in corners {
        assert!(
            has(&vol.points, q),
            "corner {q:?} missing in the volume mesh"
        );
        assert!(
            has(&surf.points, q),
            "corner {q:?} missing in the surface mesh"
        );
    }
}

/// A non-convex polygon sheet on a solid's face splits the face whichever
/// way its outline winds: the plane carries the face's area once.
#[test]
fn polygon_winding_does_not_matter() {
    let l = vec![
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 2.0],
        [2.0, 2.0],
        [2.0, 8.0],
        [0.0, 8.0],
    ];
    for pts in [l.clone(), l.iter().rev().copied().collect()] {
        let mut g = Geometry::new(Some(2.0));
        g.add(Cuboid::new([20.0, 20.0, 10.0]).at([-5.0, -5.0, -5.0]))
            .unwrap();
        g.add(Cuboid::new([12.0, 12.0, 1.0]).at([-2.0, -2.0, 0.0]))
            .unwrap();
        g.add_sheet(&Sheet::polygon(pts, [0.0, 0.0, 1.0]), 1, None)
            .unwrap();
        let m = g.model().unwrap();
        let v = |i: u32| m.plc.vertices[i as usize];
        let area: f64 = m
            .plc
            .triangles
            .iter()
            .filter(|t| t.iter().all(|&i| (v(i)[2] - 1.0).abs() < 1e-12))
            .map(|t| {
                let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
                let (u, w) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
                0.5 * (u[0] * w[1] - u[1] * w[0]).abs()
            })
            .sum();
        assert!((area - 144.0).abs() < 1e-9, "plane area {area}, want 144");
    }
}

/// A conductor where a port arm joins a ring (a rapidfem RFIC layout,
/// #366): two sharp notches in a prism, one ending in an edge of 0.1, its
/// walls tilted planes whose samples lie off their plane by their
/// rounding. It meshes watertight at sizes far above the notches and down
/// to a fraction of them (#369; their angle of 1.5 degrees leaves flat tets
/// in them at the coarse sizes).
#[test]
fn tilted_walls_by_a_sharp_notch_mesh() {
    let points = vec![
        [-57.669178, 109.886431],
        [-61.95484492300267, 107.41210027857939],
        [-57.0, 110.10240174572485],
        [-57.0, 102.4],
        [-67.0, 102.4],
        [-67.0, 104.32363451093639],
        [-62.044197836073565, 107.3605123549652],
        [-66.329852, 104.886189],
        [-67.0, 106.04691937652714],
        [-67.0, 112.4],
        [-59.12038774885818, 112.4],
    ];
    for h in [3.0, 2.0, 1.5, 1.0, 0.7, 0.3] {
        let mut g = Geometry::new(Some(40.0));
        g.add_solid(
            Prism::new(points.clone(), 1.26).at([0.0, 0.0, 4.365]),
            Some(h),
            false,
        )
        .unwrap();
        let m = g.mesh(&MeshOptions::default()).unwrap();
        assert!(m.diagnostics().mesh.watertight);
    }
}

/// Finely sized strips on a coarse solid's face under a coarse air box:
/// the air above is meshed up to the lid, none of it dropped (a lid
/// vertex fanned over the fine face once left tets centred above the lid).
#[test]
fn fine_sheet_on_a_face_keeps_the_air_above() {
    let mut g = Geometry::new(Some(1.0));
    g.add(Cuboid::new([22.0, 18.0, 6.5]).at([-11.0, -9.0, -3.0]))
        .unwrap();
    g.add_solid(
        Cuboid::new([16.0, 12.0, 0.5]).at([-8.0, -6.0, 0.0]),
        Some(0.4),
        false,
    )
    .unwrap();
    for k in 0..9 {
        let x = -6.0 + 1.5 * k as f64;
        g.add_sheet(&Sheet::xy(0.3, 8.0, [x, -4.0, 0.5]), 1, Some(0.1))
            .unwrap();
    }
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let vol: f64 = m
        .tets
        .iter()
        .map(|t| {
            let p = t.map(|v| m.points[v]);
            let d = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let (a, b, c) = (d(p[1], p[0]), d(p[2], p[0]), d(p[3], p[0]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                .abs()
                / 6.0
        })
        .sum();
    let want = 22.0 * 18.0 * 6.5;
    assert!(
        (vol - want).abs() < 1e-6 * want,
        "volume {vol}, want {want}"
    );
}

/// A face named by its origin (its solid and role) keeps that name when
/// shapes are added after the naming: the selection resolves when the mesh
/// is made, and by origin it means the same face (#128).
#[test]
fn a_name_by_origin_survives_later_shapes() {
    let mut g = Geometry::new(Some(0.4));
    let b = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    // Role 1 of a box: its +z face.
    g.name(&Scope::surf(Some(FaceFilter::origin(b.index, 1))), "lid")
        .unwrap();
    g.add(Cuboid::new([1.0, 1.0, 1.0]).at([-3.0, 0.0, 0.0]))
        .unwrap();
    g.add(Cuboid::new([1.0, 1.0, 0.5]).at([0.0, 0.0, 1.0]))
        .unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let lid = &m.sets().faces["lid"];
    assert!(!lid.is_empty());
    let v = m.view();
    for &f in lid {
        for p in v.topo.faces[f as usize] {
            let q = m.points[p as usize];
            assert!(
                (q[2] - 1.0).abs() < 1e-9 && (0.0..=1.0).contains(&q[0]),
                "a lid point off the lid: {q:?}"
            );
        }
    }
}

#[test]
fn a_nurbs_sheet_meshes_onto_its_surface() {
    // A bicubic bump over [0.4, 1.6]^2 at mid height of an air box.
    let ctrl: Vec<Vec<[f64; 3]>> = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| {
                    let (x, y) = (0.4 + 0.3 * i as f64, 0.4 + 0.3 * j as f64);
                    let bump = if (1..4).contains(&i) && (1..4).contains(&j) {
                        0.3
                    } else {
                        0.0
                    };
                    [x, y, 1.0 + bump]
                })
                .collect()
        })
        .collect();
    let sheet = Sheet::nurbs(ctrl, [3, 3], None, None).unwrap();
    let Sheet::Nurbs { surface, .. } = &sheet else {
        unreachable!()
    };
    let mut g = Geometry::new(Some(0.25));
    g.add(Cuboid::new([2.0, 2.0, 2.0])).unwrap();
    g.add_sheet(&sheet, 5, None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let d = m.diagnostics();
    assert!(d.mesh.watertight);
    assert_eq!(d.defects().count(), 0);
    let on: Vec<usize> = m
        .faces
        .iter()
        .filter(|f| f.face_tag.0 == 5)
        .flat_map(|f| f.tri)
        .collect();
    assert!(on.len() > 30, "the sheet is in the mesh");
    for v in on {
        let p = m.points[v];
        let t = surface.closest_param(p);
        let q = surface.eval(t[0], t[1]);
        let off = (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt();
        assert!(off < 1e-9, "vertex {v} at {p:?} is {off} off the surface");
    }
}

#[test]
fn a_revolved_spline_meshes_onto_its_surface() {
    use rapidmesh::shapes::{ProfileEdge, Revolve};
    use rapidmesh_geom::Surface;
    let vase = Revolve {
        edges: vec![
            ProfileEdge::Line,
            ProfileEdge::Spline(vec![[1.1, 0.6], [0.5, 1.3]]),
            ProfileEdge::Line,
            ProfileEdge::Line,
        ],
        ..Revolve::new(vec![[0.0, 0.0], [0.8, 0.0], [0.4, 2.0], [0.0, 2.0]])
    };
    let mut g = Geometry::new(Some(0.25));
    g.add(vase).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let d = m.diagnostics();
    assert!(d.mesh.watertight);
    assert_eq!(d.defects().count(), 0);
    let mut on = 0;
    for f in &m.faces {
        let Some(Surface::Revolved { profile, .. }) = &m.surfaces[f.surface as usize] else {
            continue;
        };
        for v in f.tri {
            let p = m.points[v];
            let q = [p[0].hypot(p[1]), p[2]];
            let c = profile.eval(profile.closest_param(q));
            let off = (c[0] - q[0]).hypot(c[1] - q[1]);
            assert!(off < 1e-9, "vertex {v} at {p:?} is {off} off the surface");
            on += 1;
        }
    }
    assert!(on > 100, "the spline wall is in the mesh");
}

fn meshed_volume(m: &rapidmesh::Mesh) -> f64 {
    m.tets
        .iter()
        .map(|t| {
            let [a, b, c, d] = t.map(|v| m.points[v]);
            let (u, v, w) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
            );
            (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0]))
                .abs()
                / 6.0
        })
        .sum()
}

fn clean(m: &rapidmesh::Mesh) {
    let d = m.diagnostics();
    assert!(d.mesh.watertight);
    assert_eq!(d.defects().count(), 0);
}

/// Faces go by name from Rust as from Python: a box's sides, the cut
/// faces numbered on over later cuts, a copy with the same names.
#[test]
fn roles_are_named_in_rust() {
    use rapidmesh::EdgePick;
    let mut g = Geometry::new(Some(0.2));
    let cube = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    assert_eq!(g.roles(cube), ["-z", "+z", "-y", "+y", "-x", "+x"]);
    let (top, side) = (g.role(cube, "+z").unwrap(), g.role(cube, "+x").unwrap());
    assert_eq!((top, side), (1, 5));
    assert!(g.role(cube, "top").is_err());
    g.chamfer(cube, &[EdgePick::Between(top, side)], 0.2, false)
        .unwrap();
    g.chamfer(cube, &[EdgePick::Between(0, 4)], 0.2, false)
        .unwrap();
    let first = g.role(cube, "chamfer0").unwrap();
    assert_eq!(first, 6);
    assert!(g.role(cube, "chamfer1").unwrap() > first);
    let copy = match g.copy(cube).unwrap() {
        rapidmesh::Object::Solid(s) => s,
        _ => unreachable!(),
    };
    assert_eq!(g.roles(copy), g.roles(cube));
    // A cone up to its apex has a side and a bottom, no top.
    let cone = g.add(rapidmesh::shapes::Cone::new(0.5, 0.0, 1.0)).unwrap();
    assert_eq!(g.roles(cone), ["side", "bottom"]);
}

#[test]
fn chamfered_cube_edges_cut_exact_planes() {
    use rapidmesh::EdgePick;
    // One edge: +z (role 1) against +x (role 5).
    let mut g = Geometry::new(Some(0.2));
    let cube = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    assert_eq!(
        g.chamfer(cube, &[EdgePick::Between(1, 5)], 0.2, false)
            .unwrap(),
        [(cube, 6)]
    );
    // Role 6 is the chamfer face, square to (1, 0, 1).
    let topo = g.topology().unwrap();
    let face = topo
        .faces
        .iter()
        .find(|f| f.owner == cube.index && f.role == 6)
        .unwrap();
    let n = face.normal;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (n[0].abs() - h).abs() < 1e-9 && n[1].abs() < 1e-9 && (n[2].abs() - h).abs() < 1e-9,
        "{n:?}"
    );
    let m = g.mesh(&MeshOptions::default()).unwrap();
    clean(&m);
    assert!((meshed_volume(&m) - (1.0 - 0.02)).abs() < 1e-9);
    // All twelve: L^3 - 6 L d^2 + 6 d^3.
    let mut g = Geometry::new(Some(0.2));
    let cube = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    assert_eq!(
        g.chamfer(cube, &[EdgePick::All], 0.2, false).unwrap().len(),
        12
    );
    let m = g.mesh(&MeshOptions::default()).unwrap();
    clean(&m);
    assert!(
        (meshed_volume(&m) - 0.808).abs() < 1e-9,
        "{}",
        meshed_volume(&m)
    );
}

#[test]
fn chamfered_rims_and_countersinks_cut_exact_cones() {
    use rapidmesh::EdgePick;
    use std::f64::consts::PI;
    // Meshed volumes fall short of the exact ones by the chords of the
    // barrel; the same barrel without the chamfer takes that out.
    let (r, h, d) = (0.5, 1.0, 0.15);
    let rod = |chamfer: bool| {
        let mut g = Geometry::new(Some(0.1));
        let rod = g.add(Cylinder::new(r, h)).unwrap();
        if chamfer {
            assert_eq!(
                g.chamfer(rod, &[EdgePick::Between(0, 1)], d, false)
                    .unwrap()
                    .iter()
                    .map(|f| f.1)
                    .collect::<Vec<_>>(),
                [3]
            );
        }
        let m = g.mesh(&MeshOptions::default()).unwrap();
        clean(&m);
        meshed_volume(&m)
    };
    // The top rim of a rod: the ring of the triangle (R, H), (R - d, H),
    // (R, H - d), pi d^2 (R - d / 3) by Pappus.
    let ring = PI * d * d * (r - d / 3.0);
    let cut = rod(false) - rod(true);
    assert!((cut - ring).abs() < 0.03 * ring, "{cut} vs {ring}");
    // A countersink on a hole through a plate: the ring of (R, H),
    // (R + d, H), (R, H - d) is left empty.
    let plate = |sink: bool| {
        let mut g = Geometry::new(Some(0.1));
        let plate = g
            .add(Cuboid::new([2.0, 2.0, 0.5]).at([-1.0, -1.0, 0.0]))
            .unwrap();
        let hole = g.cut(Cylinder::new(r, 1.0).at([0.0, 0.0, -0.25])).unwrap();
        if sink {
            let with = EdgePick::With(1, hole.index, 0);
            assert_eq!(g.chamfer(plate, &[with], d, true).unwrap().len(), 1);
        }
        let m = g.mesh(&MeshOptions::default()).unwrap();
        clean(&m);
        meshed_volume(&m)
    };
    let ring = PI * d * d * (r + d / 3.0);
    let cut = plate(false) - plate(true);
    assert!((cut - ring).abs() < 0.03 * ring, "{cut} vs {ring}");
}

#[test]
fn a_concave_edge_takes_no_chamfer() {
    use rapidmesh::shapes::Prism;
    use rapidmesh::EdgePick;
    let l = vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let mut g = Geometry::new(Some(0.3));
    let s = g.add(Prism::new(l, 1.0)).unwrap();
    let err = g.chamfer(s, &[EdgePick::All], 0.1, false).unwrap_err();
    assert!(err.to_string().contains("not convex"), "{err}");
}

#[test]
fn a_filleted_cube_edge_is_an_exact_quarter_cylinder() {
    use rapidmesh::EdgePick;
    use rapidmesh_geom::Surface;
    use std::f64::consts::PI;
    let r = 0.3;
    let cube = |fillet: bool| {
        let mut g = Geometry::new(Some(0.05));
        let cube = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
        if fillet {
            let faces = g
                .fillet(cube, &[EdgePick::Between(1, 5)], r, false)
                .unwrap();
            assert_eq!(faces, [(cube, 6)]);
        }
        let m = g.mesh(&MeshOptions::default()).unwrap();
        clean(&m);
        m
    };
    let (plain, round) = (cube(false), cube(true));
    // The corner square less the quarter disc comes off along the edge.
    let want = (1.0 - PI / 4.0) * r * r;
    let cut = meshed_volume(&plain) - meshed_volume(&round);
    assert!((cut - want).abs() < 0.03 * want, "{cut} vs {want}");
    // Every point of the round lies on the cylinder about (1 - r, y, 1 - r).
    let mut on = 0;
    for f in &round.faces {
        if !matches!(
            round.surfaces[f.surface as usize],
            Some(Surface::Cylinder { .. })
        ) {
            continue;
        }
        for v in f.tri {
            let p = round.points[v];
            let off = ((p[0] - (1.0 - r)).hypot(p[2] - (1.0 - r)) - r).abs();
            assert!(off < 1e-9, "{p:?} is {off} off the round");
            on += 1;
        }
    }
    assert!(on > 100);
}

#[test]
fn fillets_round_rims_and_every_cube_edge() {
    use rapidmesh::EdgePick;
    use std::f64::consts::PI;
    // The top rim of a rod: the spandrel (1 - pi/4) r^2 turned about the
    // axis at R - r (10 - 3 pi) / (12 - 3 pi), its centroid (Pappus).
    let (big, h, r) = (0.5, 1.0, 0.15);
    let rod = |fillet: bool| {
        let mut g = Geometry::new(Some(0.05));
        let rod = g.add(Cylinder::new(big, h)).unwrap();
        if fillet {
            g.fillet(rod, &[EdgePick::Between(0, 1)], r, false).unwrap();
        }
        let m = g.mesh(&MeshOptions::default()).unwrap();
        clean(&m);
        meshed_volume(&m)
    };
    let centroid = big - r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
    let want = 2.0 * PI * centroid * (1.0 - PI / 4.0) * r * r;
    // The mesh's chords lie inside the torus, a sagitta h^2 / 8 r deep:
    // the round loses some percent more than the exact spandrel.
    let cut = rod(false) - rod(true);
    assert!((cut - want).abs() < 0.08 * want, "{cut} vs {want}");
    // Every edge of a cube; the rounds cross where three meet.
    let mut g = Geometry::new(Some(0.1));
    let cube = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    assert_eq!(
        g.fillet(cube, &[EdgePick::All], 0.2, false).unwrap().len(),
        12
    );
    clean(&g.mesh(&MeshOptions::default()).unwrap());
    // The rim of a hole through a plate, left empty.
    let mut g = Geometry::new(Some(0.08));
    let plate = g
        .add(Cuboid::new([2.0, 2.0, 0.5]).at([-1.0, -1.0, 0.0]))
        .unwrap();
    let hole = g
        .cut(Cylinder::new(0.4, 1.0).at([0.0, 0.0, -0.25]))
        .unwrap();
    g.fillet(plate, &[EdgePick::With(1, hole.index, 0)], 0.1, true)
        .unwrap();
    clean(&g.mesh(&MeshOptions::default()).unwrap());
}

fn region_volume(m: &rapidmesh::Mesh, region: u32) -> f64 {
    let tets: Vec<[usize; 4]> = m
        .tets
        .iter()
        .zip(&m.tet_regions)
        .filter(|(_, r)| r.0 == region)
        .map(|(t, _)| *t)
        .collect();
    tets.iter()
        .map(|t| {
            let [a, b, c, d] = t.map(|v| m.points[v]);
            let (u, v, w) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
            );
            (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0]))
                .abs()
                / 6.0
        })
        .sum()
}

#[test]
fn transforms_move_solids_and_keep_their_faces() {
    use rapidmesh::Transform;
    use std::f64::consts::FRAC_PI_2;
    let mut g = Geometry::new(Some(0.3));
    let b = g.add(Cuboid::new([1.0, 2.0, 3.0])).unwrap();
    g.transform(b, Transform::Translate([5.0, 0.0, 0.0]))
        .unwrap();
    g.transform(
        b,
        Transform::Rotate {
            angle: FRAC_PI_2,
            axis: [0.0, 0.0, 1.0],
            center: [5.0, 0.0, 0.0],
        },
    )
    .unwrap();
    g.transform(
        b,
        Transform::Mirror {
            normal: [0.0, 0.0, 1.0],
            point: [0.0, 0.0, 0.0],
        },
    )
    .unwrap();
    g.transform(
        b,
        Transform::Stretch {
            factors: [1.0, 1.0, 2.0],
            center: [0.0, 0.0, 0.0],
        },
    )
    .unwrap();
    let topo = g.topology().unwrap();
    let [lo, hi] = topo.region_bbox[0];
    let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12);
    assert!(
        close(lo, [3.0, 0.0, -6.0]) && close(hi, [5.0, 1.0, 0.0]),
        "{lo:?} {hi:?}"
    );
    // The +z face, mirrored and stretched: now the bottom at z = -6.
    let top = topo
        .faces
        .iter()
        .find(|f| f.owner == b.index && f.role == 1)
        .unwrap();
    assert!((top.centroid[2] + 6.0).abs() < 1e-12, "{:?}", top.centroid);
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!((region_volume(&m, b.region) - 12.0).abs() < 1e-9);
    let bad = Transform::Rotate {
        angle: 1.0,
        axis: [0.0; 3],
        center: [0.0; 3],
    };
    assert!(g.transform(b, bad).is_err());
}

#[test]
fn arrays_copy_and_intersect_cuts_exactly() {
    use rapidmesh::{Object, Transform};
    use std::f64::consts::FRAC_PI_2;
    let mut g = Geometry::new(Some(0.3));
    let b = g.add(Cuboid::new([1.0, 1.0, 1.0])).unwrap();
    let row = g
        .array(b, 3, Transform::Translate([2.0, 0.0, 0.0]))
        .unwrap();
    assert_eq!(row.len(), 3);
    let rod = g.add(Cylinder::new(0.2, 1.0).at([20.0, 3.0, 0.0])).unwrap();
    let ring = g
        .array(
            rod,
            4,
            Transform::Rotate {
                angle: FRAC_PI_2,
                axis: [0.0, 0.0, 1.0],
                center: [20.0, 0.0, 0.0],
            },
        )
        .unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    for o in row {
        let Object::Solid(s) = o else { unreachable!() };
        assert!((region_volume(&m, s.region) - 1.0).abs() < 1e-9);
    }
    assert_eq!(ring.len(), 4);
    // Two cubes overlapping in a unit cube.
    let mut g = Geometry::new(Some(0.3));
    let a = g.add(Cuboid::new([2.0, 2.0, 2.0])).unwrap();
    let t = g
        .add(Cuboid::new([2.0, 2.0, 2.0]).at([1.0, 1.0, 1.0]))
        .unwrap();
    g.intersect(a, &[t]).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!((region_volume(&m, a.region) - 1.0).abs() < 1e-9);
    assert_eq!(region_volume(&m, t.region), 0.0, "the tool is used up");
}

/// Sheet booleans in one plane: a plate with a round hole extrudes into a
/// block with an exact bore; a ground with a tapered slot and a disc cut
/// out (a Vivaldi antenna's) meshes with the area it should have.
#[test]
fn sheet_booleans_keep_round_rims() {
    use rapidmesh::BoolOp;
    use rapidmesh_geom::Surface;
    use std::f64::consts::PI;
    let mut g = Geometry::new(Some(0.25));
    let plate = g
        .add_sheet(&Sheet::xy(4.0, 4.0, [0.0; 3]), 1, None)
        .unwrap();
    let hole = g
        .add_sheet(&Sheet::disc(1.0, [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]), 2, None)
        .unwrap();
    let plate = g.sheet_boolean(BoolOp::Difference, plate, &[hole]).unwrap();
    let block = g.extrude(plate, [0.0, 0.0, 1.0], None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    clean(&m);
    let bore: Vec<usize> = m
        .faces
        .iter()
        .filter(|f| {
            matches!(
                m.surfaces[f.surface as usize],
                Some(Surface::Cylinder { .. })
            )
        })
        .flat_map(|f| f.tri)
        .collect();
    assert!(!bore.is_empty());
    for v in bore {
        let q = m.points[v];
        assert!(
            ((q[0] - 2.0).hypot(q[1] - 2.0) - 1.0).abs() < 1e-9,
            "{q:?} is off the bore"
        );
    }
    let vol = region_volume(&m, block.region);
    assert!((vol - (16.0 - PI)).abs() < 0.01 * PI, "{vol}");
    assert!(
        m.faces.iter().all(|f| f.face_tag.0 != 2),
        "the tool is used up"
    );

    let mut g = Geometry::new(Some(0.5));
    let ground = g
        .add_sheet(&Sheet::xy(10.0, 6.0, [0.0; 3]), 1, None)
        .unwrap();
    let taper = g
        .add_sheet(
            &Sheet::polygon(vec![[3.0, 3.0], [10.5, 1.0], [10.5, 5.0]], [0.0; 3]),
            1,
            None,
        )
        .unwrap();
    let disc = g
        .add_sheet(&Sheet::disc(1.0, [1.5, 3.0, 0.0], [0.0, 0.0, 1.0]), 1, None)
        .unwrap();
    g.sheet_boolean(BoolOp::Difference, ground, &[taper, disc])
        .unwrap();
    let sm = g.surface_mesh(&SurfaceOptions::default()).unwrap();
    let area: f64 = sm
        .faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.tri.map(|v| sm.points[v]);
            let (u, w) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
            0.5 * (u[0] * w[1] - u[1] * w[0]).abs()
        })
        .sum();
    // The taper's part on the ground (to x = 10) and the disc, whose rim
    // the mesh follows by chords at the default angle tolerance.
    let taper_area = 0.5 * 7.0 * (4.0 * 7.0 / 7.5);
    let want = 60.0 - taper_area - PI;
    assert!((area - want).abs() < 0.07 * PI, "{area} against {want}");
}

/// A disc cut out across a plate's edge leaves an arc. Its ends are where
/// the outline leaves the plate's edge for the disc's, so corners; at x = 4
/// they are samples of the disc's rim, at 4.3 crossings between them, which
/// the B-rep edge passes by nearness to the rim (#376). Either way the arc
/// keeps its circle.
#[test]
fn a_rim_cut_across_an_edge_keeps_its_circle() {
    use rapidmesh::BoolOp;
    for x in [4.0, 4.3] {
        let mut g = Geometry::new(Some(0.5));
        let plate = g
            .add_sheet(&Sheet::xy(4.0, 4.0, [0.0; 3]), 1, None)
            .unwrap();
        let disc = g
            .add_sheet(&Sheet::disc(1.0, [x, 2.0, 0.0], [0.0, 0.0, 1.0]), 2, None)
            .unwrap();
        g.sheet_boolean(BoolOp::Difference, plate, &[disc]).unwrap();
        let m = g.model().unwrap();
        let arcs = m
            .brep
            .edges
            .iter()
            .filter(|e| {
                matches!(&e.curve, rapidmesh_brep::Curve::Piece { curve, .. } if curve.as_circle().is_some())
            })
            .count();
        assert_eq!(arcs, 1, "x {x}: the arc the cut leaves");
    }
}

/// A polygon in any plane: given by its plane's axes, it lies where they
/// say, and its booleans with a disc in that plane keep the points exact
/// where the plane is square to an axis.
#[test]
fn polygons_lie_in_any_plane() {
    use rapidmesh::BoolOp;
    let l = vec![
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let mut g = Geometry::new(Some(0.5));
    let s = g
        .add_sheet(
            &Sheet::polygon_on(l.clone(), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            1,
            None,
        )
        .unwrap();
    let d = g
        .add_sheet(&Sheet::disc(0.3, [1.5, 2.0, 3.5], [0.0, 1.0, 0.0]), 1, None)
        .unwrap();
    g.sheet_boolean(BoolOp::Difference, s, &[d]).unwrap();
    let sm = g.surface_mesh(&SurfaceOptions::default()).unwrap();
    assert!(
        sm.points.iter().all(|p| p[1] == 2.0),
        "all in the plane y = 2"
    );
    for c in [
        [1.0, 2.0, 3.0],
        [4.0, 2.0, 3.0],
        [2.0, 2.0, 4.0],
        [1.0, 2.0, 5.0],
    ] {
        assert!(sm.points.contains(&c), "{c:?} is a corner");
    }
    // Tilted: the L's area, wherever its axes put it.
    let (u, v) = ([0.6, 0.8, 0.0], [0.0, 0.0, 1.0]);
    let mut g = Geometry::new(Some(0.5));
    g.add_sheet(&Sheet::polygon_on(l, [0.0; 3], u, v), 1, None)
        .unwrap();
    let sm = g.surface_mesh(&SurfaceOptions::default()).unwrap();
    let area: f64 = sm
        .faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.tri.map(|i| sm.points[i]);
            let (p, q) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let n = [
                p[1] * q[2] - p[2] * q[1],
                p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0],
            ];
            0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
        })
        .sum();
    assert!((area - 4.0).abs() < 1e-9, "{area}");
    let flat = Sheet::polygon_on(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.0; 3], u, u);
    assert!(Geometry::new(None).add_sheet(&flat, 1, None).is_err());
}

#[test]
fn sheets_extrude_into_solids_on_them() {
    use rapidmesh::Transform;
    use rapidmesh_geom::Surface;
    use std::f64::consts::PI;
    // A rectangle swept obliquely: area times the height.
    let mut g = Geometry::new(Some(0.3));
    let s = g
        .add_sheet(&Sheet::xy(2.0, 1.0, [0.0; 3]), 7, None)
        .unwrap();
    let p = g.extrude(s, [0.5, 0.2, 1.5], None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!((region_volume(&m, p.region) - 3.0).abs() < 1e-9);
    assert!(m.faces.iter().any(|f| f.face_tag.0 == 7), "the sheet stays");
    // A polygon with a hole, turned first, swept along its turned normal.
    let mut g = Geometry::new(Some(0.3));
    let sq = Sheet::Polygon {
        points: vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        holes: vec![vec![[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]],
        position: [0.0; 3],
        u: [1.0, 0.0, 0.0],
        v: [0.0, 1.0, 0.0],
    };
    let s = g.add_sheet(&sq, 1, None).unwrap();
    g.transform(
        s,
        Transform::Rotate {
            angle: PI / 2.0,
            axis: [1.0, 0.0, 0.0],
            center: [0.0; 3],
        },
    )
    .unwrap();
    let p = g.extrude(s, [0.0, -0.5, 0.0], None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!((region_volume(&m, p.region) - 1.5).abs() < 1e-9);
    // A disc: its walls are one cylinder, which the mesh follows.
    let mut g = Geometry::new(Some(0.1));
    let d = g
        .add_sheet(&Sheet::disc(0.5, [0.0; 3], [0.0, 0.0, 1.0]), 2, None)
        .unwrap();
    let c = g.extrude(d, [0.0, 0.0, -1.0], None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let barrel: Vec<_> = m
        .faces
        .iter()
        .filter(|f| {
            matches!(
                m.surfaces[f.surface as usize],
                Some(Surface::Cylinder { .. })
            )
        })
        .flat_map(|f| f.tri)
        .collect();
    assert!(!barrel.is_empty());
    for v in barrel {
        let q = m.points[v];
        assert!(
            (q[0].hypot(q[1]) - 0.5).abs() < 1e-9,
            "{q:?} is off the cylinder"
        );
    }
    let vol = region_volume(&m, c.region);
    assert!((vol - PI * 0.25).abs() < 0.02 * PI * 0.25, "{vol}");
    // Oblique over a disc, or in the sheet's plane: refused.
    assert!(g.extrude(d, [0.3, 0.0, 1.0], None).is_err());
    assert!(g.extrude(d, [1.0, 0.0, 0.0], None).is_err());
}

#[test]
fn msh_files_read_back_as_they_were_written() {
    let mut g = cell();
    let top = Scope::surf(Some(FaceFilter::near([1.0, 1.0, 3.0])));
    g.name(&top, "port").unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let mut first = Vec::new();
    m.write_msh_to(&mut first).unwrap();
    let back = rapidmesh::read_msh(first.as_slice()).unwrap();
    assert_eq!(back.tets.len(), m.tets.len());
    assert_eq!(back.points.len(), m.points.len());
    let sets = back.sets();
    for name in ["air", "substrate"] {
        // The file groups tets by region: the same cells, renumbered.
        assert_eq!(sets.cells[name].len(), m.sets().cells[name].len(), "{name}");
    }
    for name in ["patch", "port"] {
        assert!(!sets.faces[name].is_empty(), "{name}");
    }
    // Written again, the same file.
    let mut second = Vec::new();
    back.write_msh_to(&mut second).unwrap();
    assert!(first == second, "the rewritten file differs");
}

#[test]
fn msh_2_2_files_and_refusals() {
    // Two tets over a shared face, in two physical volumes, one triangle
    // with a named surface group.
    let old = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
        $PhysicalNames\n3\n3 1 \"left\"\n3 2 \"right side\"\n2 5 \"wall\"\n$EndPhysicalNames\n\
        $Nodes\n5\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n5 1 1 1\n$EndNodes\n\
        $Elements\n3\n1 4 2 1 1 1 2 3 4\n2 4 2 2 2 2 3 4 5\n3 2 2 5 7 1 2 3\n$EndElements\n";
    let m = rapidmesh::read_msh(old.as_bytes()).unwrap();
    assert_eq!(m.tets.len(), 2);
    let sets = m.sets();
    assert_eq!(sets.cells["left"].len(), 1);
    assert_eq!(sets.cells["right side"].len(), 1);
    assert!(!sets.faces["wall"].is_empty());
    // The shared face is an interface with both regions on it.
    let shared: Vec<_> = m
        .faces
        .iter()
        .filter(|f| {
            let mut t = f.tri;
            t.sort_unstable();
            t == [1, 2, 3]
        })
        .collect();
    assert_eq!(shared.len(), 1);
    let mut rs = shared[0].regions.map(|r| r.0);
    rs.sort_unstable();
    assert_eq!(rs, [1, 2]);
    let binary = "$MeshFormat\n4.1 1 8\n$EndMeshFormat\n";
    assert!(rapidmesh::read_msh(binary.as_bytes()).is_err());
    assert!(rapidmesh::read_msh("hello".as_bytes()).is_err());
}

/// CAD parts from STEP files mesh bottom-up: every solid a region of its
/// own, watertight, no defect and no poor tet.
#[test]
fn step_parts_mesh_clean() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rapidmesh-step/fixtures");
    for (name, maxh, solids) in [
        ("bracket", 3.0, 1),
        ("assembly", 1.5, 3),
        ("flange", 0.6, 5),
    ] {
        let mut g = Geometry::new(Some(maxh));
        let s = g
            .import_step(dir.join(format!("{name}.step")), None)
            .unwrap();
        assert_eq!(s.len(), solids, "{name}");
        let m = g.mesh(&MeshOptions::default()).unwrap();
        let d = m.diagnostics();
        assert!(d.mesh.watertight, "{name}");
        assert!(
            d.mesh.defects.is_empty(),
            "{name}: {:?}",
            &d.mesh.defects[..3.min(d.mesh.defects.len())]
        );
        assert!(
            d.mesh.quality.min_dihedral_deg > 15.0,
            "{name}: {}",
            d.mesh.quality.min_dihedral_deg
        );
        let mut regions: Vec<u32> = m.tet_regions.iter().map(|r| r.0).collect();
        regions.sort_unstable();
        regions.dedup();
        assert_eq!(regions.len(), solids, "{name}");
    }
}

/// The bodies of a STEP file are named as the file names them, so the
/// mesh's sets and physical groups carry the names of the parts.
#[test]
fn step_bodies_take_their_names() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rapidmesh-step/fixtures");
    let mut g = Geometry::new(Some(1.5));
    let solids = g.import_step(dir.join("assembly.step"), None).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let groups = m.labels.region_groups();
    assert_eq!(groups.len(), solids.len());
    // Each name on the part it names: the plate in 0..5, the boss above
    // it, the base below.
    for (name, z) in [("plate", 2.5), ("boss", 11.0), ("base", -2.0)] {
        let (_, regions) = groups.iter().find(|(n, _)| n == name).expect(name);
        assert_eq!(regions.len(), 1, "{name}");
        let at = m
            .tets
            .iter()
            .zip(&m.tet_regions)
            .filter(|(_, r)| r.0 == regions[0])
            .map(|(t, _)| t.iter().map(|&v| m.points[v][2]).sum::<f64>() / 4.0)
            .sum::<f64>()
            / m.tet_regions.iter().filter(|r| r.0 == regions[0]).count() as f64;
        assert!((at - z).abs() < 1.5, "{name} at height {at}");
    }
}

/// Every edge of the STEP parts takes its curve from the file, found by
/// its samples, closed B-splines too (#374): no edge falls back to the
/// meeting of its faces or to its chain.
#[test]
fn step_edges_take_their_file_curves() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rapidmesh-step/fixtures");
    for name in ["loft", "bracket", "flange", "turned_part"] {
        let mut g = Geometry::new(None);
        g.import_step(dir.join(format!("{name}.step")), None)
            .unwrap();
        let m = g.model().unwrap();
        for (i, e) in m.brep.edges.iter().enumerate() {
            assert!(
                matches!(e.curve, rapidmesh_brep::Curve::Piece { .. }),
                "{name}: edge {i} is {:?}",
                e.curve
            );
        }
    }
}

/// A STEP file read once gives its unit and its bodies, which go in one
/// by one like any shape: in another order, some left out, moved.
#[test]
fn step_bodies_go_in_one_by_one() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rapidmesh-step/fixtures");
    let step = rapidmesh::read_step(dir.join("assembly.step")).unwrap();
    assert_eq!(step.metres_per_unit, 1e-3);
    let at = |name: &str| step.bodies.iter().position(|b| b.name == name).expect(name);
    let mut g = Geometry::new(Some(1.5));
    g.add_body(&step.bodies[at("base")], None, false);
    let plate = g.add_body(&step.bodies[at("plate")], None, false);
    g.transform(plate, rapidmesh::Transform::Translate([100.0, 0.0, 0.0]))
        .unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    assert!(m.diagnostics().mesh.watertight);
    let groups = m.labels.region_groups();
    let mut names: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["base", "plate"]);
    let x = m
        .tets
        .iter()
        .zip(&m.tet_regions)
        .filter(|(_, r)| r.0 == plate.region)
        .map(|(t, _)| m.points[t[0]][0])
        .fold(f64::INFINITY, f64::min);
    assert!(x > 50.0, "the plate moved, its tets from x {x}");
}

#[test]
fn only_tets_on_a_curved_surface_are_curved() {
    // A cylinder: tets with an edge on its mantle are curved, every other
    // tet keeps its mid-edge nodes in the middle of its edges, and the two
    // kinds meet on straight faces only.
    let mut g = Geometry::new(Some(0.3));
    g.add(Cylinder::new(1.0, 1.0)).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    let so = m.second_order();
    assert_eq!(so.curved_tets.len(), so.tets.len());
    let n = so.curved_tets.iter().filter(|&&c| c).count();
    assert!(n > 0 && n < so.tets.len());
    let mid = |t: &[u32; 10], e: usize| {
        let [i, j] = rapidmesh::TET10_EDGES[e];
        let (a, b) = (so.points[t[i] as usize], so.points[t[j] as usize]);
        std::array::from_fn::<f64, 3, _>(|k| 0.5 * (a[k] + b[k]))
    };
    let mut face_kinds: std::collections::HashMap<[u32; 3], Vec<bool>> = Default::default();
    for (t, &c) in so.tets.iter().zip(&so.curved_tets) {
        let straight = (0..6).all(|e| so.points[t[4 + e] as usize] == mid(t, e));
        assert_eq!(straight, !c);
        for f in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut k = f.map(|i| t[i]);
            k.sort_unstable();
            face_kinds.entry(k).or_default().push(c);
        }
    }
    // A face between a curved and a straight tet is straight: its three
    // mid-edge nodes in the middle of its edges.
    for (t, &c) in so.tets.iter().zip(&so.curved_tets) {
        if !c {
            continue;
        }
        for (f, edges) in [
            ([0, 1, 2], [0, 1, 2]),
            ([0, 1, 3], [0, 4, 3]),
            ([0, 2, 3], [2, 5, 3]),
            ([1, 2, 3], [1, 5, 4]),
        ] {
            let mut k = f.map(|i| t[i]);
            k.sort_unstable();
            if face_kinds[&k].contains(&false) {
                assert!(edges
                    .iter()
                    .all(|&e| so.points[t[4 + e] as usize] == mid(t, e)));
            }
        }
    }
}

/// A cube as a mesh boolean writes it (manifold, issue #1 of the public
/// repo): a corner on the front top edge, with the flat cap over it that
/// keeps the edge matched. It imports closed and meshes.
#[test]
fn a_mesh_boolean_cube_with_a_cap_meshes() {
    let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\nv 0.5 0 1\n\
               f 1 4 3\nf 1 3 2\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 9\nf 1 9 5\nf 6 5 9\n\
               f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n";
    let path = std::env::temp_dir().join("rapidmesh_facade_cap_cube.obj");
    std::fs::write(&path, obj).unwrap();
    let mut g = Geometry::new(Some(0.3));
    g.add(rapidmesh::shapes::Import::new(&path)).unwrap();
    let m = g.mesh(&MeshOptions::default()).unwrap();
    clean(&m);
    assert!((meshed_volume(&m) - 1.0).abs() < 1e-9);
}

/// Triangles that make no solid are an error, not a panic.
#[test]
fn triangles_that_make_no_solid_are_refused() {
    use rapidmesh::shapes::Triangles;
    let tet = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for (verts, tris) in [
        (tet.clone(), vec![]),
        (
            tet.clone(),
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 9]],
        ),
        (tet.clone(), vec![[0, 2, 1], [0, 1, 3], [0, 3, 2]]),
        (
            vec![[0.0; 3]; 4],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        ),
    ] {
        let mut g = Geometry::new(Some(0.3));
        let added = g.add(Triangles { verts, tris });
        assert!(
            matches!(added, Err(rapidmesh::Error::Invalid(_))),
            "{added:?}"
        );
    }
}
