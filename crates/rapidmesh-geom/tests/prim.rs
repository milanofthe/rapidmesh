//! Primitive invariants: watertightness, exact volumes where analytic,
//! polygon triangulation with holes, sheet embedding in the arrangement.

use num_rational::BigRational;
use num_traits::Zero;
use rapidmesh_csg::{arrange, boolean, BoolOp, Tri};
use rapidmesh_exact::Point3;
use rapidmesh_geom::{
    cylinder, extrude_polygon, frustum, sheet_polygon, sheet_rect, solid_box, sphere,
    triangulate_polygon, Faceted,
};
use rapidmesh_testutil::{assert_watertight, check_invariants, rat, volume6};

/// Indexes a Faceted's triangles by exact (bitwise) vertex identity so the
/// watertightness/volume helpers can run on it. Builders construct each
/// shared vertex once, so bitwise identity is exact identity here.
fn index_mesh(f: &Faceted) -> (Vec<Point3>, Vec<[usize; 3]>) {
    let mut map: std::collections::HashMap<[u64; 3], usize> = std::collections::HashMap::new();
    let mut verts: Vec<Point3> = Vec::new();
    let mut tris = Vec::new();
    for t in &f.tris {
        let idx: [usize; 3] = std::array::from_fn(|k| {
            let v = t.v[k];
            let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
            *map.entry(key).or_insert_with(|| {
                verts.push(Point3::Explicit(v));
                verts.len() - 1
            })
        });
        tris.push(idx);
    }
    (verts, tris)
}

fn solid_volume6(f: &Faceted) -> BigRational {
    let (verts, tris) = index_mesh(f);
    assert_watertight(&tris);
    volume6(&verts, &tris)
}

/// Exact rational shoelace area (times 2) of the 2D triangles.
fn area2_of_tris(tris: &[[[f64; 2]; 3]]) -> BigRational {
    tris.iter().fold(BigRational::zero(), |acc, t| {
        let term = (rat(t[1][0]) - rat(t[0][0])) * (rat(t[2][1]) - rat(t[0][1]))
            - (rat(t[1][1]) - rat(t[0][1])) * (rat(t[2][0]) - rat(t[0][0]));
        acc + term
    })
}

const L_SHAPE: [[f64; 2]; 6] = [
    [0.0, 0.0],
    [3.0, 0.0],
    [3.0, 1.0],
    [1.0, 1.0],
    [1.0, 2.0],
    [0.0, 2.0],
];

#[test]
fn polygon_l_shape_exact_area() {
    let tris = triangulate_polygon(&L_SHAPE, &[]);
    // Area = 3*1 + 1*1 = 4 (times 2 = 8), every triangle counterclockwise.
    assert_eq!(area2_of_tris(&tris), rat(8.0));
    for t in &tris {
        let d = (t[1][0] - t[0][0]) * (t[2][1] - t[0][1])
            - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]);
        assert!(d > 0.0, "output triangle not counterclockwise");
    }
}

#[test]
fn polygon_with_hole_exact_area() {
    let outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
    let tris = triangulate_polygon(&outer, &[hole]);
    // 16 - 4 = 12 (times 2 = 24).
    assert_eq!(area2_of_tris(&tris), rat(24.0));
}

#[test]
fn box_exact_volume() {
    let b = solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
    assert_eq!(solid_volume6(&b), rat(6.0 * 24.0));
}

#[test]
fn cylinder_is_an_exact_prism() {
    // Axis along +z: the polyhedral cylinder is a prism, so its volume is
    // exactly (n-gon area) * h, computable rationally from the tessellated
    // ring.
    let c = cylinder([0.5, -0.25, 1.0], [0.0, 0.0, 2.0], 1.5, 16);
    let v6 = solid_volume6(&c);
    // Ring vertices: bottom-cap triangles share the base center; collect the
    // ring from the barrel triangles at z == 1.0.
    let mut ring: Vec<[f64; 2]> = Vec::new();
    for (t, &s) in c.tris.iter().zip(&c.face_surface) {
        if s == 0 {
            // Barrel triangles (bottom[i], bottom[j], top[j]): take vertex 0.
            if t.v[0][2] == 1.0 && t.v[1][2] == 1.0 {
                ring.push([t.v[0][0], t.v[0][1]]);
            }
        }
    }
    assert_eq!(ring.len(), 16, "expected one barrel quad pair per segment");
    let area2: BigRational = (0..ring.len()).fold(BigRational::zero(), |acc, i| {
        let j = (i + 1) % ring.len();
        acc + (rat(ring[i][0]) * rat(ring[j][1]) - rat(ring[j][0]) * rat(ring[i][1]))
    });
    // 6V = 3 * area2 * h with h = 2 exactly.
    assert_eq!(v6, area2 * rat(6.0), "cylinder volume must equal prism volume");
}

#[test]
fn extruded_polygon_with_hole_is_an_exact_prism() {
    let outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
    let p = extrude_polygon(
        &outer,
        &[hole],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    );
    // V = 12 * 0.5 = 6 exactly.
    assert_eq!(solid_volume6(&p), rat(36.0));
}

#[test]
fn extruded_polygon_orientation_input_invariant() {
    // Clockwise outer ring and counterclockwise hole get normalized.
    let outer: Vec<[f64; 2]> = L_SHAPE.iter().rev().copied().collect();
    let p = extrude_polygon(
        &outer,
        &[],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    assert_eq!(solid_volume6(&p), rat(24.0));
}

#[test]
fn frustum_and_cone_are_watertight() {
    let f = frustum([0.0, 0.0, 0.0], [0.0, 0.5, 2.0], 1.0, 0.4, 12);
    assert!(solid_volume6(&f) > BigRational::zero());
    let cone = frustum([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], 0.75, 0.0, 9);
    assert!(solid_volume6(&cone) > BigRational::zero());
}

#[test]
fn sphere_watertight_with_sane_volume() {
    let s = sphere([1.0, 2.0, 3.0], 2.0, 24, 12);
    let v6 = solid_volume6(&s);
    let exact = 6.0 * 4.0 / 3.0 * std::f64::consts::PI * 8.0;
    // Inscribed polyhedron: below the smooth volume, but close.
    let approx = rat(0.95 * exact);
    let upper = rat(exact);
    assert!(v6 > approx && v6 < upper, "sphere volume out of range");
}

#[test]
fn primitive_booleans_stay_watertight() {
    // Cylinder drilled through a box: classic via/anti-pad situation.
    let b = solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]);
    let c = cylinder([0.0, 0.0, -0.5], [0.0, 0.0, 2.0], 0.75, 16);
    let res = boolean(&b.to_solid(), &c.to_solid(), BoolOp::Difference);
    assert_watertight(&res.triangles);
    let v6 = volume6(&res.vertices, &res.triangles);
    // Box minus the prism volume of the polyhedral cylinder cross-section.
    assert!(v6 > rat(6.0 * (16.0 - 0.75 * 0.75 * std::f64::consts::PI)) && v6 < rat(6.0 * 16.0));
}

#[test]
fn embedded_sheet_is_cut_by_and_cuts_the_arrangement() {
    // A sheet poking through a box wall: both get subdivided consistently.
    let b = solid_box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let sheet = sheet_rect([1.0, -1.0, 0.5], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]);
    let mut tris: Vec<Tri> = b.tris.clone();
    tris.extend(sheet.tris.iter().copied());
    let arr = arrange(&tris);
    for (i, t) in tris.iter().enumerate() {
        check_invariants(t, &arr.facets[i], &arr.constraints[i]);
    }
    // The sheet crosses the walls y = 0 and y = 2: its facets must be split.
    let sheet_subs: usize = arr.facets[12..].iter().map(|f| f.triangles.len()).sum();
    assert!(sheet_subs > 2, "sheet must be subdivided, got {sheet_subs}");
    // Fully interior sheet stays intact and splits nothing.
    let inner = sheet_rect([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let mut tris2: Vec<Tri> = b.tris.clone();
    tris2.extend(inner.tris.iter().copied());
    let arr2 = arrange(&tris2);
    for f in &arr2.facets {
        assert_eq!(f.triangles.len(), 1, "nothing intersects: no subdivision");
    }
}

#[test]
fn sheet_polygon_with_hole_in_plane() {
    // Ground plane with an antipad hole, as a sheet.
    let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
    let hole = vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]];
    let s = sheet_polygon(
        &outer,
        &[hole],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    // Rational area via the 3D triangles (z = 1 everywhere).
    let area2: BigRational = s.tris.iter().fold(BigRational::zero(), |acc, t| {
        let term = (rat(t.v[1][0]) - rat(t.v[0][0])) * (rat(t.v[2][1]) - rat(t.v[0][1]))
            - (rat(t.v[1][1]) - rat(t.v[0][1])) * (rat(t.v[2][0]) - rat(t.v[0][0]));
        acc + term
    });
    assert_eq!(area2, rat(2.0 * 96.0));
    assert!(s.tris.iter().all(|t| t.v.iter().all(|v| v[2] == 1.0)));
}

#[test]
fn torus_watertight_and_volume_sane() {
    let f = rapidmesh_geom::torus([1.0, 2.0, 3.0], [0.0, 0.0, 2.0], 2.0, 0.5, 48, 24);
    let v6 = solid_volume6(&f);
    // Analytic 2 pi^2 R r^2 = 9.8696; the chordal tessellation underestimates.
    let v = v6.to_string();
    let _ = v;
    let approx = {
        let (verts, tris) = index_mesh(&f);
        assert_watertight(&tris);
        let _ = verts;
        // f64 estimate of the rational volume via the builder triangles.
        let mut acc = 0.0f64;
        for t in &f.tris {
            let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
            acc += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        acc / 6.0
    };
    let analytic = 2.0 * std::f64::consts::PI.powi(2) * 2.0 * 0.25;
    assert!(v6 > BigRational::zero());
    assert!(
        (approx - analytic).abs() / analytic < 0.02,
        "torus volume {approx} vs analytic {analytic}"
    );
}

#[test]
fn wedge_exact_volume() {
    // Trapezoid profile: ((dx + top_x) / 2) * dz, extruded dy.
    let f = rapidmesh_geom::wedge([1.0, 1.0, 1.0], 4.0, 2.0, 3.0, 1.0);
    // ((4 + 1) / 2 * 3) * 2 = 15; times 6 = 90.
    assert_eq!(solid_volume6(&f), rat(90.0));
    // Triangular prism at top_x = 0.
    let g = rapidmesh_geom::wedge([0.0, 0.0, 0.0], 4.0, 2.0, 3.0, 0.0);
    // (4 * 3 / 2) * 2 = 12; times 6 = 72.
    assert_eq!(solid_volume6(&g), rat(72.0));
}

#[test]
fn pipe_straight_exact_prism_volume() {
    // A straight pipe is an exact n-gon prism: V = n/2 r^2 sin(2 pi / n) * L.
    let f = rapidmesh_geom::pipe(&[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], 1.0, 8);
    let v6 = solid_volume6(&f);
    let analytic = 8.0 / 2.0 * (2.0 * std::f64::consts::PI / 8.0).sin() * 2.0;
    let approx: f64 = {
        let mut acc = 0.0f64;
        for t in &f.tris {
            let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
            acc += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        acc / 6.0
    };
    assert!(v6 > BigRational::zero());
    assert!((approx - analytic).abs() < 1e-9, "{approx} vs {analytic}");
}

#[test]
fn pipe_bent_watertight() {
    let path = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.5, 0.0],
        [3.0, 1.5, 0.5],
        [3.5, 2.5, 1.5],
    ];
    let f = rapidmesh_geom::pipe(&path, 0.2, 12);
    let v6 = solid_volume6(&f);
    assert!(v6 > BigRational::zero());
}

#[test]
fn helix_watertight() {
    let f = rapidmesh_geom::helix([0.0, 0.0, 0.0], 2.0, 1.0, 2.5, 0.15, 16, 8);
    let v6 = solid_volume6(&f);
    assert!(v6 > BigRational::zero());
    // Wire volume ~ pi r^2 * path length; sanity within 20%.
    let turns: f64 = 2.5;
    let circumference = (2.0 * std::f64::consts::PI * 2.0f64).hypot(1.0);
    let length = turns * circumference;
    let analytic = std::f64::consts::PI * 0.15f64.powi(2) * length;
    let approx: f64 = {
        let mut acc = 0.0f64;
        for t in &f.tris {
            let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
            acc += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        acc / 6.0
    };
    assert!(
        (approx - analytic).abs() / analytic < 0.2,
        "helix volume {approx} vs analytic {analytic}"
    );
}

#[test]
fn loft_frustum_exact_volume() {
    // Square 4x4 lofted to square 2x2 at height 3 (pyramidal frustum):
    // V = h/3 (A1 + A2 + sqrt(A1 A2)) = 1 * (16 + 4 + 8) = 28; times 6 = 168.
    let a = [
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0],
    ];
    let b = [
        [-1.0, -1.0, 3.0],
        [1.0, -1.0, 3.0],
        [1.0, 1.0, 3.0],
        [-1.0, 1.0, 3.0],
    ];
    let f = rapidmesh_geom::loft(&a, &b);
    assert_eq!(solid_volume6(&f), rat(168.0));
    // Orientation normalization: reversed input must give the same solid.
    let g = rapidmesh_geom::loft(&b, &a);
    assert_eq!(solid_volume6(&g), rat(168.0));
}

#[test]
fn mirrored_preserves_volume_and_orientation() {
    let f = rapidmesh_geom::wedge([1.0, 0.0, 0.0], 4.0, 2.0, 3.0, 1.0);
    let m = f.mirrored([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert_eq!(solid_volume6(&m), rat(90.0));
}

#[test]
fn scaled_volume_and_surface_degradation() {
    let f = cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 12);
    let v0 = solid_volume6(&f);
    // Uniform: volume x 8, cylinder kind keeps a scaled radius.
    let u = f.scaled([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
    assert_eq!(solid_volume6(&u), v0.clone() * rat(8.0));
    assert!(u.surfaces.iter().any(|s| matches!(
        s,
        rapidmesh_geom::SurfaceKind::Cylinder { radius, .. } if (*radius - 2.0).abs() < 1e-12
    )));
    // Non-uniform: volume x fx fy fz, curved kinds degrade to Plane.
    let n = f.scaled([2.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    assert_eq!(solid_volume6(&n), v0 * rat(2.0));
    assert!(n
        .surfaces
        .iter()
        .all(|s| matches!(s, rapidmesh_geom::SurfaceKind::Plane)));
    // Negative single factor flips orientation; winding is corrected.
    let r = f.scaled([-1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    assert!(solid_volume6(&r) > BigRational::zero());
}

// ---------------------------------------------------------- planar facets

// Conformal-tessellation representation (WP1b): every flat face is carried as
// a first-class boundary polygon plus a helper triangulation that must tile it
// exactly. These checks prove the two views agree (coplanar, equal area,
// disjoint helper ranges), which the conformal arrangement relies on.

use rapidmesh_exact::{orient3d, Axis};
use rapidmesh_geom::{pipe, sheet_disk, wedge, FlatFacet};

fn proj(p: [f64; 3], drop: Axis) -> [f64; 2] {
    match drop {
        Axis::X => [p[1], p[2]],
        Axis::Y => [p[2], p[0]],
        Axis::Z => [p[0], p[1]],
    }
}

/// Twice the signed shoelace area of a closed 3D loop in the given projection.
fn loop_area2(loop3: &[[f64; 3]], drop: Axis) -> BigRational {
    let n = loop3.len();
    let mut acc = BigRational::zero();
    for i in 0..n {
        let a = proj(loop3[i], drop);
        let b = proj(loop3[(i + 1) % n], drop);
        acc += rat(a[0]) * rat(b[1]) - rat(b[0]) * rat(a[1]);
    }
    acc
}

fn abs_rat(x: BigRational) -> BigRational {
    if x < BigRational::zero() {
        -x
    } else {
        x
    }
}

/// First exactly non-collinear triple of `loop3` (panics if fully collinear).
fn noncollinear_triple(loop3: &[[f64; 3]]) -> (Point3, Point3, Point3) {
    let n = loop3.len();
    for i in 0..n {
        let (a, b, c) = (
            Point3::Explicit(loop3[i]),
            Point3::Explicit(loop3[(i + 1) % n]),
            Point3::Explicit(loop3[(i + 2) % n]),
        );
        // Non-collinear iff some axis projection has nonzero orient2d, i.e. the
        // three are not on a line; detect via a nonzero scalar triple with a
        // synthetic offset point per axis is overkill -- use 3D area sign.
        for drop in [Axis::X, Axis::Y, Axis::Z] {
            let pa = proj(loop3[i], drop);
            let pb = proj(loop3[(i + 1) % n], drop);
            let pc = proj(loop3[(i + 2) % n], drop);
            let d = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
            if d != 0.0 {
                return (a, b, c);
            }
        }
    }
    panic!("flat facet outer loop is fully collinear");
}

/// Asserts every flat facet of `f` tiles its helper triangles by area, with
/// disjoint and valid ranges. When `exact_planar` is set, also asserts exact
/// coplanarity (only true for axis-constructed shapes; an f64 rotation rounds
/// the vertices fractionally off their plane while keeping the tiling identity,
/// which is combinatorial over the shared vertices and stays exact).
fn assert_flats_consistent(f: &Faceted, exact_planar: bool) {
    let mut covered = vec![false; f.tris.len()];
    for FlatFacet { facet, tris, .. } in &f.flats {
        assert!(tris.end <= f.tris.len() && tris.start < tris.end, "bad range");
        // Disjoint coverage.
        for i in tris.clone() {
            assert!(!covered[i], "helper triangle {i} claimed by two flats");
            covered[i] = true;
        }
        // Coplanarity: every helper-triangle vertex lies in the facet plane.
        if exact_planar {
            let (a, b, c) = noncollinear_triple(&facet.outer);
            for i in tris.clone() {
                for k in 0..3 {
                    let p = Point3::Explicit(f.tris[i].v[k]);
                    assert_eq!(
                        orient3d(&a, &b, &c, &p),
                        Some(rapidmesh_exact::Sign::Zero),
                        "helper vertex off the facet plane"
                    );
                }
            }
        }
        // Area: |outer| - sum|holes| == sum of helper-triangle areas, in the
        // facet's own projection (drop the dominant normal axis).
        let (axis, _) = facet.projection_axis();
        let mut poly = abs_rat(loop_area2(&facet.outer, axis));
        for h in &facet.holes {
            poly -= abs_rat(loop_area2(h, axis));
        }
        let mut tiled = BigRational::zero();
        for i in tris.clone() {
            let t = &f.tris[i];
            tiled += abs_rat(loop_area2(&t.v, axis));
        }
        assert_eq!(poly, tiled, "helper triangles do not tile the facet area");
    }
}

#[test]
fn box_flats_tile_six_faces() {
    let f = solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
    assert_eq!(f.flats.len(), 6, "box has six planar faces");
    assert_flats_consistent(&f, true);
    // All box triangles belong to a flat.
    let claimed: usize = f.flats.iter().map(|fl| fl.tris.len()).sum();
    assert_eq!(claimed, f.tris.len());
}

#[test]
fn frustum_caps_are_flats_barrel_is_not() {
    // At least the two caps; axis-aligned cone barrel quads may also be
    // exactly coplanar and get grouped (correct).
    let f = frustum([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 0.5, 16);
    assert!(f.flats.len() >= 2, "at least top and bottom caps");
    assert_flats_consistent(&f, true);
    // A cone (r_top = 0) has only the bottom cap; its apex barrel is triangles.
    let cone = frustum([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 0.0, 16);
    assert_eq!(cone.flats.len(), 1);
    assert_flats_consistent(&cone, true);
}

#[test]
fn cylinder_barrel_quads_are_flats() {
    // Each axis-aligned cylinder barrel quad is an exact planar rectangle, so
    // the barrel is segments flats plus the two caps.
    let c = cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 12);
    assert_eq!(c.flats.len(), 12 + 2, "12 barrel quads + 2 caps");
    assert_flats_consistent(&c, true);
}

#[test]
fn extrude_caps_and_walls_are_flats() {
    let outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
    let f = extrude_polygon(
        &outer,
        &[hole],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    // 2 caps + 4 outer walls + 4 hole walls.
    assert_eq!(f.flats.len(), 10);
    assert_flats_consistent(&f, true);
}

#[test]
fn lshape_extrude_cap_tiles_with_hole() {
    let f = extrude_polygon(
        &L_SHAPE,
        &[],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    assert_flats_consistent(&f, true);
}

#[test]
fn sheets_and_disk_are_flats() {
    let r = sheet_rect([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
    assert_eq!(r.flats.len(), 1);
    assert_flats_consistent(&r, true);
    let d = sheet_disk([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 12);
    assert_eq!(d.flats.len(), 1);
    assert_flats_consistent(&d, true);
    let p = sheet_polygon(
        &L_SHAPE,
        &[],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    assert_eq!(p.flats.len(), 1);
    assert_flats_consistent(&p, true);
}

#[test]
fn pipe_end_caps_are_flats() {
    let f = pipe(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], 0.3, 12);
    assert_eq!(f.flats.len(), 2, "two end caps; barrel stays curved");
    assert_flats_consistent(&f, true);
}

#[test]
fn wedge_flats_consistent_through_transform() {
    let f = wedge([0.0, 0.0, 0.0], 2.0, 1.0, 1.0, 0.5);
    assert_flats_consistent(&f, true);
    // Transforms keep the flat polygons tiling their helper triangles; an f64
    // rotation rounds points fractionally off-plane, so coplanarity is checked
    // exactly only on the axis-built shape.
    let t = f.rotated([0.0, 0.0, 0.0], [0.3, 0.4, 0.5], 0.7);
    assert_flats_consistent(&t, false);
    let m = f.mirrored([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert_flats_consistent(&m, true);
    let sc = f.scaled([-1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    assert_flats_consistent(&sc, true);
}
