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
