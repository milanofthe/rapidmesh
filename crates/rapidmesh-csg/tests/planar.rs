//! Conformal planar-facet arrangement: a flat face carried as a boundary
//! polygon plus a helper triangulation is cut conformally, with the helper's
//! interior structure merged away.

use rapidmesh_csg::{arrange, arrange_facets, PlanarFacet, PlanarInput, Tri};
use rapidmesh_exact::{Axis, Point3};

/// f64 area (x2) of a sub-triangle in the facet's projection.
fn tri_area2(v: &[Point3], t: [usize; 3], axis: Axis) -> f64 {
    let p = |i: usize| -> [f64; 2] {
        let c = v[i].approx().expect("valid");
        match axis {
            Axis::X => [c[1], c[2]],
            Axis::Y => [c[2], c[0]],
            Axis::Z => [c[0], c[1]],
        }
    };
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

fn total_area2(ft: &rapidmesh_csg::FacetTriangulation) -> f64 {
    ft.triangles
        .iter()
        .map(|&t| tri_area2(&ft.vertices, t, ft.axis))
        .sum()
}

fn square(z: f64) -> PlanarFacet {
    PlanarFacet::new(vec![
        [-1.0, -1.0, z],
        [1.0, -1.0, z],
        [1.0, 1.0, z],
        [-1.0, 1.0, z],
    ])
}

#[test]
fn single_facet_no_cuts_tiles_its_area() {
    // A square carried with a fan helper from an OFF-CENTRE interior point; the
    // convex seed must ignore that and tile the bare square (area 4).
    let p = [0.3, 0.0, 0.0];
    let c = square(0.0).outer;
    let helpers = vec![
        Tri::new(p, c[0], c[1]),
        Tri::new(p, c[1], c[2]),
        Tri::new(p, c[2], c[3]),
        Tri::new(p, c[3], c[0]),
    ];
    let arr = arrange_facets(&[PlanarInput {
        boundary: square(0.0),
        helpers,
    }]);
    assert_eq!(arr.facets.len(), 1);
    let ft = &arr.facets[0];
    assert!((total_area2(ft) - 8.0).abs() < 1e-12, "area {}", total_area2(ft)); // 2x area = 2*4
    // No interior helper vertex leaked into the result.
    for v in &ft.vertices {
        let q = v.approx().unwrap();
        assert!(
            (q[0] - 0.3).abs() > 1e-9 || q[1].abs() > 1e-9 || q[2].abs() > 1e-9,
            "off-centre fan apex leaked into the triangulation"
        );
    }
}

#[test]
fn holed_facet_tiles_outer_minus_hole() {
    // A 4x4 square with a 2x2 hole, helpers = an explicit valid triangulation
    // of the annulus. Area must be 16 - 4 = 12.
    let o = vec![
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
    ];
    let h = vec![
        [1.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [3.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
    ];
    // Annulus triangulation: 8 triangles bridging outer and hole rings.
    let mut helpers = Vec::new();
    for i in 0..4 {
        let j = (i + 1) % 4;
        helpers.push(Tri::new(o[i], o[j], h[j]));
        helpers.push(Tri::new(o[i], h[j], h[i]));
    }
    let arr = arrange_facets(&[PlanarInput {
        boundary: PlanarFacet::with_holes(o, vec![h]),
        helpers,
    }]);
    let ft = &arr.facets[0];
    assert!((total_area2(ft) - 24.0).abs() < 1e-12, "area {}", total_area2(ft)); // 2x area = 2*12
}

#[test]
fn fan_facet_pierced_conformally_no_sliver() {
    // A square (fan helper from an off-centre point) pierced by a vertical
    // wall at x = 0. The fan radials cross the cut line; the merge must fuse
    // the per-helper sub-segments into one clean constraint with endpoints at
    // (0, -1, 0) and (0, 1, 0) -- the true face crossings -- leaving no
    // near-twin slivers and no interior fan apex.
    let p = [0.3, 0.0, 0.0];
    let c = square(0.0).outer;
    let helpers = vec![
        Tri::new(p, c[0], c[1]),
        Tri::new(p, c[1], c[2]),
        Tri::new(p, c[2], c[3]),
        Tri::new(p, c[3], c[0]),
    ];
    let cap = PlanarInput {
        boundary: square(0.0),
        helpers,
    };
    // Wall: a big vertical triangle in the plane x = 0.
    let wall = PlanarInput::tri(Tri::new(
        [0.0, -2.0, -1.0],
        [0.0, 2.0, -1.0],
        [0.0, 0.0, 2.0],
    ));
    let arr = arrange_facets(&[cap, wall]);
    let cap_ft = &arr.facets[0];

    // Area conserved.
    assert!((total_area2(cap_ft) - 8.0).abs() < 1e-12); // 2x area = 2*4
    // Exactly the four corners plus the two true crossings -> 6 vertices.
    assert_eq!(cap_ft.vertices.len(), 6, "expected 4 corners + 2 crossings");
    // The merged constraint edge (0,-1,0)-(0,1,0) is present.
    let find = |target: [f64; 3]| -> usize {
        cap_ft
            .vertices
            .iter()
            .position(|v| {
                let q = v.approx().unwrap();
                (0..3).all(|k| (q[k] - target[k]).abs() < 1e-12)
            })
            .expect("crossing vertex present")
    };
    let a = find([0.0, -1.0, 0.0]);
    let b = find([0.0, 1.0, 0.0]);
    assert!(cap_ft.has_edge(a, b), "merged cut constraint must be an edge");
    // No sliver: every sub-triangle has a healthy area.
    for &t in &cap_ft.triangles {
        assert!(
            tri_area2(&cap_ft.vertices, t, cap_ft.axis) > 1e-6,
            "sliver sub-triangle survived the conformal cut"
        );
    }
}

#[test]
fn all_triangle_input_matches_triangle_soup() {
    // For an all-triangle scene the conformal path must reproduce the
    // triangle-soup arrangement (same sub-triangle count and area per facet):
    // single-helper facets make the merge a no-op.
    let tris = vec![
        // A unit square in z = 0 as two triangles.
        Tri::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]),
        Tri::new([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]),
        // A vertical triangle slicing through it at x = 1.
        Tri::new([1.0, -1.0, -1.0], [1.0, 3.0, -1.0], [1.0, 1.0, 2.0]),
    ];
    let soup = arrange(&tris);
    let facets: Vec<PlanarInput> = tris.iter().map(|t| PlanarInput::tri(*t)).collect();
    let conf = arrange_facets(&facets);
    assert_eq!(soup.facets.len(), conf.facets.len());
    for (a, b) in soup.facets.iter().zip(&conf.facets) {
        assert_eq!(
            a.triangles.len(),
            b.triangles.len(),
            "sub-triangle count diverged on a triangle facet"
        );
        assert!((total_area2(a) - total_area2(b)).abs() < 1e-12);
    }
}
