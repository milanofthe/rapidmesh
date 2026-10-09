//! The order the input comes in does not reach the mesh: the same shapes
//! added in another order, a union's operands either way round and a
//! triangle soup with its triangles reordered mesh to the same tets.

use rapidmesh::shapes::{Cuboid, Cylinder, Sphere, Triangles};
use rapidmesh::{Geometry, MeshOptions};
use std::collections::BTreeSet;

/// The tets of a mesh by the places of their corners, each sorted.
fn tets(g: &Geometry) -> BTreeSet<[[u64; 3]; 4]> {
    let m = g.mesh(&MeshOptions::default()).unwrap();
    m.tets
        .iter()
        .map(|t| {
            let mut k = t.map(|i| m.points[i].map(f64::to_bits));
            k.sort_unstable();
            k
        })
        .collect()
}

#[test]
#[ignore = "the PLC and B-rep in an order of their own first (#365)"]
fn shapes_added_in_another_order_mesh_alike() {
    let build = |reverse: bool| {
        let mut g = Geometry::new(Some(0.5));
        let mut steps: Vec<Box<dyn Fn(&mut Geometry)>> = vec![
            Box::new(|g| {
                g.add(Cuboid::new([4.0, 3.0, 2.0])).unwrap();
            }),
            Box::new(|g| {
                g.add_solid(
                    Cylinder::new(0.5, 1.0).at([1.0, 1.5, 2.0]),
                    Some(0.3),
                    false,
                )
                .unwrap();
            }),
            Box::new(|g| {
                g.add_solid(Sphere::new(0.6).at([3.0, 1.5, -0.6]), Some(0.3), false)
                    .unwrap();
            }),
        ];
        if reverse {
            steps.reverse();
        }
        for s in &steps {
            s(&mut g);
        }
        g
    };
    assert_eq!(tets(&build(false)), tets(&build(true)));
}

#[test]
fn a_union_either_way_round_meshes_alike() {
    let build = |flip: bool| {
        let mut g = Geometry::new(Some(0.4));
        let a = g.add(Cuboid::new([2.0, 2.0, 2.0])).unwrap();
        let b = g.add(Cylinder::new(0.6, 3.0).at([1.0, 1.0, -0.5])).unwrap();
        g.union(&if flip { [b, a] } else { [a, b] }).unwrap();
        g
    };
    assert_eq!(tets(&build(false)), tets(&build(true)));
}

#[test]
#[ignore = "the PLC and B-rep in an order of their own first (#365)"]
fn a_soup_with_its_triangles_reordered_meshes_alike() {
    // An octahedron, its faces split once.
    let mut verts: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for [a, b, c] in faces {
        let m = verts.len() as u32;
        let (pa, pb, pc) = (verts[a], verts[b], verts[c]);
        verts.push(std::array::from_fn(|k| (pa[k] + pb[k] + pc[k]) / 3.0));
        let (a, b, c) = (a as u32, b as u32, c as u32);
        tris.extend([[a, b, m], [b, c, m], [c, a, m]]);
    }
    let build = |tris: Vec<[u32; 3]>| {
        let mut g = Geometry::new(Some(0.3));
        g.add(Triangles {
            verts: verts.clone(),
            tris,
        })
        .unwrap();
        g
    };
    let mut turned = tris.clone();
    turned.reverse();
    turned.rotate_left(5);
    assert_eq!(tets(&build(tris)), tets(&build(turned)));
}
