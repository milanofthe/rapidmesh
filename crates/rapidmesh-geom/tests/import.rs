//! STL/OBJ import: parsing, degenerate-facet dropping, closedness validation,
//! and meshing an imported solid end to end at the geom level (Scene
//! assembly).

use rapidmesh_geom::{import_obj, import_stl, solid_box, validate_closed, ImportError, Scene};
use std::io::Write as _;
use std::path::PathBuf;

fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("rapidmesh_import_{name}"));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(bytes).expect("write temp file");
    path
}

/// The unit tetrahedron as 4 outward-oriented triangles.
const TET_TRIS: [[[f64; 3]; 3]; 4] = [
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
];

fn ascii_stl(tris: &[[[f64; 3]; 3]]) -> Vec<u8> {
    let mut s = String::from("solid test\n");
    for t in tris {
        s.push_str("  facet normal 0 0 0\n    outer loop\n");
        for v in t {
            s.push_str(&format!("      vertex {} {} {}\n", v[0], v[1], v[2]));
        }
        s.push_str("    endloop\n  endfacet\n");
    }
    s.push_str("endsolid test\n");
    s.into_bytes()
}

fn binary_stl(tris: &[[[f64; 3]; 3]]) -> Vec<u8> {
    let mut b = vec![0u8; 80];
    b.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in tris {
        b.extend_from_slice(&[0u8; 12]); // facet normal, ignored
        for v in t {
            for &x in v {
                b.extend_from_slice(&(x as f32).to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 2]); // attribute byte count
    }
    b
}

#[test]
fn ascii_stl_roundtrip() {
    let path = temp_file("tet.stl", &ascii_stl(&TET_TRIS));
    let f = import_stl(&path).expect("import");
    assert_eq!(f.tris.len(), 4);
    validate_closed(&f).expect("closed");
}

#[test]
fn binary_stl_roundtrip() {
    let path = temp_file("tet_bin.stl", &binary_stl(&TET_TRIS));
    let f = import_stl(&path).expect("import");
    assert_eq!(f.tris.len(), 4);
    validate_closed(&f).expect("closed");
}

#[test]
fn binary_stl_with_solid_header_detected() {
    // A binary file whose 80-byte header happens to start with "solid".
    let mut b = binary_stl(&TET_TRIS);
    b[..5].copy_from_slice(b"solid");
    let path = temp_file("tet_trap.stl", &b);
    let f = import_stl(&path).expect("import");
    assert_eq!(f.tris.len(), 4);
}

#[test]
fn degenerate_facets_dropped() {
    let mut tris: Vec<[[f64; 3]; 3]> = TET_TRIS.to_vec();
    // Exactly collinear facet.
    tris.push([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
    let path = temp_file("tet_degen.stl", &ascii_stl(&tris));
    let f = import_stl(&path).expect("import");
    assert_eq!(f.tris.len(), 4);
}

#[test]
fn open_surface_rejected() {
    let path = temp_file("open.stl", &ascii_stl(&TET_TRIS[..3]));
    let f = import_stl(&path).expect("import");
    assert!(matches!(validate_closed(&f), Err(ImportError::NotClosed(_))));
}

#[test]
fn inconsistent_orientation_rejected() {
    let mut tris: Vec<[[f64; 3]; 3]> = TET_TRIS.to_vec();
    // Flip one facet: every edge still has 2 incidences but windings clash.
    tris[3] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let path = temp_file("flipped.stl", &ascii_stl(&tris));
    let f = import_stl(&path).expect("import");
    assert!(matches!(validate_closed(&f), Err(ImportError::NotClosed(_))));
}

#[test]
fn obj_roundtrip_with_quads_and_negative_indices() {
    // Unit cube as quads, last face via negative (relative) indices and
    // texture/normal suffixes.
    let obj = b"\
# cube
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f -8/1/1 -4/2/2 -1/3/3 -5/4/4
";
    let path = temp_file("cube.obj", obj);
    let f = import_obj(&path).expect("import");
    assert_eq!(f.tris.len(), 12);
    validate_closed(&f).expect("closed");
}

#[test]
fn imported_solid_assembles_as_scene() {
    // An imported tetrahedron inside a primitive box must survive Scene
    // assembly: 4 interface facets + 12 box facets, correct region tags.
    let path = temp_file("tet_scene.stl", &ascii_stl(&TET_TRIS));
    let f = import_stl(&path).expect("import");
    validate_closed(&f).expect("closed");
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]));
    let inner = scene.add_solid(f);
    let plc = scene.assemble();
    let n_inner = plc
        .region_tags
        .iter()
        .filter(|rt| rt.contains(&inner))
        .count();
    assert_eq!(n_inner, 4, "tetrahedron interface facets");
    assert_eq!(plc.triangles.len(), 16);
}
