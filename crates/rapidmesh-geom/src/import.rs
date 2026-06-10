//! Surface mesh import: STL (binary and ASCII) and Wavefront OBJ.
//!
//! Imported shapes become [`Faceted`] solids whose facets ARE the exact
//! geometry: every triangle is its own [`SurfaceKind::Plane`], so the mesher
//! reproduces the input surface exactly (PLC semantics, no approximation).
//! Exactly degenerate (collinear) facets are dropped on import; duplicated
//! facets are rejected. [`validate_closed`] checks the watertight,
//! consistently-oriented 2-manifold invariant that [`crate::Scene`] solids
//! require.

use crate::faceted::{Faceted, SurfaceKind};
use rapidmesh_csg::Tri;
use rapidmesh_exact::collinear;
use std::collections::HashMap;
use std::io::Read as _;
use std::path::Path;

/// Import failure.
#[derive(Debug)]
pub enum ImportError {
    /// I/O failure.
    Io(std::io::Error),
    /// Malformed file content (message describes the location).
    Parse(String),
    /// Structural defect found by [`validate_closed`].
    NotClosed(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io(e) => write!(f, "io error: {e}"),
            ImportError::Parse(m) => write!(f, "parse error: {m}"),
            ImportError::NotClosed(m) => write!(f, "surface not closed: {m}"),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<std::io::Error> for ImportError {
    fn from(e: std::io::Error) -> ImportError {
        ImportError::Io(e)
    }
}

/// Builds a [`Faceted`] from raw triangles: drops exactly degenerate facets,
/// gives every surviving facet its own plane surface.
fn faceted_from_tris(tris: Vec<Tri>) -> Faceted {
    let mut f = Faceted::new();
    for t in tris {
        if collinear(&t.point(0), &t.point(1), &t.point(2)) == Some(true) {
            continue;
        }
        let s = f.add_surface(SurfaceKind::Plane);
        f.push_tri(t, s);
    }
    f
}

// ----------------------------------------------------------------- STL

/// Reads an STL file (binary or ASCII, auto-detected) into a [`Faceted`].
/// Facet normals in the file are ignored; orientation comes from the vertex
/// winding (the STL convention requires both to agree).
pub fn import_stl(path: &Path) -> Result<Faceted, ImportError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let tris = if stl_is_binary(&bytes) {
        parse_stl_binary(&bytes)?
    } else {
        parse_stl_ascii(&bytes)?
    };
    Ok(faceted_from_tris(tris))
}

/// Binary detection: the 80-byte header is free-form (may even start with
/// "solid"), so the reliable test is the binary length invariant
/// `84 + 50 * n_triangles`.
fn stl_is_binary(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    bytes.len() == 84 + 50 * n
}

fn parse_stl_binary(bytes: &[u8]) -> Result<Vec<Tri>, ImportError> {
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let mut tris = Vec::with_capacity(n);
    let f32_at = |off: usize| -> f64 {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) as f64
    };
    for i in 0..n {
        let base = 84 + 50 * i;
        // 12 bytes facet normal (skipped), then 3 vertices of 12 bytes.
        let v: [[f64; 3]; 3] = std::array::from_fn(|j| {
            std::array::from_fn(|k| f32_at(base + 12 + 12 * j + 4 * k))
        });
        if v.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ImportError::Parse(format!("non-finite vertex in facet {i}")));
        }
        tris.push(Tri::new(v[0], v[1], v[2]));
    }
    Ok(tris)
}

fn parse_stl_ascii(bytes: &[u8]) -> Result<Vec<Tri>, ImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| ImportError::Parse("ascii stl is not valid utf-8".to_string()))?;
    let mut tris = Vec::new();
    let mut verts: Vec<[f64; 3]> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("vertex") => {
                let mut v = [0.0f64; 3];
                for x in &mut v {
                    *x = it
                        .next()
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| {
                            ImportError::Parse(format!("bad vertex on line {}", ln + 1))
                        })?;
                }
                verts.push(v);
            }
            Some("endloop") => {
                if verts.len() != 3 {
                    return Err(ImportError::Parse(format!(
                        "facet with {} vertices on line {}",
                        verts.len(),
                        ln + 1
                    )));
                }
                tris.push(Tri::new(verts[0], verts[1], verts[2]));
                verts.clear();
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err(ImportError::Parse("no facets found".to_string()));
    }
    Ok(tris)
}

// ----------------------------------------------------------------- OBJ

/// Reads a Wavefront OBJ file into a [`Faceted`]. Only `v` and `f` records
/// are interpreted; faces with more than three corners are fan-triangulated;
/// `f` indices may be 1-based or negative (relative), with optional
/// `/texture/normal` suffixes.
pub fn import_obj(path: &Path) -> Result<Faceted, ImportError> {
    let text = std::fs::read_to_string(path)?;
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<Tri> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let mut v = [0.0f64; 3];
                for x in &mut v {
                    *x = it
                        .next()
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| {
                            ImportError::Parse(format!("bad vertex on line {}", ln + 1))
                        })?;
                }
                verts.push(v);
            }
            Some("f") => {
                let mut idx: Vec<usize> = Vec::new();
                for tok in it {
                    let first = tok.split('/').next().unwrap_or("");
                    let i: i64 = first.parse().map_err(|_| {
                        ImportError::Parse(format!("bad face index on line {}", ln + 1))
                    })?;
                    let resolved = if i > 0 {
                        i as usize - 1
                    } else if i < 0 {
                        let r = verts.len() as i64 + i;
                        if r < 0 {
                            return Err(ImportError::Parse(format!(
                                "face index out of range on line {}",
                                ln + 1
                            )));
                        }
                        r as usize
                    } else {
                        return Err(ImportError::Parse(format!(
                            "face index 0 on line {}",
                            ln + 1
                        )));
                    };
                    if resolved >= verts.len() {
                        return Err(ImportError::Parse(format!(
                            "face index out of range on line {}",
                            ln + 1
                        )));
                    }
                    idx.push(resolved);
                }
                if idx.len() < 3 {
                    return Err(ImportError::Parse(format!(
                        "face with {} corners on line {}",
                        idx.len(),
                        ln + 1
                    )));
                }
                for j in 1..idx.len() - 1 {
                    tris.push(Tri::new(verts[idx[0]], verts[idx[j]], verts[idx[j + 1]]));
                }
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err(ImportError::Parse("no faces found".to_string()));
    }
    Ok(faceted_from_tris(tris))
}

// ---------------------------------------------------------- validation

/// Checks the closed-solid invariant [`crate::Scene::add_solid`] requires:
/// after welding bit-identical vertices, every undirected edge must be shared
/// by exactly two facets with opposite directions (watertight, consistently
/// oriented 2-manifold), and no facet may appear twice.
pub fn validate_closed(f: &Faceted) -> Result<(), ImportError> {
    let mut vid: HashMap<[u64; 3], u32> = HashMap::new();
    let mut key = |p: [f64; 3]| -> u32 {
        let bits: [u64; 3] = std::array::from_fn(|k| {
            // Weld +0.0 and -0.0; all other coordinates by exact bits.
            let x = if p[k] == 0.0 { 0.0 } else { p[k] };
            x.to_bits()
        });
        let next = vid.len() as u32;
        *vid.entry(bits).or_insert(next)
    };
    // Per undirected edge: net winding count (+1 forward, -1 backward) and
    // total incidence count.
    let mut edges: HashMap<(u32, u32), (i64, u64)> = HashMap::new();
    let mut seen_facets: HashMap<[u32; 3], usize> = HashMap::new();
    for (fi, t) in f.tris.iter().enumerate() {
        let v: [u32; 3] = std::array::from_fn(|i| key(t.v[i]));
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            return Err(ImportError::NotClosed(format!(
                "facet {fi} has repeated vertices after welding"
            )));
        }
        let mut sorted = v;
        sorted.sort_unstable();
        if let Some(&prev) = seen_facets.get(&sorted) {
            return Err(ImportError::NotClosed(format!(
                "facet {fi} duplicates facet {prev}"
            )));
        }
        seen_facets.insert(sorted, fi);
        for e in 0..3 {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
            entry.0 += if a < b { 1 } else { -1 };
            entry.1 += 1;
        }
    }
    for (&(a, b), &(net, count)) in &edges {
        if count != 2 || net != 0 {
            return Err(ImportError::NotClosed(format!(
                "edge ({a}, {b}) has {count} incident facets (net winding {net}), expected 2 with opposite orientation"
            )));
        }
    }
    Ok(())
}
