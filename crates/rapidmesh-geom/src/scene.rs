//! Scene assembly: solids (material regions) and embedded sheets into one
//! conforming tagged PLC.
//!
//! All facets are arranged together; every arrangement sub-triangle is
//! classified per side against every solid by its exact barycenter; region
//! priority (later-added solid wins) resolves overlaps. Solid sub-facets
//! survive iff the regions on their two sides differ; sheet sub-facets always
//! survive and carry their face tag. Coincident survivors (a sheet lying on a
//! material interface, or two solids sharing a face) are merged into one
//! facet with combined tags. Vertices are exact until the final snap to f64.

use crate::faceted::{Faceted, SurfaceKind};
use crate::plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc};
use rapidmesh_csg::{arrange, classify, Placement, Tri, VertexPool};
use rapidmesh_exact::Point3;
use std::collections::HashMap;

/// A scene of material solids and embedded sheets.
#[derive(Default)]
pub struct Scene {
    solids: Vec<Faceted>,
    sheets: Vec<(Faceted, FaceTag)>,
}

/// Source bookkeeping for one input facet.
struct Src {
    /// Index of the owning solid, `None` for sheet facets.
    solid: Option<usize>,
    /// Sheet face tag (0 for solid facets).
    tag: FaceTag,
    /// Global surface table index.
    surface: u32,
}

impl Scene {
    /// Empty scene.
    pub fn new() -> Scene {
        Scene::default()
    }

    /// Adds a closed, outward-oriented solid; returns its region tag.
    /// On overlap, the solid added later wins.
    pub fn add_solid(&mut self, f: Faceted) -> RegionTag {
        self.solids.push(f);
        RegionTag(self.solids.len() as u32)
    }

    /// Adds an embedded sheet with a face tag (use a nonzero tag).
    pub fn add_sheet(&mut self, f: Faceted, tag: FaceTag) {
        self.sheets.push((f, tag));
    }

    /// Assembles the conforming tagged PLC.
    pub fn assemble(&self) -> TaggedPlc {
        // ------------------------------------------------------- flatten
        let mut tris: Vec<Tri> = Vec::new();
        let mut src: Vec<Src> = Vec::new();
        let mut surfaces: Vec<SurfaceKind> = Vec::new();
        let mut flatten = |f: &Faceted, solid: Option<usize>, tag: FaceTag| {
            let base = surfaces.len() as u32;
            surfaces.extend(f.surfaces.iter().cloned());
            for (t, &fs) in f.tris.iter().zip(&f.face_surface) {
                tris.push(*t);
                src.push(Src {
                    solid,
                    tag,
                    surface: base + fs,
                });
            }
        };
        for (k, f) in self.solids.iter().enumerate() {
            flatten(f, Some(k), FaceTag(0));
        }
        for (f, tag) in &self.sheets {
            flatten(f, None, *tag);
        }

        let trace = std::env::var_os("RAPIDMESH_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let arr = arrange(&tris);
        if trace {
            eprintln!("assemble: arrange {:.1?}", t0.elapsed());
        }
        let t1 = std::time::Instant::now();

        // Scene bounding box for ray targets.
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for t in &tris {
            for v in &t.v {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }

        // --------------------------------------------- classify and keep
        let mut pool = VertexPool::default();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        let mut face_tags: Vec<FaceTag> = Vec::new();
        let mut surface_refs: Vec<SurfaceRef> = Vec::new();
        let mut region_tags: Vec<[RegionTag; 2]> = Vec::new();
        // Unordered vertex triple of already-emitted facets, for merging
        // coincident survivors.
        let mut emitted: HashMap<[u32; 3], usize> = HashMap::new();

        for (fi, ft) in arr.facets.iter().enumerate() {
            let s = &src[fi];
            for sub in &ft.triangles {
                let (p0, p1, p2) = (
                    &ft.vertices[sub[0]],
                    &ft.vertices[sub[1]],
                    &ft.vertices[sub[2]],
                );
                let bary = Point3::bary(p0.clone(), p1.clone(), p2.clone());

                // Per-side region resolution, highest-priority solid first.
                let mut front: Option<u32> = None;
                let mut back: Option<u32> = None;
                for j in (0..self.solids.len()).rev() {
                    if front.is_some() && back.is_some() {
                        break;
                    }
                    let region = j as u32 + 1;
                    if s.solid == Some(j) {
                        // Own boundary: interior behind the outward normal.
                        back.get_or_insert(region);
                        continue;
                    }
                    let (in_front, in_back) =
                        match classify(&bary, &tris[fi], &self.solids[j].tris, (lo, hi)) {
                            Placement::Inside => (true, true),
                            Placement::Outside => (false, false),
                            // Coincident facets: j's interior lies behind j's
                            // outward normal, i.e. behind ours iff the
                            // normals agree.
                            Placement::Boundary { same_normal } => (!same_normal, same_normal),
                        };
                    if in_front {
                        front.get_or_insert(region);
                    }
                    if in_back {
                        back.get_or_insert(region);
                    }
                }
                let fr = RegionTag(front.unwrap_or(0));
                let br = RegionTag(back.unwrap_or(0));

                // Solid facets survive only as region interfaces; sheets
                // always survive.
                if s.solid.is_some() && fr == br {
                    continue;
                }

                let idx: [u32; 3] = [
                    pool.insert(p0.clone()) as u32,
                    pool.insert(p1.clone()) as u32,
                    pool.insert(p2.clone()) as u32,
                ];
                let mut key = idx;
                key.sort_unstable();
                if let Some(&e) = emitted.get(&key) {
                    // Coincident facet already emitted: merge tags. Solid
                    // interfaces win the region pair (they are equal up to
                    // orientation anyway); sheets contribute their face tag.
                    face_tags[e] = face_tags[e].max(s.tag);
                    continue;
                }
                emitted.insert(key, triangles.len());
                triangles.push(idx);
                face_tags.push(s.tag);
                surface_refs.push(SurfaceRef(s.surface));
                region_tags.push([fr, br]);
            }
        }

        if trace {
            eprintln!("assemble: classify+emit {:.1?}", t1.elapsed());
        }
        // ------------------------------------------------- snap and emit
        // The PLC is pure f64 from here on. Exact arithmetic faithfully
        // preserves microscopic input asymmetries (e.g. cos and sin of the
        // same angle rounding differently make concentric-ring "radials"
        // not exactly collinear), so DISTINCT exact points can land on the
        // SAME f64 triple. Weld them, drop facets that collapse, and merge
        // facets that become coincident: zero-f64-area pieces carry no
        // region area, and duplicate indices would poison the mesher.
        let raw: Vec<[f64; 3]> = pool
            .verts
            .iter()
            .map(|p| p.approx().expect("valid point"))
            .collect();
        // Tolerance welding: exact constructions land within an ulp or two
        // of the input vertices they conceptually equal (cos and sin of the
        // same angle round differently, so concentric "radials" are not
        // exactly collinear and their crossings sit ~1e-16-relative off the
        // ring vertices). Sub-tolerance twins create twin crease chains the
        // Delaunay can never hold apart. Features below 1e-12 of the scene
        // diagonal are therefore welded, input vertices winning over
        // constructed points (inputs lie exactly on their planes).
        let diag = (0..3).map(|k| (bhi_w(&raw, k) - blo_w(&raw, k)).powi(2)).sum::<f64>().sqrt();
        fn blo_w(raw: &[[f64; 3]], k: usize) -> f64 {
            raw.iter().map(|q| q[k]).fold(f64::MAX, f64::min)
        }
        fn bhi_w(raw: &[[f64; 3]], k: usize) -> f64 {
            raw.iter().map(|q| q[k]).fold(f64::MIN, f64::max)
        }
        let tol = 1e-12 * diag.max(f64::MIN_POSITIVE);
        let cell = 2.0 * tol;
        let cell_of = |q: &[f64; 3]| -> [i64; 3] {
            std::array::from_fn(|k| (q[k] / cell).floor() as i64)
        };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(raw.len());
        let mut remap: Vec<u32> = vec![u32::MAX; raw.len()];
        let weld_pass = |explicit_only: bool,
                             grid: &mut HashMap<[i64; 3], Vec<u32>>,
                             vertices: &mut Vec<[f64; 3]>,
                             remap: &mut Vec<u32>| {
            for (i, q) in raw.iter().enumerate() {
                if remap[i] != u32::MAX {
                    continue;
                }
                if explicit_only && !matches!(pool.verts[i], Point3::Explicit(_)) {
                    continue;
                }
                let base = cell_of(q);
                let mut hit = None;
                'search: for dx in -1..=1i64 {
                    for dy in -1..=1i64 {
                        for dz in -1..=1i64 {
                            let key = [base[0] + dx, base[1] + dy, base[2] + dz];
                            if let Some(ids) = grid.get(&key) {
                                for &v in ids {
                                    let p = vertices[v as usize];
                                    let d2: f64 =
                                        (0..3).map(|k| (p[k] - q[k]).powi(2)).sum();
                                    if d2 <= tol * tol {
                                        hit = Some(v);
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }
                }
                remap[i] = hit.unwrap_or_else(|| {
                    let v = vertices.len() as u32;
                    vertices.push(*q);
                    grid.entry(base).or_default().push(v);
                    v
                });
            }
        };
        weld_pass(true, &mut grid, &mut vertices, &mut remap);
        weld_pass(false, &mut grid, &mut vertices, &mut remap);
        let mut out_triangles: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());
        let mut out_face_tags: Vec<FaceTag> = Vec::with_capacity(triangles.len());
        let mut out_surface_refs: Vec<SurfaceRef> = Vec::with_capacity(triangles.len());
        let mut out_region_tags: Vec<[RegionTag; 2]> = Vec::with_capacity(triangles.len());
        let mut emitted_snapped: HashMap<[u32; 3], usize> = HashMap::new();
        for (i, t) in triangles.iter().enumerate() {
            let m = t.map(|v| remap[v as usize]);
            if m[0] == m[1] || m[1] == m[2] || m[0] == m[2] {
                continue; // collapsed to an edge or point
            }
            let (a, b, c) = (
                vertices[m[0] as usize],
                vertices[m[1] as usize],
                vertices[m[2] as usize],
            );
            let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let v: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            if n.iter().all(|&x| x == 0.0) {
                continue; // exactly degenerate in f64
            }
            let mut key = m;
            key.sort_unstable();
            if let Some(&e) = emitted_snapped.get(&key) {
                // Coincident after welding: merge tags like the exact
                // coincident-survivor merge above.
                out_face_tags[e] = out_face_tags[e].max(face_tags[i]);
                continue;
            }
            emitted_snapped.insert(key, out_triangles.len());
            out_triangles.push(m);
            out_face_tags.push(face_tags[i]);
            out_surface_refs.push(surface_refs[i]);
            out_region_tags.push(region_tags[i]);
        }
        TaggedPlc {
            vertices,
            triangles: out_triangles,
            face_tags: out_face_tags,
            surface_refs: out_surface_refs,
            region_tags: out_region_tags,
            surfaces,
        }
    }
}
