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

        let arr = arrange(&tris);

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

        // ------------------------------------------------- snap and emit
        let vertices: Vec<[f64; 3]> = pool
            .verts
            .iter()
            .map(|p| p.approx().expect("valid point"))
            .collect();
        let plc = TaggedPlc {
            vertices,
            triangles,
            face_tags,
            surface_refs,
            region_tags,
            surfaces,
        };
        // Snapping must not have degenerated any facet.
        for t in &plc.triangles {
            let (a, b, c) = (
                plc.vertices[t[0] as usize],
                plc.vertices[t[1] as usize],
                plc.vertices[t[2] as usize],
            );
            let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let v: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            assert!(
                n.iter().any(|&x| x != 0.0),
                "facet degenerated by f64 snapping"
            );
        }
        plc
    }
}
