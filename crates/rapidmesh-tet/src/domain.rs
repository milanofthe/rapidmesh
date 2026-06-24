//! The central spatial structure of the mesher: one static octree over the
//! domain, refined to the local sizing field h(x). Everything queries it:
//!
//!   * `h_at(p)`        — the sizing field (per-leaf, cached).
//!   * `seed_points()`  — one graded volume seed per interior leaf (small leaves
//!                        near the boundary, large in the bulk -> graded density).
//!   * `region_at(p)`   — the material region (cached per leaf; exact ray-cast
//!                        only on boundary leaves), which accelerates tet
//!                        classification from O(tets * faces) to ~O(tets).
//!   * `neighbors(p, r)`— sites within `r`, from the per-leaf site buckets
//!                        (re-bucketed each Lloyd pass); the Lloyd neighbor
//!                        search runs on this same tree.
//!
//! h(x) = min( region cap, surface_h + grading * dist-to-boundary, point
//! sources ); Lipschitz by the grading term, so it grows smoothly from the fine
//! boundary into the coarse interior.

use crate::conform::MeshParams;
use crate::facetbvh::FacetBvh;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::Point3;
use rapidmesh_geom::{SurfaceKind, TaggedPlc};

type V3 = [f64; 3];

use crate::constants::DOMAIN_MAX_DEPTH as MAX_DEPTH;
/// Grading falloff cap distance is implicit in `grading`; the boundary target.
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

enum Node {
    Leaf(Leaf),
    Inner(Box<[Node; 8]>),
}

struct Leaf {
    h: f64,
    region: u32,
    /// True when no boundary facet can pass through this leaf (the center is
    /// farther from the boundary than the leaf circumradius), so the whole leaf
    /// is one region and the cached `region` is trustworthy without a ray-cast.
    uniform: bool,
    sites: Vec<u32>,
}

const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Per-region closed boundary (the PLC facets touching that region) for the
/// exact inside test.
struct RegionSoup {
    region: u32,
    tris: Vec<Tri>,
    boxes: TriBoxes,
    /// BVH over `tris` for the y,z-column prefilter of the parity ray-cast, so a
    /// boundary-leaf classification tests O(column) facets, not all of them.
    bvh: FacetBvh,
}

pub struct DomainTree {
    lo: V3,
    hi: V3,
    root: Node,
    regions: Vec<RegionSoup>,
    bbox: ([f64; 3], [f64; 3]),
    /// Finest target edge length `s0` (the global `finest_cap`): the base BCC
    /// spacing for graded seeding.
    finest: f64,
    /// Coarsest target (the bulk `maxh`): caps the grading levels.
    maxh: f64,
}

fn child_box(center: V3, half: f64, oct: usize) -> (V3, f64) {
    let h = 0.5 * half;
    let c = std::array::from_fn(|k| center[k] + if oct & (1 << k) != 0 { h } else { -h });
    (c, h)
}

impl DomainTree {
    /// Builds the domain octree from the PLC and mesh parameters. `facet_surf`
    /// is an optional per-PLC-facet surface size target (the resolved per-FACE
    /// `surf_maxh`, mapped through the brep); empty means "no per-face override".
    /// It feeds the VOLUME sizing field so a finely sized face refines the volume
    /// behind it -- without it the fine surface is collapsed back by optimize,
    /// because the volume target never knew the face was meant to be fine.
    pub fn build(plc: &TaggedPlc, params: &MeshParams, facet_surf: &[f64]) -> DomainTree {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for p in &plc.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let diag = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max);
        let center: V3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let half = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001;
        let bbox = (lo, hi);

        // Per-region closed boundary soups (facets where the region appears).
        let mut region_ids: Vec<u32> = Vec::new();
        for rt in &plc.region_tags {
            for r in rt {
                if r.0 != 0 && !region_ids.contains(&r.0) {
                    region_ids.push(r.0);
                }
            }
        }
        region_ids.sort_unstable();
        let pad = 1e-6 * diag.max(1.0);
        let regions: Vec<RegionSoup> = region_ids
            .iter()
            .map(|&r| {
                let tris: Vec<Tri> = plc
                    .triangles
                    .iter()
                    .zip(&plc.region_tags)
                    .filter(|(_, rt)| rt[0].0 == r || rt[1].0 == r)
                    .map(|(t, _)| {
                        Tri::new(
                            plc.vertices[t[0] as usize],
                            plc.vertices[t[1] as usize],
                            plc.vertices[t[2] as usize],
                        )
                    })
                    .collect();
                let boxes = TriBoxes::build(&tris, pad);
                let facets: Vec<(Tri, f64)> = tris.iter().map(|&t| (t, 0.0)).collect();
                let bvh = FacetBvh::build(&facets);
                RegionSoup { region: r, tris, boxes, bvh }
            })
            .collect();

        // Sizing parameters.
        let maxh = if params.maxh.is_finite() && params.maxh > 0.0 {
            params.maxh
        } else {
            diag / 8.0
        };
        let grading = if params.grading > 0.0 { params.grading } else { 0.5 };

        let region_of = |p: V3| -> u32 {
            for rs in &regions {
                if point_inside_solid(&Point3::Explicit(p), p, &rs.tris, &rs.boxes, bbox) {
                    return rs.region;
                }
            }
            0
        };
        let region_cap = |r: u32| -> f64 {
            if r == 0 {
                return maxh;
            }
            params
                .region_maxh
                .iter()
                .find(|(rr, _)| *rr == r)
                .map(|&(_, h)| h)
                .unwrap_or(maxh)
        };

        // The SURFACE drives the interior grading. Each boundary facet carries a
        // target edge length `h_target`, the finest of: its face tag's
        // `face_maxh`, its owning solid's `surface_maxh`, the caps of its
        // adjacent regions, else the bulk `maxh`. The sizing field then grows
        // from these wall targets into the interior (Lipschitz with `grading`),
        // so a finely meshed face refines the volume behind it and coarsens away.
        let facet_centroid = |i: usize| -> V3 {
            let t = plc.triangles[i];
            std::array::from_fn(|k| {
                (plc.vertices[t[0] as usize][k]
                    + plc.vertices[t[1] as usize][k]
                    + plc.vertices[t[2] as usize][k])
                    / 3.0
            })
        };

        // Curvature/volume-error target of a curved facet: a facet edge `h` on a
        // surface of principal radius `R` deviates by sagitta ~ h^2/(8R), so
        // bounding the relative sagitta gives `h_curv = R * sqrt(8 * frac)`. This
        // refines the VOLUME near tightly curved boundaries (an airfoil nose), so
        // the surrounding region holds the fine on-surface nodes; the grading
        // term then coarsens away. A gentle curve (R large) leaves `maxh` intact.
        let chord = (8.0 * params.tol_surf).sqrt();
        let curvature_target = |i: usize| -> f64 {
            let kind = &plc.surfaces[plc.surface_refs[i].0 as usize];
            crate::project::surface_curvature_radius(kind, facet_centroid(i)) * chord
        };

        let facet_target = |i: usize| -> f64 {
            let ft = plc.face_tags[i].0;
            let base = if let Some(&(_, h)) = params.face_maxh.iter().find(|(t, _)| *t == ft) {
                h.min(maxh)
            } else {
                let owner = plc.surface_owners[plc.surface_refs[i].0 as usize];
                if let Some(&(_, h)) = params.surface_maxh.iter().find(|(o, _)| *o == owner) {
                    h.min(maxh)
                } else {
                    let mut h = maxh;
                    for r in plc.region_tags[i] {
                        if r.0 != 0 {
                            h = h.min(region_cap(r.0));
                        }
                    }
                    h
                }
            };
            // Per-FACE override (resolved `surf_maxh`, finest wins), then curvature.
            base.min(facet_surf.get(i).copied().unwrap_or(f64::INFINITY))
                .min(curvature_target(i))
        };
        let facets: Vec<(Tri, f64)> = plc
            .triangles
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let tri = Tri::new(
                    plc.vertices[t[0] as usize],
                    plc.vertices[t[1] as usize],
                    plc.vertices[t[2] as usize],
                );
                (tri, facet_target(i))
            })
            .collect();

        // The finest volume target anywhere: the base BCC spacing `s0`. The
        // surface is oversampled finer than this; the BCC only has to resolve the
        // finest VOLUME size so it stays ~2x coarser than the surface and the
        // surface tiling dominates the restricted Delaunay (conformity).
        let mut s0 = facets.iter().map(|&(_, h)| h).fold(f64::MAX, f64::min);
        for &(_, sh) in &params.size_points {
            s0 = s0.min(sh);
        }
        let spacing = if s0.is_finite() && s0 > 0.0 { s0 } else { diag / 8.0 };
        let min_half = (0.5 * spacing).max(1e-9 * diag.max(1.0));

        // Facet BVH: O(log F) nearest-facet distance and graded-min, replacing
        // the O(F) brute scans that dominated the build on high-facet meshes.
        let bvh = FacetBvh::build(&facets);

        // ---- feature sizing framework (WP-R1) -------------------------------
        // The sizing field is the MIN over graded FEATURE sources, each a
        // Lipschitz lower bound `target + grading * dist`:
        //   - faces:  per-facet targets above (caps + sagitta curvature), `bvh`.
        //   - curved EDGES: sagitta targets on the analytic edge curve, sampled
        //     as segments (a degenerate tri = a segment, so `FacetBvh` gives the
        //     point-to-segment graded distance). Filled by WP-R3; empty here.
        //   - point sources: `size_points`.
        // Adding a feature kind is just another graded source in `h_of`.
        let edge_segments: Vec<(Tri, f64)> = edge_sizing_segments(plc, params.tol_edge, maxh);
        let edge_bvh = FacetBvh::build(&edge_segments);

        // Nearest-facet distance (for the uniform-leaf region cache).
        let dist_to_boundary = |p: V3| -> f64 { bvh.nearest_dist(p) };
        let h_of = |p: V3, region: u32| -> f64 {
            let mut h = region_cap(region)
                .min(maxh)
                .min(bvh.graded_min(p, grading))
                .min(edge_bvh.graded_min(p, grading));
            for (sp, sh) in &params.size_points {
                h = h.min(sh + grading * dist(p, *sp));
            }
            h
        };

        let root = build_node(center, half, 0, &region_of, &dist_to_boundary, &h_of, min_half);
        DomainTree { lo, hi, root, regions, bbox, finest: spacing, maxh }
    }

    pub fn lo(&self) -> V3 {
        self.lo
    }
    pub fn hi(&self) -> V3 {
        self.hi
    }
    /// Finest target edge length `s0` anywhere in the domain (the base spacing).
    pub fn finest(&self) -> f64 {
        self.finest
    }

    fn leaf_at(&self, p: V3) -> Option<(&Leaf,)> {
        let mut node = &self.root;
        let (mut c, mut h) = (
            std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k])),
            (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001,
        );
        loop {
            match node {
                Node::Leaf(l) => return Some((l,)),
                Node::Inner(ch) => {
                    let mut oct = 0;
                    for k in 0..3 {
                        if p[k] >= c[k] {
                            oct |= 1 << k;
                        }
                    }
                    let (cc, hh) = child_box(c, h, oct);
                    c = cc;
                    h = hh;
                    node = &ch[oct];
                }
            }
        }
    }

    /// Sizing field at `p`.
    pub fn h_at(&self, p: V3) -> f64 {
        self.leaf_at(p).map(|(l,)| l.h).unwrap_or(f64::INFINITY)
    }

    /// Material region at `p`: the cached leaf region when the leaf is wholly
    /// interior to one region (no boundary passes through it), else an exact
    /// per-region ray-cast.
    pub fn region_at(&self, p: V3) -> u32 {
        // Outside the domain bounding box is unconditionally exterior.
        for k in 0..3 {
            if p[k] < self.lo[k] || p[k] > self.hi[k] {
                return 0;
            }
        }
        // Exact parity ray-cast, but only against the facets in the y,z-column
        // the +x ray can cross (BVH prefilter): O(column) per region, not O(F).
        // The margin covers `point_inside_solid`'s y,z jitter band (diag * 0.02),
        // so excluded facets are never crossed and the parity stays exact.
        let exact = || {
            let (lo, hi) = self.bbox;
            let diag = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
            let margin = 0.021 * diag;
            let pad = 1e-6 * diag;
            let mut cand: Vec<u32> = Vec::new();
            for rs in &self.regions {
                rs.bvh.column_yz(p, margin, &mut cand);
                let sub: Vec<Tri> = cand.iter().map(|&i| rs.tris[i as usize]).collect();
                let sub_boxes = TriBoxes::build(&sub, pad);
                if point_inside_solid(&Point3::Explicit(p), p, &sub, &sub_boxes, self.bbox) {
                    return rs.region;
                }
            }
            0
        };
        match self.leaf_at(p) {
            Some((l,)) if l.uniform => l.region,
            Some(_) => exact(),
            None => 0,
        }
    }

    pub fn inside(&self, p: V3) -> bool {
        self.region_at(p) != 0
    }

    /// Graded BCC volume seeds, aligned to the domain origin `lo` at the base
    /// spacing `s0 = finest`.
    ///
    /// A BCC lattice is the set of grid points (in half-spacing units `u = s0/2`)
    /// whose three indices share a parity: all-even is the corner sublattice,
    /// all-odd is the body-centered sublattice. Its Delaunay dual is the well
    /// conditioned BCC tetrahedralization, and it tiles boundaries cleanly.
    ///
    /// To grade it, each node is assigned a level `L = floor(log2(h(p)/s0))` from
    /// the octree sizing field and kept only if it is a node of the level-`L` BCC
    /// (spacing `s0 * 2^L`): corner-`L` nodes have all indices `≡ 0 (mod 2^{L+1})`,
    /// body-`L` nodes all `≡ 2^L (mod 2^{L+1})`. Where `h = s0` (L = 0) every BCC
    /// node survives (the full fine lattice, identical to a uniform BCC, so flat
    /// and faceted interfaces tile exactly); coarse regions decimate to a proper
    /// coarse BCC. The result is origin-aligned, so it matches the surface
    /// sampling grid. The caller filters to the interior and clear of the surface.
    pub fn seed_points(&self) -> Vec<V3> {
        let s0 = self.finest;
        if !(s0.is_finite() && s0 > 0.0) {
            return Vec::new();
        }
        let u = 0.5 * s0; // half-spacing grid unit
        let lmax = (if self.maxh > s0 { (self.maxh / s0).log2().ceil() as i64 + 1 } else { 0 })
            .clamp(0, 30);
        // Walk the octree and emit, per leaf, only the level-L BCC nodes inside
        // it (its `h` sets L). Each leaf spans ~one level-L cell, so this is
        // O(leaves) = O(output) instead of iterating the whole finest grid over
        // the bbox (which exploded when a fine size source made `s0` tiny). A
        // node on a leaf boundary is emitted by both leaves, so dedup on the
        // integer half-unit index.
        let center: V3 = std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k]));
        let half = (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001;
        let mut seen: std::collections::HashSet<[i64; 3]> = std::collections::HashSet::new();
        let mut out = Vec::new();
        collect_leaf_bcc(&self.root, center, half, self.lo, u, s0, lmax, &mut seen, &mut out);
        out
    }

    /// Clears and re-fills the per-leaf site buckets from `positions`.
    pub fn rebucket(&mut self, positions: &[V3]) {
        clear_sites(&mut self.root);
        let center: V3 = std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k]));
        let half = (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001;
        for (i, &p) in positions.iter().enumerate() {
            insert_site(&mut self.root, center, half, p, i as u32);
        }
    }

    /// Site indices within Euclidean distance `r` of `p`.
    pub fn neighbors(&self, p: V3, r: f64) -> Vec<u32> {
        let mut out = Vec::new();
        let center: V3 = std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k]));
        let half = (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001;
        gather(&self.root, center, half, p, r * r, &mut out);
        out
    }
}

/// Circumradius of the triangle `(a, b, c)` -- the radius of the circle through
/// the three points, i.e. the osculating-circle radius of the polyline at `b`.
/// `INFINITY` for collinear points (a straight edge has no curvature).
fn circumradius(a: V3, b: V3, c: V3) -> f64 {
    let ab = dist(a, b);
    let bc = dist(b, c);
    let ca = dist(c, a);
    // Twice the triangle area from the cross product of two edges.
    let u = sub(b, a);
    let v = sub(c, a);
    let cr = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let area2 = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
    if area2 <= 1e-300 {
        f64::INFINITY
    } else {
        ab * bc * ca / (2.0 * area2)
    }
}

/// Sagitta-bounded sizing targets along curved feature edges (WP-R3), derived
/// purely from geometry. A feature edge is a PLC edge whose two adjacent facets
/// belong to DIFFERENT analytic surfaces, at least one of them curved (e.g. the
/// rim where a cylinder barrel meets a flat cap, or the circle where a plane cuts
/// a sphere). Such edges are space curves whose own curvature can be TIGHTER than
/// either bounding surface (a small circle near a sphere's pole), so the per-facet
/// surface-curvature target under-resolves them. We recover the edge-curve radius
/// `R_edge` directly as the circumradius of three consecutive edge vertices (three
/// points on a circle give its exact radius, even from a coarse facet polyline)
/// and emit segment targets `h = R_edge * sqrt(8*delta)` (a segment = a degenerate
/// tri, so `FacetBvh` yields the graded point-to-segment distance). Where the edge
/// curves no tighter than its surface (a cylinder rim), `R_edge` matches the face
/// target and the MIN composition makes this a harmless no-op.
fn edge_sizing_segments(plc: &TaggedPlc, deflection: f64, maxh: f64) -> Vec<(Tri, f64)> {
    use rustc_hash::FxHashMap;
    let chord = (8.0 * deflection).sqrt();
    let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    let is_curved = |sid: u32| !matches!(plc.surfaces[sid as usize], SurfaceKind::Plane);

    // Distinct analytic surfaces meeting along each undirected edge.
    let mut edge_surf: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    for (fi, t) in plc.triangles.iter().enumerate() {
        let s = plc.surface_refs[fi].0;
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let v = edge_surf.entry(key(a, b)).or_default();
            if !v.contains(&s) {
                v.push(s);
            }
        }
    }
    // Feature edges: two distinct surfaces meet, at least one curved. Sorted so
    // the downstream segment list (and its BVH) is order-deterministic.
    let mut feature: Vec<(u32, u32)> = edge_surf
        .iter()
        .filter(|(_, s)| s.len() >= 2 && s.iter().any(|&x| is_curved(x)))
        .map(|(&e, _)| e)
        .collect();
    feature.sort_unstable();

    // Plane geometry recovered from the PLC (the `SurfaceKind::Plane` itself
    // carries none): origin + unit normal of the first facet on each planar
    // surface. Needed so POCS can project onto a plane-cut edge, not only the
    // curved side.
    let mut plane_geom: FxHashMap<u32, (V3, V3)> = FxHashMap::default();
    for (fi, t) in plc.triangles.iter().enumerate() {
        let s = plc.surface_refs[fi].0;
        if matches!(plc.surfaces[s as usize], SurfaceKind::Plane) {
            plane_geom.entry(s).or_insert_with(|| {
                let (v0, v1, v2) =
                    (plc.vertices[t[0] as usize], plc.vertices[t[1] as usize], plc.vertices[t[2] as usize]);
                let (e1, e2) = (sub(v1, v0), sub(v2, v0));
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let nl = dot(n, n).sqrt();
                (v0, if nl > 1e-12 { [n[0] / nl, n[1] / nl, n[2] / nl] } else { [0.0, 0.0, 1.0] })
            });
        }
    }
    // Edge-curve neighbours of each feature vertex (its polyline link), and ALL
    // analytic surfaces meeting along the curve at each vertex (both sides).
    let mut nbr: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut vert_surfs: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &(a, b) in &feature {
        nbr.entry(a).or_default().push(b);
        nbr.entry(b).or_default().push(a);
        if let Some(ss) = edge_surf.get(&key(a, b)) {
            for &v in &[a, b] {
                let e = vert_surfs.entry(v).or_default();
                for &s in ss {
                    if !e.contains(&s) {
                        e.push(s);
                    }
                }
            }
        }
    }
    // Project a point onto the intersection of the analytic surfaces meeting at
    // the edge by alternating projection (POCS) onto BOTH sides -- a plane via its
    // recovered geometry, a curved surface via its oracle. This pulls the faceted
    // chain onto the true curve, so the osculating radius below reflects the REAL
    // curvature of the intersection (INFINITY for a plane-cut generator, the true
    // radius for a genuine curve) -- not the spurious tiny radius a faceted polyline
    // zigzag shows (the over-refinement that fanned out the borders).
    let pocs = |p: V3, sids: &[u32]| -> V3 {
        let mut q = p;
        for _ in 0..8 {
            for &s in sids {
                q = match plane_geom.get(&s) {
                    Some(&(o, n)) => crate::project::closest_on_plane(q, o, n),
                    None => crate::project::closest_on_surface(&plc.surfaces[s as usize], q),
                };
            }
        }
        q
    };
    // Osculating radius at a vertex (only where it has exactly two neighbours -- a
    // smooth interior point; junctions/endpoints stay INFINITY). The radius is
    // sampled with a CONTROLLED step `eps` along the curve tangent, each sample
    // POCS-projected onto the curve, NOT from the raw faceted neighbours: the
    // arrangement places intersection vertices at irregular spacing (often two
    // almost coincident), whose 3-point circumradius is a spurious tiny value even
    // on a straight edge. The fixed-baseline analytic estimate gives the TRUE
    // curvature -- INFINITY on a straight intersection (e.g. plane-cut along a
    // cylinder generator), the real radius on a genuinely curved one.
    let eps = (0.35 * maxh).max(1e-9);
    // Walk the polyline link away from `v` (first step toward `first`) until the
    // accumulated arc length reaches `eps`, returning that on-curve vertex. A
    // controlled baseline of REAL curve points -- robust to the arrangement's
    // irregular vertex spacing AND, unlike a tangent-step + POCS, it does not
    // collapse on a curved-curved intersection (where projecting a stepped point
    // snaps back near `v`).
    let walk = |start: u32, first: u32| -> u32 {
        let (mut prev, mut cur) = (start, first);
        let mut acc = dist(plc.vertices[start as usize], plc.vertices[first as usize]);
        while acc < eps {
            let next = match nbr.get(&cur) {
                Some(ns) if ns.len() == 2 => {
                    if ns[0] == prev { ns[1] } else { ns[0] }
                }
                _ => break, // junction / open end
            };
            acc += dist(plc.vertices[cur as usize], plc.vertices[next as usize]);
            prev = cur;
            cur = next;
        }
        cur
    };
    let vert_radius = |v: u32| -> f64 {
        match (nbr.get(&v), vert_surfs.get(&v)) {
            (Some(ns), Some(sids)) if ns.len() == 2 && !sids.is_empty() => {
                let a = walk(v, ns[0]);
                let b = walk(v, ns[1]);
                circumradius(
                    pocs(plc.vertices[a as usize], sids),
                    pocs(plc.vertices[v as usize], sids),
                    pocs(plc.vertices[b as usize], sids),
                )
            }
            _ => f64::INFINITY,
        }
    };

    let mut out: Vec<(Tri, f64)> = Vec::new();
    for &(a, b) in &feature {
        let r = vert_radius(a).min(vert_radius(b));
        if r.is_finite() {
            let va = plc.vertices[a as usize];
            let vb = plc.vertices[b as usize];
            // A degenerate tri (va, vb, va) is the segment va-vb for the BVH.
            out.push((Tri::new(va, vb, va), r * chord));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    center: V3,
    half: f64,
    depth: u32,
    region_of: &dyn Fn(V3) -> u32,
    dist_of: &dyn Fn(V3) -> f64,
    h_of: &dyn Fn(V3, u32) -> f64,
    min_half: f64,
) -> Node {
    let region = region_of(center);
    let d = dist_of(center);
    let h = h_of(center, region);
    // Subdivide while the cell is bigger than its target size (and not too deep
    // / too small). 2*half is the cell side.
    if 2.0 * half > h && depth < MAX_DEPTH && half > min_half {
        let children: [Node; 8] = std::array::from_fn(|oct| {
            let (cc, hh) = child_box(center, half, oct);
            build_node(cc, hh, depth + 1, region_of, dist_of, h_of, min_half)
        });
        Node::Inner(Box::new(children))
    } else {
        // No boundary facet reaches into the leaf if the center is farther from
        // the boundary than the leaf circumradius (half * sqrt(3)).
        let uniform = d > half * SQRT3;
        Node::Leaf(Leaf { h, region, uniform, sites: Vec::new() })
    }
}

/// Emits the level-L BCC nodes inside each leaf (L from the leaf's `h`), the
/// graded-seed generator. `lo` is the lattice origin, `u = s0/2` the half-unit;
/// `seen` dedups nodes shared on leaf boundaries by their integer index.
#[allow(clippy::too_many_arguments)]
fn collect_leaf_bcc(
    node: &Node,
    center: V3,
    half: f64,
    lo: V3,
    u: f64,
    s0: f64,
    lmax: i64,
    seen: &mut std::collections::HashSet<[i64; 3]>,
    out: &mut Vec<V3>,
) {
    match node {
        Node::Leaf(l) => {
            let lvl =
                (if l.h > s0 { (l.h / s0).log2().floor() as i64 } else { 0 }).clamp(0, lmax);
            let m = 1i64 << (lvl + 1); // index period of the level-L lattice
            let want = 1i64 << lvl; // body-sublattice offset (corner offset 0)
            // Half-open index range [a_lo, a_hi) covering the leaf box per axis.
            let idx_lo: [i64; 3] =
                std::array::from_fn(|k| ((center[k] - half - lo[k]) / u).ceil() as i64);
            let idx_hi: [i64; 3] =
                std::array::from_fn(|k| ((center[k] + half - lo[k]) / u).ceil() as i64);
            // First index >= lo with `idx % m == off`.
            let first = |a_lo: i64, off: i64| a_lo + (off - a_lo).rem_euclid(m);
            for off in [0, want] {
                let mut a = first(idx_lo[0], off);
                while a < idx_hi[0] {
                    let mut b = first(idx_lo[1], off);
                    while b < idx_hi[1] {
                        let mut c = first(idx_lo[2], off);
                        while c < idx_hi[2] {
                            if seen.insert([a, b, c]) {
                                out.push([
                                    lo[0] + a as f64 * u,
                                    lo[1] + b as f64 * u,
                                    lo[2] + c as f64 * u,
                                ]);
                            }
                            c += m;
                        }
                        b += m;
                    }
                    a += m;
                }
            }
        }
        Node::Inner(ch) => {
            for (oct, c) in ch.iter().enumerate() {
                let (cc, hh) = child_box(center, half, oct);
                collect_leaf_bcc(c, cc, hh, lo, u, s0, lmax, seen, out);
            }
        }
    }
}

fn clear_sites(node: &mut Node) {
    match node {
        Node::Leaf(l) => l.sites.clear(),
        Node::Inner(ch) => {
            for c in ch.iter_mut() {
                clear_sites(c);
            }
        }
    }
}

fn insert_site(node: &mut Node, center: V3, half: f64, p: V3, idx: u32) {
    match node {
        Node::Leaf(l) => l.sites.push(idx),
        Node::Inner(ch) => {
            let mut oct = 0;
            for k in 0..3 {
                if p[k] >= center[k] {
                    oct |= 1 << k;
                }
            }
            let (cc, hh) = child_box(center, half, oct);
            insert_site(&mut ch[oct], cc, hh, p, idx);
        }
    }
}

fn box_dist2(center: V3, half: f64, p: V3) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let e = (p[k] - center[k]).abs() - half;
        if e > 0.0 {
            d2 += e * e;
        }
    }
    d2
}

fn gather(node: &Node, center: V3, half: f64, p: V3, r2: f64, out: &mut Vec<u32>) {
    if box_dist2(center, half, p) > r2 {
        return;
    }
    match node {
        Node::Leaf(l) => out.extend_from_slice(&l.sites),
        Node::Inner(ch) => {
            for (oct, c) in ch.iter().enumerate() {
                let (cc, hh) = child_box(center, half, oct);
                gather(c, cc, hh, p, r2, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    fn cube_plc(s: f64) -> TaggedPlc {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [s, s, s]));
        scene.assemble()
    }

    #[test]
    fn grades_finer_near_size_point() {
        // A plain maxh box is uniform (correct: no feature drives a finer size);
        // grading appears around a refinement source. Put a size point at the
        // center and check h grows with distance from it (Lipschitz in grading).
        let plc = cube_plc(4.0);
        let t = DomainTree::build(
            &plc,
            &MeshParams {
                maxh: 2.0,
                grading: 0.5,
                size_points: vec![([2.0, 2.0, 2.0], 0.1)],
                ..Default::default()
            },
            &[],
        );
        let h_at_point = t.h_at([2.0, 2.0, 2.0]);
        let h_away = t.h_at([2.0, 2.0, 3.5]);
        assert!(h_at_point < h_away, "at point {h_at_point} finer than away {h_away}");
        assert!(h_at_point <= 0.3, "near the size point ~0.1, got {h_at_point}");
    }

    #[test]
    fn region_inside_outside() {
        let plc = cube_plc(4.0);
        let t = DomainTree::build(&plc, &MeshParams { maxh: 0.8, ..Default::default() }, &[]);
        assert_ne!(t.region_at([2.0, 2.0, 2.0]), 0, "center inside");
        assert_eq!(t.region_at([-1.0, 2.0, 2.0]), 0, "outside");
        assert!(t.inside([2.0, 2.0, 2.0]) && !t.inside([5.0, 2.0, 2.0]));
    }

    #[test]
    fn seeds_form_a_bcc_lattice() {
        // seed_points returns origin-aligned BCC candidates over the bbox; the
        // caller filters to the interior. A uniform box yields the full fine BCC:
        // both sublattices present and many candidates land inside.
        let plc = cube_plc(4.0);
        let t = DomainTree::build(&plc, &MeshParams { maxh: 0.5, grading: 0.5, ..Default::default() }, &[]);
        let seeds = t.seed_points();
        let inside: Vec<_> = seeds.iter().filter(|s| t.inside(**s)).collect();
        assert!(inside.len() > 50, "expected a populated interior lattice, got {}", inside.len());
        // s0 = 0.5, u = 0.25: corner nodes at even (i*0.5), body at odd (i*0.5+0.25).
        let has_corner = inside.iter().any(|s| (s[0] / 0.5).fract().abs() < 1e-9);
        let has_body = inside.iter().any(|s| ((s[0] - 0.25) / 0.5).fract().abs() < 1e-9);
        assert!(has_corner && has_body, "both BCC sublattices present");
    }

    #[test]
    fn seeds_grade_by_region() {
        // A coarse box (maxh) with a finer interior cube (region_maxh) seeds the
        // fine region denser than the coarse one.
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [8.0, 8.0, 8.0]));
        let inner = scene.add_solid(solid_box([3.0, 3.0, 3.0], [5.0, 5.0, 5.0]));
        let plc = scene.assemble();
        let t = DomainTree::build(
            &plc,
            &MeshParams { maxh: 4.0, region_maxh: vec![(inner.0, 1.0)], grading: 0.5, ..Default::default() },
            &[],
        );
        // h is finer inside the small cube than out in the bulk.
        assert!(t.h_at([4.0, 4.0, 4.0]) < t.h_at([0.5, 0.5, 0.5]));
    }

    #[test]
    fn neighbors_finds_nearby_sites() {
        let plc = cube_plc(4.0);
        let mut t = DomainTree::build(&plc, &MeshParams { maxh: 1.0, ..Default::default() }, &[]);
        let pts = vec![[2.0, 2.0, 2.0], [2.3, 2.0, 2.0], [3.9, 3.9, 3.9]];
        t.rebucket(&pts);
        let near = t.neighbors([2.0, 2.0, 2.0], 0.5);
        assert!(near.contains(&0) && near.contains(&1) && !near.contains(&2));
    }
}
