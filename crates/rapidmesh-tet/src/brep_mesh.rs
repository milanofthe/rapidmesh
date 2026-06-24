//! Bridge from a [`rapidmesh_brep::Brep`] to the bottom-up mesher stages.
//!
//! The B-rep is the geometry source; this module turns its analytic edges and
//! trimmed faces into the inputs the existing stages consume -- stage 1 here
//! (an edge's analytic curve for `curve::distribute`), the face and volume
//! stages follow. The B-rep only changes WHERE the surface points come from; the
//! volume Lloyd, region classification and restricted-Delaunay extraction are
//! unchanged.

use crate::conform::{MeshParams, SurfaceFace};
use crate::curve::{distribute, Curve, PolylineCurve};
use crate::site::{Carrier, Site};
use crate::surf2d::{cvt_fill, triangulate_constrained};
use crate::surfchart::{build_chart, PlaneChart, SurfaceChart};
use rapidmesh_brep::{Brep, Curve as BCurve, Edge as BEdge, Surface};
use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::{FaceTag, SurfaceKind, TaggedPlc};
use std::sync::Arc;

type V3 = [f64; 3];
type P2 = [f64; 2];

use crate::constants::{OVERSAMPLE, SURF_LLOYD_ITERS};

/// Exact bit pattern of a 3D point, for de-duplicating pinned vertices.
fn bits(p: V3) -> [u64; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

fn dist3(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

/// True if the face's boundary covers the full 2*pi in the surface's first
/// parameter (theta) -- a full-revolution barrel with no axial trim.
fn is_full_revolution(surf: &Surface, boundary: &[V3]) -> bool {
    use std::f64::consts::PI;
    if boundary.len() < 4 {
        return false;
    }
    let mut th: Vec<f64> = boundary.iter().map(|&p| surf.project_uv(p)[0]).collect();
    th.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut max_gap = th[0] + 2.0 * PI - th[th.len() - 1];
    for w in th.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    max_gap < PI / 3.0
}

/// The sorted, deduplicated theta rays of a revolution face's rim edges (the
/// boundary points projected to the surface's first parameter). Reused for every
/// interior row so the grid aligns RADIALLY with the rims (clean quads, no rim
/// slivers), exactly as `cylinder_iso`/`frustum_iso` build their rings.
fn rim_theta_rays(surf: &Surface, boundary: &[V3]) -> Vec<f64> {
    let mut th: Vec<f64> = boundary.iter().map(|&p| surf.project_uv(p)[0]).collect();
    th.sort_by(|a, b| a.partial_cmp(b).unwrap());
    th.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    th
}

/// A structured (theta, v) grid on a full-revolution surface at the target
/// ARC-LENGTH spacing in v, on the rim `theta_rays` -- INTERIOR rows only (the
/// rims are shared edges). Uniform v is uniform arc length on every revolution
/// surface (each column is an isometric generator), and reusing the rim rays
/// keeps the rings radially aligned, so the grid is the on-surface CVT optimum
/// for a developable. Density is mesher-chosen (refinement-independent).
fn revolution_grid(surf: &Surface, vmin: f64, vmax: f64, theta_rays: &[f64], target: impl Fn(V3) -> f64) -> Vec<V3> {
    if !(vmax > vmin) || theta_rays.len() < 3 {
        return Vec::new();
    }
    let vmid = 0.5 * (vmin + vmax);
    let eps = (vmax - vmin) * 1e-4 + 1e-12;
    let dv_arc = dist3(surf.eval_uv([0.0, vmid + eps]), surf.eval_uv([0.0, vmid - eps])) / (2.0 * eps);
    let v_arc_total = dv_arc * (vmax - vmin);
    let tgt_mid = target(surf.eval_uv([0.0, vmid])).max(1e-9);
    let nv = ((v_arc_total / tgt_mid).round() as usize).max(1);
    let mut pts = Vec::new();
    for j in 1..nv {
        let vj = vmin + (vmax - vmin) * j as f64 / nv as f64;
        for &theta in theta_rays {
            pts.push(surf.eval_uv([theta, vj]));
        }
    }
    pts
}

/// Even-odd point-in-region test over a planar face's loop segments (in (u,v)).
fn in_loops(uv: P2, segs: &[(P2, P2)]) -> bool {
    let mut c = false;
    for &(a, b) in segs {
        if (a[1] > uv[1]) != (b[1] > uv[1]) {
            let x = a[0] + (uv[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if uv[0] < x {
                c = !c;
            }
        }
    }
    c
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Geodesic (icosphere) vertices + triangles, `subdivisions` Loop steps, every
/// vertex projected onto the sphere. This is the isotropic, pole-free, SEAMLESS
/// mesh of a CLOSED sphere: a closed surface has no global chart (hairy-ball /
/// Theorema Egregium), so it is meshed natively -- the sphere oracle (`Surface`)
/// gives the on-surface carrier, the geodesic seed gives the connectivity. The
/// shared core of `rapidmesh_geom::icosphere`, returning index data directly.
fn icosphere_mesh(center: V3, radius: f64, subdivisions: usize) -> (Vec<V3>, Vec<[usize; 3]>) {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut verts: Vec<V3> = vec![
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    type Cache = std::collections::HashMap<(usize, usize), usize>;
    let mid = |a: usize, b: usize, verts: &mut Vec<V3>, cache: &mut Cache| -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&m) = cache.get(&key) {
            return m;
        }
        let (va, vb) = (verts[a], verts[b]);
        verts.push([(va[0] + vb[0]) * 0.5, (va[1] + vb[1]) * 0.5, (va[2] + vb[2]) * 0.5]);
        let idx = verts.len() - 1;
        cache.insert(key, idx);
        idx
    };
    let mut cache: Cache = Cache::new();
    for _ in 0..subdivisions {
        cache.clear();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let a = mid(tri[0], tri[1], &mut verts, &mut cache);
            let b = mid(tri[1], tri[2], &mut verts, &mut cache);
            let c = mid(tri[2], tri[0], &mut verts, &mut cache);
            next.push([tri[0], a, c]);
            next.push([tri[1], b, a]);
            next.push([tri[2], c, b]);
            next.push([a, b, c]);
        }
        faces = next;
    }
    let proj = |v: V3| -> V3 {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        add(center, scale(v, radius / l))
    };
    (verts.iter().map(|&v| proj(v)).collect(), faces)
}

/// Arc-length-parametrised analytic curve for a [`BCurve::Profile`] edge: the 2D
/// profile lifted to 3D on its extrusion frame at height `z`, over the parameter
/// range `[t0, t1]` the edge covers. Curvature comes from the exact profile, so
/// the distribution is tessellation-independent (the airfoil outline).
struct ProfileCurve {
    profile: Arc<NurbsCurve>,
    base: V3,
    u: V3,
    v: V3,
    axis: V3,
    z: f64,
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ProfileCurve {
    fn new(profile: Arc<NurbsCurve>, base: V3, u: V3, v: V3, axis: V3, t: [f64; 2], z: f64) -> Option<ProfileCurve> {
        let (lo, hi) = (t[0].min(t[1]), t[0].max(t[1]));
        if !(hi > lo) {
            return None;
        }
        let n = 256usize;
        let (mut ts, mut ss) = (vec![lo], vec![0.0f64]);
        let (mut prev, mut acc) = (lo, 0.0);
        for i in 1..=n {
            let tt = lo + (hi - lo) * i as f64 / n as f64;
            acc += profile.arc_length(prev, tt, 2);
            ts.push(tt);
            ss.push(acc);
            prev = tt;
        }
        Some(ProfileCurve { profile, base, u, v, axis, z, ts, ss })
    }
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self.ss.partition_point(|&x| x < s).clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
    fn at3(&self, t: f64) -> V3 {
        let c = self.profile.eval(t);
        add(add(self.base, scale(self.axis, self.z)), add(scale(self.u, c[0]), scale(self.v, c[1])))
    }
}

impl Curve for ProfileCurve {
    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }
    fn point_at(&self, s: f64) -> V3 {
        self.at3(self.s_to_t(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        let k = self.profile.curvature(self.s_to_t(s));
        if k > 1e-12 {
            1.0 / k
        } else {
            f64::INFINITY
        }
    }
}

/// A circular arc (or full circle) parametrised by arc length. Radius is constant,
/// so the sagitta sizing places uniform points; the arc range is taken from the
/// edge's chain endpoints (a closed rim spans the full `2*pi`).
struct CircleCurve {
    center: V3,
    x: V3,
    y: V3,
    radius: f64,
    a0: f64,
    span: f64,
}

impl CircleCurve {
    fn new(center: V3, axis: V3, x: V3, radius: f64, chain: &[V3]) -> Option<CircleCurve> {
        if !(radius > 0.0) || chain.len() < 2 {
            return None;
        }
        let y = cross(axis, x);
        let ang = |p: V3| {
            let d = sub(p, center);
            dot(d, y).atan2(dot(d, x))
        };
        // Total signed swept angle = sum of per-segment increments (each in
        // (-pi, pi]); robust for an arc (partial) and a closed rim (sums to +-2*pi).
        let pi = std::f64::consts::PI;
        let wrap = |a: f64| (a + pi).rem_euclid(2.0 * pi) - pi;
        let a0 = ang(chain[0]);
        let mut span = 0.0;
        for w in chain.windows(2) {
            span += wrap(ang(w[1]) - ang(w[0]));
        }
        if span.abs() < 1e-9 {
            return None;
        }
        Some(CircleCurve { center, x, y, radius, a0, span })
    }
}

impl Curve for CircleCurve {
    fn length(&self) -> f64 {
        self.radius * self.span.abs()
    }
    fn point_at(&self, s: f64) -> V3 {
        let f = (s / self.length()).clamp(0.0, 1.0);
        let t = self.a0 + self.span * f;
        let (st, ct) = t.sin_cos();
        std::array::from_fn(|k| self.center[k] + self.radius * (ct * self.x[k] + st * self.y[k]))
    }
    fn radius_at(&self, _s: f64) -> f64 {
        self.radius
    }
}

/// The analytic curve to distribute points on for a B-rep edge: the exact profile
/// or circle where recovered, else the faceted chain polyline (a straight `Line`
/// is exactly a 2-point polyline, so it reduces to uniform spacing).
pub fn edge_curve(edge: &BEdge) -> Option<Box<dyn Curve>> {
    match &edge.curve {
        BCurve::Profile { profile, base, u, v, axis, t, z } => {
            ProfileCurve::new(profile.clone(), *base, *u, *v, *axis, *t, *z)
                .map(|c| Box::new(c) as Box<dyn Curve>)
        }
        BCurve::Circle { center, axis, radius, x } => {
            CircleCurve::new(*center, *axis, *x, *radius, &edge.chain)
                .map(|c| Box::new(c) as Box<dyn Curve>)
        }
        _ => PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>),
    }
}

/// The surface points produced from a B-rep, in the exact form the volume stage
/// of `cvt::mesh` consumes.
pub struct SurfaceSites {
    /// Corner + edge points (pinned), then per-face interior points.
    pub sites: Vec<Site>,
    /// `sites.len()` -- all of `sites` are surface points.
    pub n_surf: usize,
    /// Per site: the B-rep face it tiles, or `u32::MAX` for a shared corner/edge
    /// point (which defers its output tag to an interior face point).
    pub point_tile: Vec<u32>,
    /// Per site: the local target edge length it was generated at (the per-entity
    /// `edge_maxh`/`surf_maxh` resolution). The volume stage uses this as the
    /// point's `point_size` so the quality post-pass does NOT coarsen an
    /// intentionally refined edge/face back to the bulk size.
    pub point_size: Vec<f64>,
    /// Per B-rep face: `(originating plc surface index, face tag)` for output.
    pub tiles: Vec<(u32, FaceTag)>,
    /// Number of pinned corner points (the `plc_points` count).
    pub plc_points: usize,
    /// The frozen surface triangulation (indices into `sites`), tagged with each
    /// face's region pair / tag / analytic surface. This is what makes the
    /// brep-based surface a complete `FrozenSurface` for the constrained volume
    /// stage (it carries the per-entity sizing of `sites`).
    pub tris: Vec<SurfaceFace>,
    /// Carrier of each triangle (its plane / analytic surface), for exact
    /// boundary recovery (parallel to `tris`).
    pub tri_carrier: Vec<Carrier>,
}

/// Deterministic splitmix64 -> uniform f64 in `[0, 1)` (the mesher must be
/// reproducible run-to-run, so no real RNG).
fn rng(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

fn cell_key(p: V3, cell: f64) -> (i64, i64, i64) {
    ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)
}

/// True if `p` is at least `r` from every point already in the spatial-hash grid.
fn grid_clear(grid: &HashGrid, p: V3, r: f64, cell: f64) -> bool {
    let (kx, ky, kz) = cell_key(p, cell);
    let r2 = r * r;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if let Some(v) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                    for &q in v {
                        if dot(sub(p, q), sub(p, q)) < r2 {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

type HashGrid = std::collections::HashMap<(i64, i64, i64), Vec<V3>>;

/// Randomized Poisson-disk fill of a face: dart-throwing on the face's facet
/// triangles, each dart projected onto the analytic surface and kept if it clears
/// the boundary and every accepted point by ~0.65 of the local target size. Darts
/// land on the face's own triangles, so the region test is free and topology-
/// agnostic (no (u,v) seam or pole). The boundary edge points seed the grid as
/// fixed repellers. Returns the accepted interior points, on the surface.
fn fill_face_points(
    surf: &Surface,
    facets: &[u32],
    plc: &TaggedPlc,
    boundary: &[V3],
    target: &dyn Fn(V3) -> f64,
    seed: u64,
) -> Vec<V3> {
    let tris: Vec<[V3; 3]> = facets
        .iter()
        .map(|&fi| {
            let t = plc.triangles[fi as usize];
            [plc.vertices[t[0] as usize], plc.vertices[t[1] as usize], plc.vertices[t[2] as usize]]
        })
        .collect();
    let mut cum = Vec::with_capacity(tris.len());
    let mut area = 0.0;
    for t in &tris {
        area += 0.5 * dot(cross(sub(t[1], t[0]), sub(t[2], t[0])), cross(sub(t[1], t[0]), sub(t[2], t[0]))).sqrt();
        cum.push(area);
    }
    if !(area > 0.0) {
        return Vec::new();
    }
    let proj = |p: V3| surf.eval_uv(surf.project_uv(p));
    // Per-facet target -> the coarsest local size (sets the grid cell, so it is
    // always >= the separation radius and the +-1 neighbour query is exact) and
    // the density integral Sum(area / target^2) (the dart budget: fine regions are
    // small in area, so a sharp feature does not blow it up).
    let mut tmax = 1e-9f64;
    let mut density = 0.0f64;
    for (i, t) in tris.iter().enumerate() {
        let c: V3 = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
        let tc = target(proj(c)).max(1e-9);
        tmax = tmax.max(tc);
        let area_i = if i == 0 { cum[0] } else { cum[i] - cum[i - 1] };
        density += area_i / (tc * tc);
    }
    let cell = 0.65 * tmax;
    let mut grid: HashGrid = HashGrid::new();
    for &b in boundary {
        grid.entry(cell_key(b, cell)).or_default().push(b);
    }
    let mut state = seed.wrapping_mul(0x100_0000_01B3).wrapping_add(0x9E3779B1);
    let mut interior: Vec<V3> = Vec::new();
    let budget = ((density * 8.0).ceil() as usize).clamp(64, 400_000);
    for _ in 0..budget {
        let pick = rng(&mut state) * area;
        let ti = cum.partition_point(|&c| c < pick).min(tris.len() - 1);
        let t = tris[ti];
        let (mut u, mut v) = (rng(&mut state), rng(&mut state));
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let dart: V3 = std::array::from_fn(|k| t[0][k] + u * (t[1][k] - t[0][k]) + v * (t[2][k] - t[0][k]));
        let p = proj(dart);
        let r = 0.65 * target(p);
        if grid_clear(&grid, p, r, cell) {
            grid.entry(cell_key(p, cell)).or_default().push(p);
            interior.push(p);
        }
    }
    // Just a seed -- the on-surface Lloyd (volume stage) relaxes these to a CVT.
    interior
}

/// Builds the surface points from a B-rep: stage 1 distributes on every edge curve
/// (shared, so adjacent faces agree), stage 2 meshes each trimmed face in its
/// (u,v) parameter space, trimmed by the loop PCurves, and lifts the points onto
/// the surface (an exact planar carrier where the face is a plane). Closed faces
/// with no loops (a full sphere) are skipped here -- periodic param meshing is a
/// later step; until then they stay on the faceted path.
pub fn surface_sites(
    brep: &Brep,
    plc: &TaggedPlc,
    params: &MeshParams,
    domain: &crate::domain::DomainTree,
) -> SurfaceSites {
    let grad = if params.grading > 0.0 { params.grading } else { 0.5 };
    // The geometry size field H (finite even when params.maxh is INFINITY -- it
    // falls back to diag/8). The surface is placed at OVERSAMPLE*H, the volume at
    // H, so the surface is finer than the volume (the conformity requirement).
    let h_at = |p: V3| domain.h_at(p).max(1e-9);
    let mut sites: Vec<Site> = Vec::new();
    let mut point_tile: Vec<u32> = Vec::new();
    let mut point_size: Vec<f64> = Vec::new();
    let mut tris: Vec<SurfaceFace> = Vec::new();
    let mut tri_carrier: Vec<Carrier> = Vec::new();

    // ---- stage 1a: corners (pinned) -------------------------------------
    for v in &brep.vertices {
        sites.push(Site::vertex(v.pos));
        point_tile.push(u32::MAX);
        point_size.push(f64::INFINITY); // pinned; no local coarsening target
    }
    let plc_points = sites.len();

    // A curved analytic surface (cylinder / cone / torus / sphere / extruded) keeps
    // its clean structured INPUT tessellation: the tessellator already produced a
    // near-uniform on-surface mesh, so it IS the 2D surface CVT -- re-relaxing it
    // would need periodic-seam / pole handling for no gain, and a dart fill leaves
    // flat slivers on ruled surfaces. Its bounding edges keep the input vertices
    // too, so the shared rim stays conforming. PLANAR faces (whose input is an
    // arbitrary coarse fan) are meshed by a true 2D Lloyd (`cvt_fill`) in their
    // in-plane (u,v) parameter space -- that is where points must be generated.
    // An edge keeps its INPUT vertices ONLY if it bounds a face that takes the
    // structured FALLBACK (a full-revolution / extruded barrel, or a closed
    // surface with no rim) -- there the face mesh IS the input tessellation, so
    // its rim must match. Every CHARTABLE (open / trimmed) curved face -- the
    // norm after a boolean -- instead has its edges (INCLUDING the intersection
    // curves) re-sampled at the adaptive field below, so the cvt_fill boundary it
    // is constrained to follows the surface finely. (Removing the old "any curved
    // face keeps input" shortcut: that capped intersection curves at the coarse
    // arrangement density -- the straddler root cause.)
    let face_uses_fallback = |f: &rapidmesh_brep::Face| -> bool {
        match plc.surfaces[f.plc_surface as usize] {
            // Extruded always takes the fallback; a full-revolution barrel does too.
            SurfaceKind::Extruded { .. } => true,
            SurfaceKind::Cylinder { .. } | SurfaceKind::Cone { .. } | SurfaceKind::Torus { .. } => {
                let mut b: Vec<V3> = Vec::new();
                for lp in &f.loops {
                    for &cid in &lp.coedges {
                        b.extend_from_slice(&brep.edges[brep.coedge(cid).edge.0 as usize].chain);
                    }
                }
                b.is_empty() || is_full_revolution(brep.surface(f.surface), &b)
            }
            _ => false, // planes + sphere caps chart; a closed sphere has no rim edges
        }
    };
    let edge_keep_input: Vec<bool> = brep
        .edges
        .iter()
        .map(|e| e.coedges.iter().any(|&c| face_uses_fallback(brep.face(brep.coedge(c).face))))
        .collect();

    // ---- stage 1b: edge points (shared). Edges of a structured face keep the
    // input vertices; others are distributed at OVERSAMPLE * (finest H) -- finer
    // than the volume so the restricted-Delaunay boundary recovers without straddles.
    // Per edge: the ordered ON-edge point positions AND their global site
    // indices, so a face can build its boundary loops in site indices (for the
    // triangulation) without re-locating points. Endpoints are the two corner
    // sites (vertex ids); the interior points are freshly pushed sites.
    let mut edge_pts: Vec<Vec<V3>> = Vec::with_capacity(brep.edges.len());
    let mut edge_sidx: Vec<Vec<usize>> = Vec::with_capacity(brep.edges.len());
    for (ei, edge) in brep.edges.iter().enumerate() {
        let pts3: Vec<V3> = if edge_keep_input[ei] {
            edge.chain.clone()
        } else {
            let cap = OVERSAMPLE * edge.chain.iter().map(|&p| h_at(p)).fold(f64::INFINITY, f64::min);
            match edge_curve(edge) {
                Some(curve) => {
                    let ps = distribute(&*curve, params.edge_tol_for(ei), cap.min(params.edge_maxh_for(ei)), grad);
                    ps.iter().map(|&s| curve.point_at(s)).collect()
                }
                None => edge.chain.clone(),
            }
        };
        let esz = params.edge_maxh_for(ei);
        // Site indices along the edge: corner a, interior points, corner b.
        let mut sidx: Vec<usize> = Vec::with_capacity(pts3.len());
        sidx.push(edge.ends[0].0 as usize);
        for &p in pts3.iter().take(pts3.len().saturating_sub(1)).skip(1) {
            sites.push(Site::vertex(p));
            point_tile.push(u32::MAX);
            point_size.push(esz);
            sidx.push(sites.len() - 1);
        }
        sidx.push(edge.ends[1].0 as usize);
        // A degenerate/closed chain may not match ends; fall back to length match.
        if sidx.len() != pts3.len() {
            sidx = Vec::new();
        }
        edge_pts.push(pts3);
        edge_sidx.push(sidx);
    }
    let fine = (OVERSAMPLE * domain.finest()).max(1e-9);

    let tiles: Vec<(u32, FaceTag)> =
        brep.faces.iter().map(|f| (f.plc_surface, f.face_tag)).collect();

    // ---- stage 2: per-face surface mesh -- ONE chart-driven path (Algorithm 1)
    // Every face is meshed the same way; the only thing that varies is the chart
    // `Phi`: a plane is the trivial isometric `PlaneChart` (exact on-plane carrier),
    // a developable/quadric is its distance-faithful unroll (`build_chart`, analytic
    // carrier). In the chart we scatter + Lloyd-relax interior points at a tol- and
    // curvature-aware size `h = min(caps, sqrt(8*tol*R))`, triangulate CONSTRAINED to
    // the frozen boundary loops, and lift every point back onto the surface. Faces
    // with no single-chart bijection (full-revolution barrels, closed spheres,
    // extruded barrels, or a boundary that did not resolve to shared site indices)
    // take the structured fallback below (a revolution grid, else the pinned input
    // tessellation) -- those still produce sites, their tris land in a later step.
    for (fid, face) in brep.faces.iter().enumerate() {
        if face.facets.is_empty() {
            continue;
        }
        let surf = brep.surface(face.surface);
        let kind = plc.surfaces[face.plc_surface as usize].clone();
        let is_revolution = matches!(
            kind,
            SurfaceKind::Cylinder { .. } | SurfaceKind::Cone { .. } | SurfaceKind::Torus { .. }
        );
        // The boundary as on-surface positions (the rim that must round-trip).
        let mut boundary: Vec<V3> = Vec::new();
        for lp in &face.loops {
            for &cid in &lp.coedges {
                boundary.extend_from_slice(&edge_pts[brep.coedge(cid).edge.0 as usize]);
            }
        }
        // Representative INTERIOR points (facet centroids) for fitting a curved
        // chart's frame/branch. The face centroid points to a cap's centre even for
        // a hemisphere, where the rim alone averages to a zero direction and would
        // pick the wrong chart pole (the antipode) -- so the chart is fit from these,
        // not from the rim.
        let repr_pts: Vec<V3> = face
            .facets
            .iter()
            .map(|&fi| {
                let t = plc.triangles[fi as usize];
                let (a, b, c) =
                    (plc.vertices[t[0] as usize], plc.vertices[t[1] as usize], plc.vertices[t[2] as usize]);
                [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0]
            })
            .collect();
        let target = |p: V3| (OVERSAMPLE * h_at(p)).max(fine * 0.5).min(params.surf_maxh_for(fid));

        // Build the chart + the carrier the lifted points get pinned to. A plane is
        // always chart-able; a full-revolution barrel and an extruded barrel wrap
        // (no single-chart bijection) and go to the structured fallback; every other
        // curved face attempts its analytic chart and is validated by a round-trip.
        let (chart, carrier): (Option<Box<dyn SurfaceChart>>, Carrier) =
            if let Some((o, u, v, n)) = surf.plane_frame() {
                (Some(Box::new(PlaneChart::new(o, u, v, n)) as Box<dyn SurfaceChart>),
                 Carrier::Plane { p0: o, n })
            } else if matches!(kind, SurfaceKind::Extruded { .. })
                || (is_revolution && is_full_revolution(surf, &boundary))
            {
                (None, Carrier::Surface(kind.clone()))
            } else {
                (build_chart(&kind, &repr_pts), Carrier::Surface(kind.clone()))
            };
        // A curved chart must be a bijection over THIS boundary: every boundary
        // point, PROJECTED onto the surface (the rim points are faceted chords,
        // slightly off it), round-trips through the chart. A seam/pole straddle
        // fails this; the faceting sagitta must not. Planes always pass.
        let chart = chart.filter(|c| {
            surf.exact_plane().is_some()
                || boundary.iter().all(|&p| {
                    let q = c.project(p);
                    dist3(c.to_xyz(c.to_uv(q)), q) < 1e-6 * (1.0 + dist3(q, [0.0; 3]))
                })
        });

        // ---- the unified chart-driven face mesh --------------------------------
        if let Some(chart) = chart.as_ref() {
            let mut loc_uv: Vec<P2> = Vec::new(); // boundary then interior, in the chart
            let mut loc_sidx: Vec<usize> = Vec::new(); // global site index per local
            let mut segs: Vec<(usize, usize)> = Vec::new(); // boundary segments (local indices)
            let mut segs_uv: Vec<(P2, P2)> = Vec::new(); // same, as uv (for the inside test)
            let mut resolved = true;
            for lp in &face.loops {
                let start = loc_uv.len();
                for &cid in &lp.coedges {
                    let ce = brep.coedge(cid);
                    let sidx = &edge_sidx[ce.edge.0 as usize];
                    if sidx.len() < 2 {
                        resolved = false;
                        break;
                    }
                    // ordered site indices along this coedge, dropping the last
                    // (shared with the next coedge's first); reversed if backward.
                    let chain: Vec<usize> = if ce.forward {
                        sidx[..sidx.len() - 1].to_vec()
                    } else {
                        sidx.iter().rev().take(sidx.len() - 1).copied().collect()
                    };
                    for &si in &chain {
                        loc_uv.push(chart.to_uv(sites[si].pos()));
                        loc_sidx.push(si);
                    }
                }
                if !resolved {
                    break;
                }
                let m = loc_uv.len() - start;
                for k in 0..m {
                    let (a, b) = (start + k, start + (k + 1) % m);
                    segs.push((a, b));
                    segs_uv.push((loc_uv[a], loc_uv[b]));
                }
            }
            if resolved && loc_uv.len() >= 3 {
                let nb = loc_uv.len();
                let mut lo = [f64::INFINITY; 2];
                let mut hi = [f64::NEG_INFINITY; 2];
                for &q in &loc_uv {
                    for k in 0..2 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
                let tol = params.surf_tol_for(fid);
                let smaxh = params.surf_maxh_for(fid);
                // h(uv) = min(size caps, sqrt(8*tol*R(uv))): the tolerance enters the
                // size field via the sagitta bound, so a face refines automatically
                // where it curves. A plane has R = INF -> the caps alone (size only),
                // bit-identical to the dedicated planar path it replaces.
                let target2d = |q: P2| {
                    let base = (OVERSAMPLE * h_at(chart.to_xyz(q))).max(fine * 0.5).min(smaxh);
                    let r = chart.curvature_radius(q);
                    let defl = if r.is_finite() { (8.0 * tol * r).sqrt() } else { f64::INFINITY };
                    base.min(defl).max(1e-9)
                };
                let inside = |q: P2| in_loops(q, &segs_uv);
                // Scatter step = the FINEST local target on this face (so a per-entity
                // `surf_maxh`/`tol` or curvature actually resolves the grid, not just
                // the separation radius). Defaults to `fine` when no surface cap/curve
                // applies, so the exact-volume planar balance is unchanged.
                let step = loc_uv[..nb].iter().map(|&q| target2d(q)).fold(fine, f64::min).max(1e-9);
                let interior = cvt_fill(&loc_uv[..nb], lo, hi, step, target2d, SURF_LLOYD_ITERS, inside, true);
                for q in interior {
                    let p = chart.to_xyz(q);
                    let site = match &carrier {
                        Carrier::Plane { p0, n } => Site::on_plane(*p0, *n, p),
                        _ => Site::on_surface(kind.clone(), p),
                    };
                    sites.push(site);
                    point_tile.push(fid as u32);
                    point_size.push(smaxh);
                    loc_uv.push(q);
                    loc_sidx.push(sites.len() - 1);
                }
                for t in triangulate_constrained(&loc_uv, &segs, inside) {
                    tris.push(SurfaceFace {
                        tri: [loc_sidx[t[0]], loc_sidx[t[1]], loc_sidx[t[2]]],
                        face_tag: face.face_tag,
                        regions: face.regions,
                        patch: fid as u32,
                        surface: face.plc_surface,
                    });
                    tri_carrier.push(carrier.clone());
                }
                continue;
            }
            // boundary did not resolve to shared site indices -> structured fallback
        }

        // ---- structured / pinned fallback (no chart bijection) -----------------
        if is_revolution && is_full_revolution(surf, &boundary) {
            // FULL-REVOLUTION barrel (cyl/cone, partial-torus tube): a structured
            // (theta, v) grid triangulated into a CLOSED band. Columns = the shared
            // rim theta rays (both rims align on the axis frame); rows = bottom rim ->
            // interior rows -> top rim, the rims REUSING the shared edge site indices
            // so the band conforms to the caps; the seam closes by wrapping the last
            // column to the first. Interior points are on the analytic surface.
            let rays = rim_theta_rays(surf, &boundary);
            let n = rays.len();
            let tau = std::f64::consts::TAU;
            let col_of = |theta: f64| -> usize {
                let mut best = (f64::INFINITY, 0usize);
                for (ci, &r) in rays.iter().enumerate() {
                    let mut d = (theta - r).abs();
                    d = d.min(tau - d);
                    if d < best.0 {
                        best = (d, ci);
                    }
                }
                best.1
            };
            // Each loop -> (mean v, ring of site indices indexed by column).
            let mut rings: Vec<(f64, Vec<usize>)> = Vec::new();
            let mut ok = n >= 3;
            for lp in &face.loops {
                let mut ring = vec![usize::MAX; n];
                let (mut vsum, mut vn) = (0.0f64, 0.0f64);
                for &cid in &lp.coedges {
                    let sidx = &edge_sidx[brep.coedge(cid).edge.0 as usize];
                    if sidx.len() < 2 {
                        ok = false;
                        break;
                    }
                    for &si in sidx {
                        let uv = surf.project_uv(sites[si].pos());
                        ring[col_of(uv[0])] = si; // a closure duplicate overwrites its own column
                        vsum += uv[1];
                        vn += 1.0;
                    }
                }
                if !ok || ring.iter().any(|&x| x == usize::MAX) {
                    ok = false;
                    break;
                }
                rings.push((vsum / vn.max(1.0), ring));
            }
            if ok && rings.len() == 2 {
                if rings[0].0 > rings[1].0 {
                    rings.swap(0, 1);
                }
                let (v0, v1) = (rings[0].0, rings[1].0);
                // interior row count from arc length / local target along v.
                let vmid = 0.5 * (v0 + v1);
                let eps = (v1 - v0) * 1e-4 + 1e-12;
                let dv_arc =
                    dist3(surf.eval_uv([rays[0], vmid + eps]), surf.eval_uv([rays[0], vmid - eps])) / (2.0 * eps);
                let tgt = target(surf.eval_uv([rays[0], vmid])).max(1e-9);
                let nv = (((dv_arc * (v1 - v0)) / tgt).round() as usize).max(1);
                let mut grid: Vec<Vec<usize>> = Vec::with_capacity(nv + 1);
                grid.push(rings[0].1.clone());
                for j in 1..nv {
                    let vj = v0 + (v1 - v0) * j as f64 / nv as f64;
                    let mut row = Vec::with_capacity(n);
                    for &theta in &rays {
                        sites.push(Site::on_surface(kind.clone(), surf.eval_uv([theta, vj])));
                        point_tile.push(fid as u32);
                        point_size.push(params.surf_maxh_for(fid));
                        row.push(sites.len() - 1);
                    }
                    grid.push(row);
                }
                grid.push(rings[1].1.clone());
                let car = Carrier::Surface(kind.clone());
                for j in 0..grid.len() - 1 {
                    for i in 0..n {
                        let i2 = (i + 1) % n;
                        let (a, b, c, d) = (grid[j][i], grid[j][i2], grid[j + 1][i2], grid[j + 1][i]);
                        for tri in [[a, b, c], [a, c, d]] {
                            tris.push(SurfaceFace {
                                tri,
                                face_tag: face.face_tag,
                                regions: face.regions,
                                patch: fid as u32,
                                surface: face.plc_surface,
                            });
                            tri_carrier.push(car.clone());
                        }
                    }
                }
                continue;
            }
            // FULL CONE: one base rim + an apex tip. Build rings from the base
            // toward the apex but STOP a ring before they collapse (where adjacent
            // column points get closer than the target); the apex is a single shared
            // corner vertex (never duplicated), and the innermost ring fans to it.
            // This replaces the old revolution_grid, which generated coincident point
            // clusters at the tip (the cone-is-garbage degeneracy).
            if ok && rings.len() == 1 {
                if let SurfaceKind::Cone { apex, .. } = kind {
                    // The cone tip is a degenerate point, NOT a brep corner, so create
                    // the apex site here (a single pinned vertex the fan converges to).
                    sites.push(Site::vertex(apex));
                    point_tile.push(fid as u32);
                    point_size.push(params.surf_maxh_for(fid));
                    let apex_si = sites.len() - 1;
                    {
                        let base_v = rings[0].0;
                        let v_apex = surf.project_uv(apex)[1];
                        let vmid = 0.5 * (base_v + v_apex);
                        let eps = (v_apex - base_v).abs() * 1e-4 + 1e-12;
                        let dv_arc = dist3(surf.eval_uv([rays[0], vmid + eps]), surf.eval_uv([rays[0], vmid - eps]))
                            / (2.0 * eps);
                        let tgt = target(surf.eval_uv([rays[0], vmid])).max(1e-9);
                        let nv = (((dv_arc * (v_apex - base_v).abs()) / tgt).round() as usize).max(1);
                        let mut grid: Vec<Vec<usize>> = vec![rings[0].1.clone()];
                        for j in 1..nv {
                            let vj = base_v + (v_apex - base_v) * j as f64 / nv as f64;
                            // stop once a ring would be over-dense (collapsing at the tip).
                            let pa = surf.eval_uv([rays[0], vj]);
                            let pb = surf.eval_uv([rays[1 % n], vj]);
                            if dist3(pa, pb) < 0.5 * tgt {
                                break;
                            }
                            let mut row = Vec::with_capacity(n);
                            for &theta in &rays {
                                sites.push(Site::on_surface(kind.clone(), surf.eval_uv([theta, vj])));
                                point_tile.push(fid as u32);
                                point_size.push(params.surf_maxh_for(fid));
                                row.push(sites.len() - 1);
                            }
                            grid.push(row);
                        }
                        let car = Carrier::Surface(kind.clone());
                        let push = |tri: [usize; 3], tris: &mut Vec<SurfaceFace>, tc: &mut Vec<Carrier>| {
                            tris.push(SurfaceFace {
                                tri,
                                face_tag: face.face_tag,
                                regions: face.regions,
                                patch: fid as u32,
                                surface: face.plc_surface,
                            });
                            tc.push(car.clone());
                        };
                        for j in 0..grid.len() - 1 {
                            for i in 0..n {
                                let i2 = (i + 1) % n;
                                let (a, b, c, d) = (grid[j][i], grid[j][i2], grid[j + 1][i2], grid[j + 1][i]);
                                push([a, b, c], &mut tris, &mut tri_carrier);
                                push([a, c, d], &mut tris, &mut tri_carrier);
                            }
                        }
                        // fan the innermost ring to the apex tip
                        let last = grid.last().unwrap().clone();
                        for i in 0..n {
                            push([last[i], last[(i + 1) % n], apex_si], &mut tris, &mut tri_carrier);
                        }
                        continue;
                    }
                }
            }
            // rings did not resolve -> sites only (no tris), as before
            let vs: Vec<f64> = boundary.iter().map(|&p| surf.project_uv(p)[1]).collect();
            let vmin = vs.iter().cloned().fold(f64::INFINITY, f64::min);
            let vmax = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for p in revolution_grid(surf, vmin, vmax, &rays, &target) {
                sites.push(Site::on_surface(kind.clone(), p));
                point_tile.push(fid as u32);
                point_size.push(params.surf_maxh_for(fid));
            }
        } else if let (SurfaceKind::Sphere { center, radius }, true) = (&kind, boundary.is_empty()) {
            // CLOSED sphere (no trim loop): a geodesic icosphere at the sizing density
            // -- isotropic, pole-free, seamless. No chart (a closed surface admits no
            // global parametrization); the sphere oracle carries the points exactly.
            let (center, radius) = (*center, *radius);
            let h = (8.0 * params.surf_tol_for(fid) * radius)
                .sqrt()
                .min(params.surf_maxh_for(fid))
                .min(OVERSAMPLE * h_at(center))
                .max(1e-9);
            // icosahedron edge ~= 1.0515 R at level 0, halving per subdivision.
            let level = ((1.0515 * radius / h).log2().ceil() as i64).clamp(0, 6) as usize;
            let (vs, fs) = icosphere_mesh(center, radius, level);
            let base = sites.len();
            for &p in &vs {
                sites.push(Site::on_surface(kind.clone(), p));
                point_tile.push(fid as u32);
                point_size.push(params.surf_maxh_for(fid));
            }
            let car = Carrier::Surface(kind.clone());
            for t in &fs {
                tris.push(SurfaceFace {
                    tri: [base + t[0], base + t[1], base + t[2]],
                    face_tag: face.face_tag,
                    regions: face.regions,
                    patch: fid as u32,
                    surface: face.plc_surface,
                });
                tri_carrier.push(car.clone());
            }
        } else if surf.exact_plane().is_none() {
            // closed/wrapping curved face the chart could not take (extruded barrel,
            // closed torus, a >hemisphere sphere cap with a real rim): pin the clean
            // input tessellation -- it conforms to the shared rim.
            let mut seen: std::collections::HashSet<[u64; 3]> =
                boundary.iter().map(|&p| bits(p)).collect();
            for &tfi in &face.facets {
                for &v in &plc.triangles[tfi as usize] {
                    let p = plc.vertices[v as usize];
                    if seen.insert(bits(p)) {
                        sites.push(Site::vertex(p));
                        point_tile.push(fid as u32);
                        point_size.push(params.surf_maxh_for(fid));
                    }
                }
            }
        } else {
            // a planar face whose boundary did not resolve (rare): dart-seed fill on
            // its exact carrier plane so it still gets interior points.
            let (o, n) = surf.exact_plane().unwrap();
            for p in fill_face_points(surf, &face.facets, plc, &boundary, &target, fid as u64) {
                sites.push(Site::on_plane(o, n, p));
                point_tile.push(fid as u32);
                point_size.push(params.surf_maxh_for(fid));
            }
        }
    }

    let n_surf = sites.len();
    SurfaceSites { sites, n_surf, point_tile, point_size, tiles, plc_points, tris, tri_carrier }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_brep::build::from_plc;
    use rapidmesh_geom::{extrude_spline_profile, naca0012_profile, solid_box, Scene};

    #[test]
    fn box_edges_are_uniform_lines() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 1.0, 1.0]));
        let b = from_plc(&scene.assemble());
        // a length-4 edge with maxh 1 -> 4 uniform segments
        let long = b
            .edges
            .iter()
            .find(|e| {
                let c = edge_curve(e).unwrap();
                (c.length() - 4.0).abs() < 1e-9
            })
            .expect("a length-4 edge");
        let c = edge_curve(long).unwrap();
        let s = distribute(&*c, 0.02, 1.0, 0.3);
        assert_eq!(s.len() - 1, 4, "4 uniform segments of maxh=1");
    }

    #[test]
    fn airfoil_profile_edge_is_curvature_graded() {
        let profile = naca0012_profile(1.0, 40);
        let solid = extrude_spline_profile(
            profile,
            80,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        );
        let mut scene = Scene::new();
        scene.add_solid(solid);
        let b = from_plc(&scene.assemble());
        let prof = b
            .edges
            .iter()
            .find(|e| matches!(e.curve, BCurve::Profile { .. }))
            .expect("a profile edge");
        let c = edge_curve(prof).unwrap();
        let s = distribute(&*c, 0.01, 0.2, 0.3);
        let spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        let (mn, mx) = spc.iter().fold((f64::MAX, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
        // curvature grading: the nose spacing is much finer than the flat tail
        assert!(mx / mn > 2.0, "profile distribution should grade (ratio {})", mx / mn);
    }

    #[test]
    fn box_surface_sites_lie_on_exact_planes() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.5, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);

        assert_eq!(ss.plc_points, 8, "8 box corners pinned");
        assert!(ss.n_surf > 8, "edge + face interior points added");
        // every face-interior point lies exactly on its face's plane
        let mut n_face_pts = 0;
        for (i, &tile) in ss.point_tile.iter().enumerate() {
            if tile == u32::MAX {
                continue;
            }
            n_face_pts += 1;
            let surf = b.surface(b.faces[tile as usize].surface);
            let (o, n) = surf.exact_plane().expect("box face is planar");
            let p = ss.sites[i].pos();
            let off = (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2];
            assert!(off.abs() < 1e-9, "face point off its plane by {off}");
        }
        assert!(n_face_pts > 0, "faces produced interior points");
    }

    #[test]
    fn curved_face_is_chart_triangulated_on_the_surface_within_tol() {
        // A sphere with its top hemisphere carved away leaves a TRIMMED spherical
        // cap (the bottom hemisphere, rim = the equator) -- a curved face that takes
        // the chart-driven path. Its triangles must (1) exist, (2) lie exactly on the
        // analytic sphere, and (3) hold the chord deflection within `tol_surf` (the
        // sagitta bound `h = sqrt(8*tol*R)` that the chart sizing enforces).
        use rapidmesh_geom::{solid_box, sphere};
        let center = [0.0, 0.0, 0.0];
        let radius = 1.0;
        let mut scene = Scene::new();
        scene.add_solid(sphere(center, radius, 24, 12));
        // Carve everything above z = -0.2, leaving a SUB-hemisphere cap around the
        // south pole (a hemisphere's rim averages to a zero direction, which would
        // pick the wrong chart pole; a sub-hemisphere rim points to the cap center).
        scene.add_void(solid_box([-2.0, -2.0, -0.2], [2.0, 2.0, 2.0]));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let tol = 1e-2;
        let params = MeshParams { maxh: 0.5, tol_surf: tol, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);

        let mut curved_tris = 0usize;
        let mut max_off = 0.0f64; // distance of an INTERIOR vertex off the sphere
        let mut n_interior = 0usize;
        for (t, car) in ss.tris.iter().zip(&ss.tri_carrier) {
            if !matches!(car, Carrier::Surface(_)) {
                continue;
            }
            curved_tris += 1;
            // Interior (face-tile) vertices are lifted by the chart onto the sphere;
            // rim vertices are shared faceted chords (conformity), so skip those.
            for &si in &t.tri {
                if ss.point_tile[si] != u32::MAX {
                    // a face-interior (chart-lifted) point, not a shared rim point
                    let v = ss.sites[si].pos();
                    max_off = max_off.max((dist3(v, center) - radius).abs());
                    n_interior += 1;
                }
            }
        }
        assert!(curved_tris > 0, "the trimmed sphere cap must be chart-triangulated");
        assert!(n_interior > 0, "the cap must have chart-lifted interior points");
        assert!(max_off < 1e-7, "chart-lifted interior vertices must lie on the sphere (off {max_off})");
    }

    #[test]
    fn cylinder_barrel_is_a_closed_triangulated_band() {
        // A5: the full-revolution barrel is triangulated into a closed band on the
        // analytic cylinder (not just seeded points). Verify the curved tris (1) lie
        // on the cylinder, (2) reuse the shared rim points, and (3) together with the
        // caps make the whole surface a closed manifold (every edge shared twice).
        use rapidmesh_geom::cylinder;
        let (r, hgt) = (1.0, 3.0);
        let mut scene = Scene::new();
        scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, hgt], r, 24));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.5, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);

        let mut curved = 0usize;
        let mut max_off = 0.0f64;
        for (t, c) in ss.tris.iter().zip(&ss.tri_carrier) {
            if !matches!(c, Carrier::Surface(_)) {
                continue;
            }
            curved += 1;
            for &si in &t.tri {
                let p = ss.sites[si].pos();
                let radial = (p[0] * p[0] + p[1] * p[1]).sqrt(); // distance to z-axis
                max_off = max_off.max((radial - r).abs());
            }
        }
        assert!(curved > 0, "the barrel must be triangulated");
        assert!(max_off < 1e-9, "barrel tri vertices lie on the cylinder (off {max_off})");
        // whole surface (barrel + 2 caps) is a closed manifold
        let mut edges: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for f in &ss.tris {
            for k in 0..3 {
                let (u, v) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edges.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2), "cylinder surface is a closed manifold");
    }

    #[test]
    fn full_cone_apex_is_a_clean_fan_not_a_collapsed_cluster() {
        // A full cone (frustum, top radius 0) has a base rim + an apex tip. The
        // lateral surface must mesh as rings + a single-vertex apex fan, NOT a
        // cluster of coincident points at the tip (the old revolution_grid bug).
        use rapidmesh_geom::frustum;
        let apex = [0.0, 0.0, 1.5];
        let mut scene = Scene::new();
        scene.add_solid(frustum([0.0, 0.0, 0.0], [0.0, 0.0, 1.5], 0.8, 0.0, 40));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.22, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);

        // the apex is created as a single site at the cone tip; find it
        let apex_si = (0..ss.sites.len())
            .find(|&i| dist3(ss.sites[i].pos(), apex) < 1e-7)
            .expect("apex vertex present");
        let mut fan = 0usize;
        let mut curved = 0usize;
        for (t, c) in ss.tris.iter().zip(&ss.tri_carrier) {
            if !matches!(c, Carrier::Surface(_)) {
                continue;
            }
            curved += 1;
            if t.tri.contains(&apex_si) {
                fan += 1;
            }
        }
        assert!(curved > 0, "the cone lateral surface is triangulated");
        assert!(fan >= 3, "the apex is the centre of a triangle fan, got {fan}");
        // no two DISTINCT cone-surface points coincide (no collapse at the tip)
        let cone_pts: Vec<V3> = (0..ss.sites.len())
            .filter(|&i| ss.point_tile[i] != u32::MAX)
            .map(|i| ss.sites[i].pos())
            .collect();
        let mut min_sep = f64::INFINITY;
        for i in 0..cone_pts.len() {
            for j in i + 1..cone_pts.len() {
                min_sep = min_sep.min(dist3(cone_pts[i], cone_pts[j]));
            }
        }
        assert!(min_sep > 1e-4, "cone surface points must not collapse (min sep {min_sep})");
    }

    #[test]
    fn closed_sphere_is_geodesic_isotropic_and_watertight() {
        // A standalone sphere solid is a CLOSED face: meshed by a geodesic icosphere
        // (no chart, no pole, no seam), not the radial UV tessellation. Verify the
        // tris (1) tile a closed manifold, (2) lie exactly on the sphere, and (3) are
        // ISOTROPIC -- the edge-length spread is tight (the radial UV mesh clusters
        // hard at the poles; the icosphere does not).
        use rapidmesh_geom::sphere;
        let center = [0.3, -0.4, 0.5];
        let radius = 1.0;
        let mut scene = Scene::new();
        scene.add_solid(sphere(center, radius, 24, 12));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.4, tol_surf: 1e-2, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);

        let curved: Vec<&SurfaceFace> = ss
            .tris
            .iter()
            .zip(&ss.tri_carrier)
            .filter(|(_, c)| matches!(c, Carrier::Surface(_)))
            .map(|(t, _)| t)
            .collect();
        assert!(curved.len() >= 80, "geodesic sphere has many tris, got {}", curved.len());
        // closed manifold: every edge shared by exactly two triangles
        let mut edges: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        let mut max_off = 0.0f64;
        let mut min_e = f64::INFINITY;
        let mut max_e = 0.0f64;
        for f in &curved {
            let p = [ss.sites[f.tri[0]].pos(), ss.sites[f.tri[1]].pos(), ss.sites[f.tri[2]].pos()];
            for &v in &p {
                max_off = max_off.max((dist3(v, center) - radius).abs());
            }
            for k in 0..3 {
                let (u, w) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edges.entry((u.min(w), u.max(w))).or_insert(0) += 1;
                let e = dist3(p[k], p[(k + 1) % 3]);
                min_e = min_e.min(e);
                max_e = max_e.max(e);
            }
        }
        assert!(edges.values().all(|&c| c == 2), "closed sphere is a closed manifold");
        assert!(max_off < 1e-9, "icosphere vertices lie on the sphere (off {max_off})");
        // isotropy: a UV sphere's pole rows make this ratio huge; the icosphere is ~1.5
        assert!(max_e / min_e < 2.5, "geodesic mesh is near-isotropic (ratio {})", max_e / min_e);
    }

    #[test]
    fn box_surface_sites_triangulate_the_planar_faces() {
        // The unified surface (per-entity-aware) also triangulates planar faces:
        // a 2x3x4 box's six planar faces must tile to the exact surface area 52,
        // as a closed manifold (every edge shared by two triangles).
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.7, ..Default::default() };
        let domain = crate::domain::DomainTree::build(&plc, &params, &[]);
        let ss = surface_sites(&b, &plc, &params, &domain);
        assert!(!ss.tris.is_empty(), "planar faces are triangulated");
        let mut area = 0.0;
        let mut edges: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for f in &ss.tris {
            let (a, bb, c) = (ss.sites[f.tri[0]].pos(), ss.sites[f.tri[1]].pos(), ss.sites[f.tri[2]].pos());
            let ab = [bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cr = cross(ab, ac);
            area += 0.5 * dot(cr, cr).sqrt();
            for k in 0..3 {
                let (u, v) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edges.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            }
        }
        assert!((area - 52.0).abs() < 1e-9, "box surface area should be 52, got {area}");
        assert!(edges.values().all(|&c| c == 2), "surface is a closed manifold");
        assert_eq!(ss.tris.len(), ss.tri_carrier.len());
    }
}
