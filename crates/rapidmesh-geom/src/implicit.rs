//! Implicit (SDF) surface carrier: signed-distance expression trees, the
//! [`ImplicitSurface`] carrier queried by the mesher, and the surface-nets
//! tessellation that lets an implicit solid enter the pipeline.
//!
//! The design mirrors the STL import path (`import.rs`): an implicit solid is
//! tessellated once into a tagged [`Faceted`] proxy whose facets all carry ONE
//! [`SurfaceKind::Implicit`] carrier. The exact-CSG arrangement operates on
//! the proxy triangles like on any other input; refinement and optimization
//! then project onto the *analytic* field via gradient Newton, so the final
//! mesh converges to the true implicit surface — the same
//! discretize-then-remesh contract as `SurfaceKind::Discrete`, but with an
//! exact carrier instead of the frozen soup.
//!
//! Fields are signed-distance-*like*: primitives are exact SDFs, but smooth
//! booleans and non-uniform placements only bound the distance. Every
//! consumer therefore uses first-order normalized queries (`F/|∇F|`) rather
//! than trusting the raw field value as a metric distance.

use crate::faceted::{Faceted, SurfaceKind};
use rapidmesh_csg::Tri;
use crate::vec3::{add, cross, dot, len, normalize, scale, sub, V3};

// ======================================================================================
// SDF expression tree
// ======================================================================================

/// A signed-distance expression: negative inside, positive outside. A data
/// tree (not closures) so carriers stay `Debug`-printable, cloneable, and
/// composable under placement transforms.
#[derive(Debug, Clone)]
pub enum Sdf {
    /// Sphere around `center`.
    Sphere { center: V3, radius: f64 },
    /// Axis-aligned box around `center` with half extents `half` (exact SDF,
    /// including corner distance outside).
    Box { center: V3, half: V3 },
    /// Capped cylinder between the axis endpoints `a` and `b`.
    Cylinder { a: V3, b: V3, radius: f64 },
    /// Capsule (sphere-swept segment) between `a` and `b`.
    Capsule { a: V3, b: V3, radius: f64 },
    /// Torus around `center` with plane normal `axis`.
    Torus { center: V3, axis: V3, major: f64, minor: f64 },
    /// Half space: negative on the anti-`normal` side of `point`.
    HalfSpace { point: V3, normal: V3 },
    /// Boolean union, `min(a, b)`.
    Union(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Boolean intersection, `max(a, b)`.
    Intersect(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Boolean difference, `max(a, -b)`.
    Difference(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Smooth (filleted) union with blend radius `k` (polynomial smooth-min).
    SmoothUnion { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Smooth intersection with blend radius `k`.
    SmoothIntersect { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Smooth difference with blend radius `k`.
    SmoothDifference { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Offset surface: grows the solid by `d` (rounds convex edges — the
    /// fillet/plating/coating primitive).
    Offset { a: std::boxed::Box<Sdf>, d: f64 },
    /// Shell of half thickness `t` around the zero set, `|a| - t`.
    Shell { a: std::boxed::Box<Sdf>, t: f64 },
}

/// Polynomial smooth-min (quadratic, Inigo Quilez formulation): equals
/// `min(a, b)` outside the `k`-band around `a = b`, blends inside it.
#[inline]
fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

impl Sdf {
    /// Field value at `p` (negative inside).
    pub fn eval(&self, p: V3) -> f64 {
        match self {
            Sdf::Sphere { center, radius } => len(sub(p, *center)) - radius,
            Sdf::Box { center, half } => {
                let q: V3 = std::array::from_fn(|i| (p[i] - center[i]).abs() - half[i]);
                let outside: V3 = std::array::from_fn(|i| q[i].max(0.0));
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                len(outside) + inside
            }
            Sdf::Cylinder { a, b, radius } => {
                let ab = sub(*b, *a);
                let l = len(ab);
                let axis = scale(ab, 1.0 / l);
                let ap = sub(p, *a);
                let t = dot(ap, axis);
                let radial = len(sub(ap, scale(axis, t))) - radius;
                let axial = (-t).max(t - l);
                let (qr, qa) = (radial.max(0.0), axial.max(0.0));
                radial.max(axial).min(0.0) + (qr * qr + qa * qa).sqrt()
            }
            Sdf::Capsule { a, b, radius } => {
                let ab = sub(*b, *a);
                let t = (dot(sub(p, *a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
                len(sub(p, add(*a, scale(ab, t)))) - radius
            }
            Sdf::Torus { center, axis, major, minor } => {
                let n = normalize(*axis);
                let cp = sub(p, *center);
                let h = dot(cp, n);
                let radial = len(sub(cp, scale(n, h)));
                let d = radial - major;
                (d * d + h * h).sqrt() - minor
            }
            Sdf::HalfSpace { point, normal } => dot(sub(p, *point), normalize(*normal)),
            Sdf::Union(a, b) => a.eval(p).min(b.eval(p)),
            Sdf::Intersect(a, b) => a.eval(p).max(b.eval(p)),
            Sdf::Difference(a, b) => a.eval(p).max(-b.eval(p)),
            Sdf::SmoothUnion { a, b, k } => smooth_min(a.eval(p), b.eval(p), *k),
            Sdf::SmoothIntersect { a, b, k } => -smooth_min(-a.eval(p), -b.eval(p), *k),
            Sdf::SmoothDifference { a, b, k } => -smooth_min(-a.eval(p), b.eval(p), *k),
            Sdf::Offset { a, d } => a.eval(p) - d,
            Sdf::Shell { a, t } => a.eval(p).abs() - t,
        }
    }
}

// ======================================================================================
// The carrier
// ======================================================================================

/// An implicit surface as a meshing carrier: the zero set of an [`Sdf`] under
/// an optional placement affine, restricted to a local-frame bounding box.
///
/// Mirrors the query API of `DiscreteSurface`: closest-point projection with
/// outward normal, curvature radius for the sizing field, first-order signed
/// offset, and segment intersections for the crossing pulls.
#[derive(Debug)]
pub struct ImplicitSurface {
    sdf: Sdf,
    /// Placement: world = linear · local + offset (identity at construction;
    /// composed by [`Self::transformed`]).
    linear: [[f64; 3]; 3],
    offset: V3,
    /// Inverse of `linear` (placements are invertible by construction — the
    /// scene layer only produces rotations/uniform scales/translations).
    inv_linear: [[f64; 3]; 3],
    /// Local-frame axis-aligned bounds the zero set lives in (with margin).
    pub bbox: (V3, V3),
    /// Query length scale: the finite-difference step and the projection
    /// tolerance derive from the bbox diagonal.
    scale: f64,
}

fn mat_vec(m: &[[f64; 3]; 3], v: V3) -> V3 {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(det.abs() > 1e-300, "implicit placement must be invertible");
    let inv_det = 1.0 / det;
    let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    [
        [cof(1, 2, 1, 2) * inv_det, -cof(0, 2, 1, 2) * inv_det, cof(0, 1, 1, 2) * inv_det],
        [-cof(1, 2, 0, 2) * inv_det, cof(0, 2, 0, 2) * inv_det, -cof(0, 1, 0, 2) * inv_det],
        [cof(1, 2, 0, 1) * inv_det, -cof(0, 2, 0, 1) * inv_det, cof(0, 1, 0, 1) * inv_det],
    ]
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl ImplicitSurface {
    /// New carrier in its own frame, bounded by `bbox` (must enclose the zero
    /// set with some margin; the tessellation samples only inside it).
    pub fn new(sdf: Sdf, bbox: (V3, V3)) -> ImplicitSurface {
        let scale = len(sub(bbox.1, bbox.0));
        assert!(scale > 0.0, "implicit bbox must be non-degenerate");
        ImplicitSurface {
            sdf,
            linear: IDENTITY,
            offset: [0.0; 3],
            inv_linear: IDENTITY,
            bbox,
            scale,
        }
    }

    /// Field value at a world point (negative inside).
    pub fn eval(&self, p: V3) -> f64 {
        let local = mat_vec(&self.inv_linear, sub(p, self.offset));
        self.sdf.eval(local)
    }

    /// World-frame gradient by central differences (uniform across every node
    /// kind, including the min/max kinks of booleans, where the one-sided
    /// derivative a smooth query lands on is the correct branch).
    pub fn grad(&self, p: V3) -> V3 {
        let h = self.scale * 1e-7;
        std::array::from_fn(|i| {
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            (self.eval(a) - self.eval(b)) / (2.0 * h)
        })
    }

    /// First-order signed distance to the zero set, `F/|∇F|` (exact for pure
    /// primitive SDFs, first-order for blends/placements).
    pub fn signed_offset(&self, p: V3) -> f64 {
        let g = len(self.grad(p));
        if g < 1e-300 {
            return self.eval(p);
        }
        self.eval(p) / g
    }

    /// Closest point on the surface and the outward normal there — the
    /// projection contract of the meshing path. Damped gradient Newton on
    /// `F(x) = 0`: converges quadratically near the surface, which is where
    /// every mesher query originates (points drift off by one smoothing step).
    pub fn closest(&self, p: V3) -> (V3, V3) {
        let tol = self.scale * 1e-12;
        let mut x = p;
        for _ in 0..48 {
            let f = self.eval(x);
            let g = self.grad(x);
            let g2 = dot(g, g);
            if g2 < 1e-300 {
                break;
            }
            let step = f / g2;
            let dx = scale(g, -step);
            x = add(x, dx);
            if len(dx) < tol {
                break;
            }
        }
        (x, normalize(self.grad(x)))
    }

    /// Local curvature radius at (the projection of) `p`, from the normal
    /// turn across two tangent probes — the sizing-field query. Mirrors the
    /// discrete carrier's estimate; the probe length is resolution-free.
    pub fn curvature_radius(&self, p: V3) -> f64 {
        let (x, n) = self.closest(p);
        let h = self.scale * 1e-4;
        let t1 = {
            let cand = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            normalize(cross(n, cand))
        };
        let t2 = cross(n, t1);
        let mut worst: f64 = f64::INFINITY;
        for t in [t1, t2] {
            let (_, n_probe) = self.closest(add(x, scale(t, h)));
            let turn = len(sub(n_probe, n));
            if turn > 1e-14 {
                worst = worst.min(h / turn);
            }
        }
        worst
    }

    /// Parameters `t ∈ (0, 1)` where the segment `a→b` crosses the zero set,
    /// ascending: sign-change sampling at a resolution tied to the query
    /// scale, refined by bisection. Candidate seeding for the crossing pulls
    /// (the caller re-verifies against `signed_offset`, mirroring the
    /// discrete carrier's contract).
    pub fn segment_hits(&self, a: V3, b: V3, out: &mut Vec<f64>) {
        let seg = len(sub(b, a));
        if seg == 0.0 {
            return;
        }
        // Enough samples to see features near the proxy resolution; capped so
        // a degenerate long segment stays cheap.
        let n = ((seg / self.scale) * 256.0).ceil().clamp(8.0, 256.0) as usize;
        let at = |t: f64| add(a, scale(sub(b, a), t));
        let mut t_prev = 0.0;
        let mut f_prev = self.eval(a);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let f = self.eval(at(t));
            if f_prev == 0.0 {
                out.push(t_prev);
            } else if f_prev * f < 0.0 {
                // Bisection to ~1e-12 of the segment.
                let (mut lo, mut hi) = (t_prev, t);
                let (mut flo, _) = (f_prev, f);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let fm = self.eval(at(mid));
                    if fm == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fm < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fm;
                    }
                }
                out.push(0.5 * (lo + hi));
            }
            t_prev = t;
            f_prev = f;
        }
        if f_prev == 0.0 {
            out.push(1.0);
        }
        out.retain(|&t| t > 0.0 && t < 1.0);
        out.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
    }

    /// The carrier under a placement `world = linear · local + offset`,
    /// composed onto any existing placement. The SDF tree itself is untouched;
    /// only the frame maps.
    pub fn transformed(&self, linear: [[f64; 3]; 3], offset: V3) -> ImplicitSurface {
        let new_linear = mat_mul(&linear, &self.linear);
        let new_offset = add(mat_vec(&linear, self.offset), offset);
        // The query scale follows the placement (uniform scaling is the only
        // scale-changing placement the scene layer produces).
        let corners = [self.bbox.0, self.bbox.1];
        let w0 = add(mat_vec(&new_linear, corners[0]), new_offset);
        let w1 = add(mat_vec(&new_linear, corners[1]), new_offset);
        ImplicitSurface {
            sdf: self.sdf.clone(),
            linear: new_linear,
            offset: new_offset,
            inv_linear: mat_inv(&new_linear),
            bbox: self.bbox,
            scale: len(sub(w1, w0)).max(1e-300),
        }
    }
}

// ======================================================================================
// Surface-nets tessellation — the pipeline entry proxy
// ======================================================================================

/// Tessellate the zero set into a watertight triangle proxy by naive surface
/// nets over a `cells³` grid on the carrier's bbox: one vertex per
/// sign-changing cell (mean of its edge crossings), one quad (two triangles)
/// per sign-changing grid edge, wound outward by the field sign. The proxy is
/// closed and manifold as long as the zero set stays strictly inside the bbox
/// and features are resolved by the grid.
pub fn tessellate_surface_nets(surf: &ImplicitSurface, cells: usize) -> (Vec<V3>, Vec<[u32; 3]>) {
    let n = cells.max(4);
    let (lo, hi) = surf.bbox;
    let d: V3 = std::array::from_fn(|i| (hi[i] - lo[i]) / n as f64);
    let np = n + 1;
    let pt = |i: usize, j: usize, k: usize| -> V3 {
        [lo[0] + d[0] * i as f64, lo[1] + d[1] * j as f64, lo[2] + d[2] * k as f64]
    };

    // Sample the field at grid points (local frame == world frame here: the
    // proxy is built at construction, before any placement).
    let idx = |i: usize, j: usize, k: usize| (i * np + j) * np + k;
    let mut field = vec![0.0f64; np * np * np];
    for i in 0..np {
        for j in 0..np {
            for k in 0..np {
                field[idx(i, j, k)] = surf.eval(pt(i, j, k));
            }
        }
    }

    // One vertex per cell that has a sign change on any of its 12 edges: the
    // mean of the edge crossings (linear interpolation along each edge).
    let cell_id = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    let mut cell_vert = vec![u32::MAX; n * n * n];
    let mut verts: Vec<V3> = Vec::new();
    const CELL_EDGES: [([usize; 3], [usize; 3]); 12] = [
        ([0, 0, 0], [1, 0, 0]), ([0, 1, 0], [1, 1, 0]), ([0, 0, 1], [1, 0, 1]), ([0, 1, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 1, 0]), ([1, 0, 0], [1, 1, 0]), ([0, 0, 1], [0, 1, 1]), ([1, 0, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 1]), ([0, 1, 0], [0, 1, 1]), ([1, 1, 0], [1, 1, 1]),
    ];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut sum = [0.0f64; 3];
                let mut cnt = 0usize;
                for (ea, eb) in CELL_EDGES {
                    let fa = field[idx(i + ea[0], j + ea[1], k + ea[2])];
                    let fb = field[idx(i + eb[0], j + eb[1], k + eb[2])];
                    if (fa < 0.0) != (fb < 0.0) {
                        let t = fa / (fa - fb);
                        let pa = pt(i + ea[0], j + ea[1], k + ea[2]);
                        let pb = pt(i + eb[0], j + eb[1], k + eb[2]);
                        for c in 0..3 {
                            sum[c] += pa[c] + t * (pb[c] - pa[c]);
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    cell_vert[cell_id(i, j, k)] = verts.len() as u32;
                    verts.push(std::array::from_fn(|c| sum[c] / cnt as f64));
                }
            }
        }
    }

    // One quad per sign-changing INTERIOR grid edge, spanned by the four
    // cells around the edge, wound so triangle normals point from inside
    // (field < 0) to outside.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut quad = |v: [u32; 3+1], flip: bool| {
        let [a, b, c, dd] = v;
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || dd == u32::MAX {
            return; // zero set touched the bbox shell — proxy stays open there
        }
        if flip {
            tris.push([a, c, b]);
            tris.push([a, dd, c]);
        } else {
            tris.push([a, b, c]);
            tris.push([a, c, dd]);
        }
    };
    for i in 0..np {
        for j in 0..np {
            for k in 0..np {
                let f0 = field[idx(i, j, k)];
                // x-edge (i,j,k)→(i+1,j,k): cells (i, j-1..j, k-1..k)
                if i < n && j > 0 && j < n && k > 0 && k < n {
                    let f1 = field[idx(i + 1, j, k)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i, j - 1, k - 1)],
                                cell_vert[cell_id(i, j, k - 1)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i, j - 1, k)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
                // y-edge (i,j,k)→(i,j+1,k): cells (i-1..i, j, k-1..k)
                if j < n && i > 0 && i < n && k > 0 && k < n {
                    let f1 = field[idx(i, j + 1, k)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i - 1, j, k - 1)],
                                cell_vert[cell_id(i - 1, j, k)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i, j, k - 1)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
                // z-edge (i,j,k)→(i,j,k+1): cells (i-1..i, j-1..j, k)
                if k < n && i > 0 && i < n && j > 0 && j < n {
                    let f1 = field[idx(i, j, k + 1)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i - 1, j - 1, k)],
                                cell_vert[cell_id(i, j - 1, k)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i - 1, j, k)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
            }
        }
    }
    (verts, tris)
}

/// Build a [`Faceted`] solid from an implicit surface: the surface-nets proxy
/// at `cells` resolution, every facet tagged with the ONE implicit carrier —
/// the exact mirror of the STL import path, with an analytic carrier.
pub fn implicit_solid(surf: ImplicitSurface, cells: usize) -> Result<Faceted, String> {
    let (verts, tris) = tessellate_surface_nets(&surf, cells);
    if tris.is_empty() {
        return Err("implicit solid: zero set not found inside the bbox".to_string());
    }
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Implicit(std::sync::Arc::new(surf)));
    for t in &tris {
        let tri = Tri::new(
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        f.push_tri(tri, s);
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(r: f64) -> ImplicitSurface {
        ImplicitSurface::new(
            Sdf::Sphere { center: [0.0; 3], radius: r },
            ([-r - 0.5; 3], [r + 0.5; 3]),
        )
    }

    #[test]
    fn closest_projects_onto_sphere() {
        let s = sphere(1.0);
        let (x, n) = s.closest([2.0, 0.3, -0.4]);
        assert!((len(x) - 1.0).abs() < 1e-9, "|x| = {}", len(x));
        let out = normalize(x);
        assert!(dot(n, out) > 0.999, "normal must point outward");
    }

    #[test]
    fn curvature_radius_of_sphere_is_radius() {
        let s = sphere(2.0);
        let r = s.curvature_radius([2.5, 0.0, 0.0]);
        assert!((r - 2.0).abs() < 0.05, "r = {r}");
    }

    #[test]
    fn segment_hits_finds_both_crossings() {
        let s = sphere(1.0);
        let mut hits = Vec::new();
        s.segment_hits([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], &mut hits);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0] - 0.25).abs() < 1e-9 && (hits[1] - 0.75).abs() < 1e-9, "{hits:?}");
    }

    #[test]
    fn surface_nets_proxy_is_closed_and_oriented() {
        let s = sphere(1.0);
        let (verts, tris) = tessellate_surface_nets(&s, 24);
        assert!(!tris.is_empty());
        // Closed: every directed edge has its opposite exactly once.
        let mut edges = std::collections::HashMap::new();
        for t in &tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry(e).or_insert(0i32) += 1;
            }
        }
        for (&(a, b), &c) in &edges {
            assert_eq!(c, 1, "duplicate directed edge {a}-{b}");
            assert_eq!(edges.get(&(b, a)), Some(&1), "unmatched edge {a}-{b}");
        }
        // Oriented outward: signed volume positive and near the sphere volume.
        let mut vol6 = 0.0;
        for t in &tris {
            let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
            vol6 += dot(a, cross(b, c));
        }
        let vol = vol6 / 6.0;
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            (vol - exact).abs() < 0.05 * exact,
            "signed volume {vol} vs sphere {exact}"
        );
    }

    #[test]
    fn transformed_carrier_queries_in_world_frame() {
        let s = sphere(1.0);
        // Uniform scale by 2 and shift: the surface is |p - c| = 2 around c.
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let c = [5.0, -1.0, 3.0];
        let w = s.transformed(m, c);
        let (x, _) = w.closest(add(c, [3.5, 0.0, 0.0]));
        assert!((len(sub(x, c)) - 2.0).abs() < 1e-8, "{:?}", sub(x, c));
        assert!(w.signed_offset(add(c, [2.0, 0.0, 0.0])).abs() < 1e-6);
    }
}
