//! Unified point abstraction for the mesher: a `Site` is an f64 position plus a
//! `Carrier` that knows how the point may move. Relaxation just calls
//! `move_to(target)`; the carrier re-projects onto itself, so a point stays on
//! its edge, its plane, or its analytic surface, and a free volume point stays
//! in the interior (the caller mirrors it back in if a move would leave).
//!
//! f64 throughout: the carrier governs only WHERE a point may go, not arithmetic
//! exactness. Conformity comes from the hierarchy (fixed lower-dimensional
//! boundaries) and surface oversampling, not from exact coordinates.

use crate::project;
use rapidmesh_geom::SurfaceKind;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// One side of a curved feature edge: the analytic geometry a point on the edge
/// also lies on. A feature edge is the intersection of TWO such faces.
#[derive(Clone, Debug)]
pub enum EdgeFace {
    /// A plane (point `p0`, unit normal `n`).
    Plane { p0: V3, n: V3 },
    /// An analytic surface.
    Surface(SurfaceKind),
}

impl EdgeFace {
    fn project(&self, p: V3) -> V3 {
        match self {
            EdgeFace::Plane { p0, n } => sub(p, scale(*n, dot(sub(p, *p0), *n))),
            EdgeFace::Surface(k) => project::closest_on_surface(k, p),
        }
    }
}

/// What a site is constrained to and how it moves.
#[derive(Clone, Debug)]
pub enum Carrier {
    /// Fixed PLC corner (never moves).
    Vertex,
    /// On the segment through `a`, `b` (moves along it, clamped to the segment).
    Edge { a: V3, b: V3 },
    /// On the analytic CURVE where two faces meet (a cylinder rim, an airfoil
    /// profile outline). Moves along it via alternating projection onto both
    /// faces; `ea`/`eb` are the bounding corners (skipped for a closed loop where
    /// they coincide). The geometry-derived 1D carrier: points are distributed by
    /// relaxation on the true curve, not frozen to the input tessellation.
    CurvedEdge { fa: EdgeFace, fb: EdgeFace, ea: V3, eb: V3 },
    /// On the plane (point `p0`, unit normal `n`); moves in the plane.
    Plane { p0: V3, n: V3 },
    /// On an analytic surface; moves on it via closest-point projection.
    Surface(SurfaceKind),
    /// Free in the volume.
    Volume,
}

/// A mesh point with its carrier.
#[derive(Clone, Debug)]
pub struct Site {
    pub carrier: Carrier,
    pos: V3,
}

impl Site {
    pub fn vertex(p: V3) -> Site {
        Site { carrier: Carrier::Vertex, pos: p }
    }
    pub fn free(p: V3) -> Site {
        Site { carrier: Carrier::Volume, pos: p }
    }
    /// A point on the segment `a`-`b` at parameter `t`.
    pub fn on_edge(a: V3, b: V3, t: f64) -> Site {
        Site { carrier: Carrier::Edge { a, b }, pos: add(a, scale(sub(b, a), t)) }
    }
    /// A point at `pos`, constrained to the plane (`p0`, unit `n`).
    pub fn on_plane(p0: V3, n: V3, pos: V3) -> Site {
        let mut s = Site { carrier: Carrier::Plane { p0, n }, pos };
        s.pos = s.project(pos);
        s
    }
    /// A point at `pos`, constrained to the analytic `surface`.
    pub fn on_surface(surface: SurfaceKind, pos: V3) -> Site {
        let mut s = Site { carrier: Carrier::Surface(surface), pos };
        s.pos = s.project(pos);
        s
    }
    /// A point on the curve where `fa` and `fb` meet, between corners `ea`/`eb`.
    pub fn on_curved_edge(fa: EdgeFace, fb: EdgeFace, ea: V3, eb: V3, pos: V3) -> Site {
        let mut s = Site { carrier: Carrier::CurvedEdge { fa, fb, ea, eb }, pos };
        s.pos = s.project(pos);
        s
    }

    pub fn pos(&self) -> V3 {
        self.pos
    }
    pub fn is_volume(&self) -> bool {
        matches!(self.carrier, Carrier::Volume)
    }

    /// The carrier's projection of an arbitrary target back onto itself.
    fn project(&self, tgt: V3) -> V3 {
        match &self.carrier {
            Carrier::Vertex => self.pos,
            Carrier::Volume => tgt,
            Carrier::Edge { a, b } => {
                let ab = sub(*b, *a);
                let t = (dot(sub(tgt, *a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
                add(*a, scale(ab, t))
            }
            Carrier::Plane { p0, n } => sub(tgt, scale(*n, dot(sub(tgt, *p0), *n))),
            Carrier::Surface(kind) => project::closest_on_surface(kind, tgt),
            Carrier::CurvedEdge { fa, fb, ea, eb } => {
                // Alternating projection (POCS) onto both faces converges to their
                // intersection curve; from a target near the curve it lands on the
                // nearest curve point.
                let mut q = tgt;
                for _ in 0..16 {
                    q = fa.project(q);
                    q = fb.project(q);
                }
                // Clamp inside the corner span on an OPEN arc (skip on a closed
                // loop where `ea == eb` and the chord parameter is degenerate).
                let ab = sub(*eb, *ea);
                let ab2 = dot(ab, ab);
                if ab2 > 1e-24 {
                    let t = dot(sub(q, *ea), ab) / ab2;
                    if t <= 0.0 {
                        return *ea;
                    } else if t >= 1.0 {
                        return *eb;
                    }
                }
                q
            }
        }
    }

    /// Move toward `tgt`, re-projected onto the carrier. `Vertex` never moves.
    pub fn move_to(&mut self, tgt: V3) {
        if !matches!(self.carrier, Carrier::Vertex) {
            self.pos = self.project(tgt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: V3, b: V3) -> f64 {
        dot(sub(a, b), sub(a, b)).sqrt()
    }

    #[test]
    fn edge_clamps_to_segment() {
        let mut s = Site::on_edge([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.25);
        assert!(dist(s.pos(), [0.5, 0.0, 0.0]) < 1e-12);
        s.move_to([1.0, 5.0, 5.0]);
        assert!(dist(s.pos(), [1.0, 0.0, 0.0]) < 1e-12);
        s.move_to([9.0, 0.0, 0.0]);
        assert!(dist(s.pos(), [2.0, 0.0, 0.0]) < 1e-12);
    }

    #[test]
    fn plane_keeps_point_in_plane() {
        let mut s = Site::on_plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.3, 0.3, 9.0]);
        assert!((s.pos()[2] - 2.0).abs() < 1e-12);
        s.move_to([0.7, 0.1, -4.0]);
        assert!((s.pos()[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn surface_keeps_point_on_sphere() {
        let k = SurfaceKind::Sphere { center: [0.0, 0.0, 0.0], radius: 2.0 };
        let mut s = Site::on_surface(k, [3.0, 4.0, 0.0]);
        assert!((dot(s.pos(), s.pos()).sqrt() - 2.0).abs() < 1e-12);
        s.move_to([0.0, 0.0, 9.0]);
        assert!((dot(s.pos(), s.pos()).sqrt() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn vertex_is_fixed() {
        let mut s = Site::vertex([1.0, 2.0, 3.0]);
        s.move_to([9.0, 9.0, 9.0]);
        assert!(dist(s.pos(), [1.0, 2.0, 3.0]) < 1e-12);
    }
}
