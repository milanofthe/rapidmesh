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
use rapidmesh_exact::Point3;
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
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// A deterministic orthonormal in-plane basis `(e1, e2)` for the plane with unit
/// normal `n`: cross `n` with whichever world axis is least aligned with it
/// (stable, never near-degenerate). For an axis-aligned plane the basis is the
/// two world axes in the plane, so a [`Point3::Pac`] built on `o, o+e1, o+e2`
/// rounds back to coordinates EXACTLY on that plane (the bit-exact planar-volume
/// guarantee). Used by [`Carrier::exact`].
fn plane_basis(n: V3) -> (V3, V3) {
    let a = n[0].abs();
    let b = n[1].abs();
    let c = n[2].abs();
    let axis = if a <= b && a <= c {
        [1.0, 0.0, 0.0]
    } else if b <= c {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut e1 = cross(n, axis);
    let l1 = norm(e1);
    e1 = [e1[0] / l1, e1[1] / l1, e1[2] / l1];
    let e2 = cross(n, e1); // already unit (n, e1 orthonormal)
    (e1, e2)
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

    /// The EXACT [`Point3`] for this site's current position on its carrier: a
    /// [`Point3::Lnc`] on a straight edge and a [`Point3::Pac`] on a plane (both
    /// stay exactly on the carrier, and are closed under Steiner splits, so the
    /// constrained tetrahedralization keeps the boundary watertight and planar
    /// region volumes bit-exact). Curved carriers and the volume interior are
    /// explicit f64 (their exactness is tolerance-based, \cref{sec:conformity}).
    pub fn exact(&self) -> Point3 {
        self.carrier.exact(self.pos)
    }
}

impl Carrier {
    /// The exact [`Point3`] at f64 position `pos` on this carrier (see
    /// [`Site::exact`]). `pos` is assumed already on the carrier (the relaxation
    /// keeps it there); the construction only chooses the exact representation.
    pub fn exact(&self, pos: V3) -> Point3 {
        match self {
            Carrier::Vertex | Carrier::Volume | Carrier::Surface(_) | Carrier::CurvedEdge { .. } => {
                Point3::Explicit(pos)
            }
            Carrier::Edge { a, b } => {
                let ab = sub(*b, *a);
                let t = dot(sub(pos, *a), ab) / dot(ab, ab);
                Point3::Lnc { a: *a, b: *b, t }
            }
            Carrier::Plane { p0, n } => {
                let (e1, e2) = plane_basis(*n);
                let u = dot(sub(pos, *p0), e1);
                let v = dot(sub(pos, *p0), e2);
                Point3::Pac {
                    a: *p0,
                    b: add(*p0, e1),
                    c: add(*p0, e2),
                    u,
                    v,
                }
            }
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
    fn exact_plane_point_is_bit_exactly_on_axis_aligned_plane() {
        // A point relaxed on the plane z = 2 must reconstruct to a Pac whose f64
        // coordinates have z EXACTLY 2.0 (the bit-exact planar-volume guarantee).
        let s = Site::on_plane([5.0, -3.0, 2.0], [0.0, 0.0, 1.0], [0.37, 1.9, 2.0]);
        let e = s.exact();
        assert!(matches!(e, Point3::Pac { .. }));
        let p = e.approx().unwrap();
        assert_eq!(p[2], 2.0, "Pac z must be bit-exactly on the plane");
        assert!(dist(p, s.pos()) < 1e-12, "Pac must reconstruct the position");
    }

    #[test]
    fn exact_edge_point_lies_on_the_segment_line() {
        let s = Site::on_edge([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.3);
        let e = s.exact();
        match e {
            Point3::Lnc { a, b, t } => {
                assert_eq!(a, [0.0, 0.0, 0.0]);
                assert_eq!(b, [4.0, 0.0, 0.0]);
                assert!((t - 0.3).abs() < 1e-12);
            }
            _ => panic!("edge carrier must yield an Lnc"),
        }
        assert!(dist(e.approx().unwrap(), s.pos()) < 1e-12);
    }

    #[test]
    fn vertex_is_fixed() {
        let mut s = Site::vertex([1.0, 2.0, 3.0]);
        s.move_to([9.0, 9.0, 9.0]);
        assert!(dist(s.pos(), [1.0, 2.0, 3.0]) < 1e-12);
    }
}
