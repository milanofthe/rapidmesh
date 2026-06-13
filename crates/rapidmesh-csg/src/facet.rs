//! Planar polygon facets: flat faces carried as un-triangulated boundary
//! loops, so the arrangement can drop intersection curves on them and the
//! triangulation happens once, conformally, afterwards.
//!
//! This is the unit that makes conformal tessellation possible: a flat face
//! (a frustum cap, a box side, a sheet) has no interior vertices of its own,
//! so a curve piercing it lands exactly on the piercing surface's vertices
//! instead of creating near-twins (see docs/conformal-tessellation-plan.md).

use crate::tri::Tri;
use rapidmesh_exact::{orient2d, Axis, Point3, Sign};

/// A flat face as oriented boundary loops in a common plane: one outer loop
/// (CCW seen from the +normal side) and zero or more hole loops (CW). All
/// vertices are explicit f64 and lie in the facet's plane; the polygon is
/// simple (non-self-intersecting) and non-degenerate (positive outer area).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarFacet {
    /// Outer boundary, ordered, at least 3 vertices.
    pub outer: Vec<[f64; 3]>,
    /// Hole boundaries (each at least 3 vertices), empty for simple faces.
    pub holes: Vec<Vec<[f64; 3]>>,
}

impl PlanarFacet {
    /// New simple (hole-free) facet from an ordered outer loop.
    pub fn new(outer: Vec<[f64; 3]>) -> PlanarFacet {
        assert!(outer.len() >= 3, "planar facet needs at least 3 vertices");
        PlanarFacet { outer, holes: Vec::new() }
    }

    /// New facet with holes.
    pub fn with_holes(outer: Vec<[f64; 3]>, holes: Vec<Vec<[f64; 3]>>) -> PlanarFacet {
        assert!(outer.len() >= 3, "planar facet needs at least 3 vertices");
        for h in &holes {
            assert!(h.len() >= 3, "hole loop needs at least 3 vertices");
        }
        PlanarFacet { outer, holes }
    }

    /// Outer vertex as a [`Point3`].
    pub fn point(&self, i: usize) -> Point3 {
        Point3::Explicit(self.outer[i])
    }

    /// Axis-aligned bounding box over all loops.
    pub fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for loops in std::iter::once(&self.outer).chain(self.holes.iter()) {
            for v in loops {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        (lo, hi)
    }

    /// Approximate (f64) outward normal of the plane, from the outer loop's
    /// Newell area vector. Not unit length; sign follows the loop winding.
    pub fn normal(&self) -> [f64; 3] {
        let p = &self.outer;
        let n = p.len();
        let mut nx = 0.0;
        let mut ny = 0.0;
        let mut nz = 0.0;
        for i in 0..n {
            let a = p[i];
            let b = p[(i + 1) % n];
            nx += (a[1] - b[1]) * (a[2] + b[2]);
            ny += (a[2] - b[2]) * (a[0] + b[0]);
            nz += (a[0] - b[0]) * (a[1] + b[1]);
        }
        [nx, ny, nz]
    }

    /// A projection axis in which the outer loop has exactly nonzero area,
    /// with its 2D orientation there. Tries the axis of the largest normal
    /// component first (best-conditioned); the exactness comes from the
    /// orient2d on the first three non-collinear outer vertices. Panics on a
    /// fully degenerate (collinear) outer loop.
    pub fn projection_axis(&self) -> (Axis, Sign) {
        let n = self.normal();
        let mut axes = [Axis::X, Axis::Y, Axis::Z];
        axes.sort_by(|p, q| {
            n[q.index()]
                .abs()
                .partial_cmp(&n[p.index()].abs())
                .expect("finite coordinates")
        });
        for axis in axes {
            // Find any exactly non-collinear consecutive triple in projection.
            let m = self.outer.len();
            for i in 0..m {
                let s = orient2d(
                    &self.point(i),
                    &self.point((i + 1) % m),
                    &self.point((i + 2) % m),
                    axis,
                )
                .expect("explicit points are always valid");
                if s != Sign::Zero {
                    return (axis, s);
                }
            }
        }
        panic!("degenerate (collinear) planar facet: {:?}", self.outer);
    }

    /// True if the (hole-free) outer loop is convex in its projection: no turn
    /// is reflex (opposite the loop orientation). Collinear turns are allowed.
    /// A holed facet is never convex (the hole forbids a single fan).
    pub fn is_convex(&self) -> bool {
        if !self.holes.is_empty() {
            return false;
        }
        let (axis, orientation) = self.projection_axis();
        let n = self.outer.len();
        let reflex = orientation.flip();
        for i in 0..n {
            let s = orient2d(
                &self.point(i),
                &self.point((i + 1) % n),
                &self.point((i + 2) % n),
                axis,
            )
            .expect("explicit points are always valid");
            if s == reflex {
                return false;
            }
        }
        true
    }

    /// A copy with every loop vertex mapped by `f` (a rigid/affine map keeps
    /// the facet planar; the caller is responsible for that).
    pub fn map_points(&self, f: impl Fn([f64; 3]) -> [f64; 3]) -> PlanarFacet {
        PlanarFacet {
            outer: self.outer.iter().map(|&p| f(p)).collect(),
            holes: self
                .holes
                .iter()
                .map(|h| h.iter().map(|&p| f(p)).collect())
                .collect(),
        }
    }

    /// A copy with every loop reversed (winding flipped), as needed after an
    /// orientation-reversing map (mirror, negative-determinant scale).
    pub fn reversed(&self) -> PlanarFacet {
        PlanarFacet {
            outer: self.outer.iter().rev().copied().collect(),
            holes: self
                .holes
                .iter()
                .map(|h| h.iter().rev().copied().collect())
                .collect(),
        }
    }

    /// Triangulates the outer loop into a triangle fan (for the arrangement
    /// broadphase and for shapes that still need a triangle soup, e.g. a
    /// solid operand). Holes are NOT represented here — this is the convex/
    /// star-shaped hull fan, used only where holes are handled separately.
    /// The conformal path never fans; it triangulates from boundary +
    /// constraints in triangulate_facet.
    pub fn fan_tris(&self) -> Vec<Tri> {
        let p = &self.outer;
        (1..p.len() - 1)
            .map(|i| Tri::new(p[0], p[i], p[i + 1]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square_z() -> PlanarFacet {
        // CCW in the z=0 plane, outward normal +z.
        PlanarFacet::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    }

    #[test]
    fn bbox_covers_outer_and_holes() {
        let f = PlanarFacet::with_holes(
            vec![[0.0, 0.0, 2.0], [4.0, 0.0, 2.0], [4.0, 3.0, 2.0], [0.0, 3.0, 2.0]],
            vec![vec![[1.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 2.0]]],
        );
        let (lo, hi) = f.bbox();
        assert_eq!(lo, [0.0, 0.0, 2.0]);
        assert_eq!(hi, [4.0, 3.0, 2.0]);
    }

    #[test]
    fn normal_points_along_plane() {
        let n = unit_square_z().normal();
        // Newell normal of a CCW z=0 loop points +z.
        assert!(n[2] > 0.0, "normal {n:?} should point +z");
        assert_eq!(n[0], 0.0);
        assert_eq!(n[1], 0.0);
    }

    #[test]
    fn projection_axis_is_z_for_z_plane() {
        let (axis, sign) = unit_square_z().projection_axis();
        assert_eq!(axis, Axis::Z);
        assert_eq!(sign, Sign::Positive);
    }

    #[test]
    fn projection_skips_collinear_leading_triple() {
        // First three vertices collinear; the scan must find a later triple.
        let f = PlanarFacet::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let (axis, _) = f.projection_axis();
        assert_eq!(axis, Axis::Z);
    }

    #[test]
    fn fan_covers_outer() {
        let tris = unit_square_z().fan_tris();
        assert_eq!(tris.len(), 2); // quad -> 2 triangles
    }

    #[test]
    #[should_panic]
    fn degenerate_collinear_facet_panics() {
        PlanarFacet::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            .projection_axis();
    }
}
