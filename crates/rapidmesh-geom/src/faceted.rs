//! Faceted shapes: tessellated triangle meshes with analytic surface
//! back-references.

use rapidmesh_csg::{Solid, Tri};

/// The analytic surface a facet was tessellated from. Flat facets need no
/// snapping; curved kinds carry the data the order-2 midside snapping stage
/// projects onto. Metadata only — exactness of the mesh never depends on it.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceKind {
    /// A flat face (the triangle itself is the exact surface).
    Plane,
    /// An infinite cylinder barrel.
    Cylinder {
        /// A point on the axis.
        center: [f64; 3],
        /// Axis direction (not necessarily unit).
        axis: [f64; 3],
        /// Barrel radius.
        radius: f64,
    },
    /// A sphere.
    Sphere {
        /// Center.
        center: [f64; 3],
        /// Radius.
        radius: f64,
    },
    /// A cone barrel (frustum side with distinct radii).
    Cone {
        /// Apex point.
        apex: [f64; 3],
        /// Axis direction from apex into the cone (not necessarily unit).
        axis: [f64; 3],
        /// Tangent of the half-opening angle.
        tan_half_angle: f64,
    },
}

/// A tessellated shape: triangles plus, per triangle, the analytic surface
/// it approximates. Used for both closed solids and open sheets; closedness
/// and orientation are invariants of the builder that produced it.
#[derive(Debug, Clone)]
pub struct Faceted {
    /// The triangles.
    pub tris: Vec<Tri>,
    /// Per-triangle index into `surfaces`.
    pub face_surface: Vec<u32>,
    /// The distinct analytic surfaces of this shape.
    pub surfaces: Vec<SurfaceKind>,
}

impl Faceted {
    /// Empty shape.
    pub fn new() -> Faceted {
        Faceted {
            tris: Vec::new(),
            face_surface: Vec::new(),
            surfaces: Vec::new(),
        }
    }

    /// Registers a surface and returns its index.
    pub fn add_surface(&mut self, s: SurfaceKind) -> u32 {
        self.surfaces.push(s);
        (self.surfaces.len() - 1) as u32
    }

    /// Adds a triangle on the given surface.
    pub fn push_tri(&mut self, t: Tri, surface: u32) {
        self.tris.push(t);
        self.face_surface.push(surface);
    }

    /// Appends another shape (surface indices are re-based).
    pub fn append(&mut self, other: &Faceted) {
        let base = self.surfaces.len() as u32;
        self.surfaces.extend(other.surfaces.iter().cloned());
        self.tris.extend(other.tris.iter().copied());
        self.face_surface
            .extend(other.face_surface.iter().map(|&s| s + base));
    }

    /// The bare triangle soup as a CSG solid operand. Only meaningful for
    /// shapes built as closed, outward-oriented solids.
    pub fn to_solid(&self) -> Solid {
        Solid {
            tris: self.tris.clone(),
        }
    }

    /// Rigidly transformed copy (rotation/reflection-free linear part keeps
    /// the surface metadata valid; non-rigid linear parts would invalidate
    /// radii in `surfaces`).
    pub fn transformed(&self, linear: [[f64; 3]; 3], offset: [f64; 3]) -> Faceted {
        let map = |p: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|i| {
                linear[i][0] * p[0] + linear[i][1] * p[1] + linear[i][2] * p[2] + offset[i]
            })
        };
        let map_dir = |d: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|i| linear[i][0] * d[0] + linear[i][1] * d[1] + linear[i][2] * d[2])
        };
        Faceted {
            tris: self
                .tris
                .iter()
                .map(|t| Tri::new(map(t.v[0]), map(t.v[1]), map(t.v[2])))
                .collect(),
            face_surface: self.face_surface.clone(),
            surfaces: self
                .surfaces
                .iter()
                .map(|s| match s {
                    SurfaceKind::Plane => SurfaceKind::Plane,
                    SurfaceKind::Cylinder {
                        center,
                        axis,
                        radius,
                    } => SurfaceKind::Cylinder {
                        center: map(*center),
                        axis: map_dir(*axis),
                        radius: *radius,
                    },
                    SurfaceKind::Sphere { center, radius } => SurfaceKind::Sphere {
                        center: map(*center),
                        radius: *radius,
                    },
                    SurfaceKind::Cone {
                        apex,
                        axis,
                        tan_half_angle,
                    } => SurfaceKind::Cone {
                        apex: map(*apex),
                        axis: map_dir(*axis),
                        tan_half_angle: *tan_half_angle,
                    },
                })
                .collect(),
        }
    }

    /// Translated copy.
    pub fn translated(&self, offset: [f64; 3]) -> Faceted {
        self.transformed([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], offset)
    }

    /// Copy rotated by `angle` radians around the axis `(origin, dir)`
    /// (Rodrigues formula, right-handed).
    pub fn rotated(&self, origin: [f64; 3], dir: [f64; 3], angle: f64) -> Faceted {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!(len > 0.0, "rotation axis must be nonzero");
        let u = [dir[0] / len, dir[1] / len, dir[2] / len];
        let (s, c) = angle.sin_cos();
        let mut rot = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let kron = if i == j { 1.0 } else { 0.0 };
                // Levi-Civita term of the cross-product matrix.
                let eps = match (i, j) {
                    (0, 1) => -u[2],
                    (1, 0) => u[2],
                    (0, 2) => u[1],
                    (2, 0) => -u[1],
                    (1, 2) => -u[0],
                    (2, 1) => u[0],
                    _ => 0.0,
                };
                rot[i][j] = c * kron + s * eps + (1.0 - c) * u[i] * u[j];
            }
        }
        // Rotate about the origin point: x -> R (x - o) + o.
        let offset = std::array::from_fn(|i| {
            origin[i] - (rot[i][0] * origin[0] + rot[i][1] * origin[1] + rot[i][2] * origin[2])
        });
        self.transformed(rot, offset)
    }
}

impl Default for Faceted {
    fn default() -> Self {
        Faceted::new()
    }
}
