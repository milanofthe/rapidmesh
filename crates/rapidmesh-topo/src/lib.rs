//! Analysis-ready cell-complex view of a mesh.
//!
//! The solver-agnostic, dimension-uniform derivation of a mesh's 0/1/2/3-cell
//! incidence and per-element geometry — the connectivity downstream FEM/MoM
//! solvers otherwise rebuild from scratch. 2D and 3D run through the same code:
//! a triangle mesh's *topology* is identical whether it is planar (MoM) or
//! embedded in 3D (a surface); only *geometry* is coordinate-aware.
//!
//! This crate is basis-free. RWG / Nédélec DOF maps and quadrature layer on top.
//!
//! See `DESIGN.md` for the rationale, conventions, and roadmap.
//!
//! ```
//! use rapidmesh_topo::{TetTopology, Tets};
//! // one tet -> 6 edges, 4 faces, every face on the boundary
//! let topo = TetTopology::build(&Tets { tets: &[[0, 1, 2, 3]], n_verts: 4 });
//! assert_eq!(topo.edges.len(), 6);
//! assert_eq!(topo.faces.len(), 4);
//! ```

pub mod convention;
pub mod csr;
mod math;
mod source;
mod tet;
mod tri;
pub mod wire;

#[cfg(feature = "mesher")]
pub mod mesher;

pub use convention::{
    canonical_edge, sort3_sign, NONE, TET_EDGE_LOCAL, TET_FACE_LOCAL, TRI_EDGE_LOCAL,
};
pub use csr::Csr;
pub use source::{TetSource, TriSource, Tets, Tris};
pub use tet::{TetGeometry, TetTopology};
pub use tri::{TriGeometry, TriTopology};
pub use wire::{FrameReader, FrameWriter};
