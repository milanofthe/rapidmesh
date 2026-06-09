//! rapidmesh facade: builder API and mesh export.
//!
//! Public entry point tying the pipeline together: primitives and booleans
//! (rapidmesh-geom / rapidmesh-csg) into a tagged PLC, tet meshing
//! (rapidmesh-tet), and export of the resulting volume mesh (rapidfem format,
//! .msh/.vtk for inspection).

pub use rapidmesh_geom::TaggedPlc;
