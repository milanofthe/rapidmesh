# rapidmesh-brep: a boundary-representation layer

## Why

The mesher consumes a **faceted** PLC (triangle soup + per-facet `SurfaceKind`
back-references). The faceting is an artifact of the CSG arrangement and of the
input tessellation (`n_seg`). Meshing it forces us to fight that faceting: blunt
trailing edges become near-coincident vertices, chart-group bijectivity breaks,
edge distributions inherit the input sampling instead of the geometry's curvature.

A **B-rep** (boundary representation) removes the artifact at the root: faces are
trimmed analytic surfaces, edges are analytic curves (including the new
intersection curves a boolean creates), vertices are points. The mesher then
re-meshes from the geometry — distribute on each edge curve, mesh each trimmed
face in its parameter space, fill the volume — independent of any input
tessellation. This is the "consume NURBS as general edges and surfaces" plan.

## SOTA (researched)

- **Half-edge / DCEL** (truck, Fornjot): compact, elegant, but **2-manifold only**
  — cannot represent a multi-material interface (an edge where 3+ materials meet)
  or an embedded sheet. rapidmesh's core domain is exactly multi-material.
- **Radial-edge (Weiler / "NMG")**: built for **non-manifold** geometry — an edge
  radially links ALL faces meeting along it; each face carries front/back material
  labels. This is the right topology for rapidmesh.
- **OpenCASCADE `TopoDS`** (industrial SOTA): clean **topology ⊕ geometry**
  separation; the key piece is the **PCurve** — the edge as a parametric curve in
  each adjacent face's (u,v) space, which is what makes meshing a trimmed face in
  parameter space well-posed.
- "Topology-First B-Rep Meshing" (arXiv): topology drives the mesh, geometry
  follows — matches our bottom-up approach.

Sources: ARL non-manifold/radial-edge notes; OpenCASCADE topology blog; Weiler
radial-edge; B-rep (Wikipedia: coedges/loops/parametric curves).

## The data structure (radial-edge + PCurves + lazy intersection curves)

```text
Geometry (analytic):
  Surface = SurfaceKind            (Plane/Cylinder/Sphere/Cone/Torus/Extruded/Nurbs)
  Curve   = Line | Circle | Ellipse | Nurbs | Intersection{a: SurfaceId, b: SurfaceId}
  PCurve  = the curve in a face's (u,v) parameter space   (for trim + 2D meshing)

Topology (non-manifold, Weiler radial-edge):
  Vertex   { point }                                  (exact Point3 at corners)
  Edge     { curve, ends:[VertexId;2], t:[f64;2], radial: Vec<HalfEdgeId> }
  HalfEdge { edge, dir, pcurve, loop, twin/radial-next }
  Loop     { coedges: Vec<HalfEdgeId> }               (outer + holes, oriented)
  Face     { surface, loops, regions:[RegionTag;2] }  (front/back material)
  Shell    { faces }
  Region   { shells, tag }                            (one material volume)
  Brep     { verts, edges, faces, loops, halfedges, shells, regions, surfaces }
```

### Three decisions that keep it elegant

1. **Lazy `Curve::Intersection{a,b}`** — an intersection edge is just "surface A
   meets surface B"; the mesher projects onto A∩B on demand (alternating
   projection / Newton, reusing `project::closest_on_surface`). For
   `Extruded ∩ Plane` it reduces to the profile curve. Avoids a heavy general
   closed-form intersection-curve subsystem.
2. **PCurve per face** — a trimmed face is meshed in its own (u,v) space (2D Lloyd,
   trim loops as boundary), then lifted. This is what fixes the airfoil
   leading/trailing-edge conformity: the face is trimmed by construction, no
   restricted-Delaunay inference.
3. **Built FROM the exact CSG arrangement** — the arrangement gives the exact
   TOPOLOGY (which surfaces meet, region labels, vertex positions); the B-rep adds
   the analytic edge curves + trim loops on top. The exact CSG stays the source of
   truth, so exactness is preserved.

## How the mesher consumes it (the existing bottom-up stages)

- Vertices pinned -> Edges via `curve::distribute` on the analytic `Curve`
  (NURBS-native, tessellation-independent) -> Faces meshed in (u,v), trimmed by
  the loops, lifted -> Volume via the `SizeField` hierarchy + Delaunay. The
  curve/sizefield foundations and the volume/region/extraction stages are reused;
  only the edge/face GEOMETRY source changes from faceted to analytic.

## Build plan (incremental)

1. Crate skeleton: geometry + topology types (this commit).
2. Builder: `TaggedPlc` (+ arrangement) -> `Brep` (recover analytic edges from
   adjacent surfaces; trim loops from feature-edge chains; region labels).
3. Trimmed-face mesher: replace the chart-group path (fixes airfoil LE/TE).
4. Wire `mesh_plc_with` to mesh the `Brep`; reuse the volume/region/extraction.
5. Re-enable the embedded-sheet fixtures (sheets = faces with equal front/back
   region).
