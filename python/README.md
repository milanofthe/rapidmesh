# rapidmesh

Conforming tetrahedral and surface meshes for finite element and finite
volume solvers. The mesher is written in Rust; this package is a thin layer over its
API, so everything here is also available to Rust programs.

- **Primitives, CAD files and exact CSG**: boxes, cylinders, spheres, cones,
  tori, prisms, sweeps, helices, lofts, fillets and chamfers, sheets, STEP
  files and STL/OBJ imports, united and cut by an exact arrangement.
  Material interfaces stay exactly conforming.
- **Bottom-up meshing**: edges, then each face on its analytic surface,
  then each region on its own, with hierarchical sizing per region,
  geometric face and geometric edge. Thin layers take flat tets instead of
  being refined to their thickness.
- **Solver output**: topology with orientation signs, the geometric entity
  of every face and edge, named sets, periodic point pairs, gmsh MSH 4.1,
  VTU, OpenFOAM polyMesh with tets or polyhedral cells
  (`mesh.write_foam(case, polyhedral=True)`), CalculiX/Abaqus
  (`mesh.write_inp(path, order=2)`), second-order tets on the true geometry
  (`mesh.second_order()`), finite volume quality (`mesh.fvm_quality()`).

## Install

```bash
pip install rapidmesh
```

## Usage

```python
import rapidmesh as rm

g = rm.Geometry(maxh=0.5)
air = g.box(10, 10, 13.6)
sub = g.box(10, 10, 1.6, maxh=0.8)
g.label(air, "air")
g.label(sub, "substrate")
g.xy_plate(6, 6, position=(2, 2, 1.6), tag=1)   # a PEC patch
g.label(1, "patch")

# a periodic unit cell, and a named port face
g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))
g.periodic(g.surf(normal=(0, -1, 0)), g.surf(normal=(0, 1, 0)))
g.surf(near=(5, 5, 13.6)).name = "port"

mesh = g.mesh()
print(mesh)                 # Mesh(... tets, ... points, min dihedral ... deg, ... ms)

mesh.points                 # (n_points, 3) float64
mesh.tets                   # (n_tets, 4)   uint64
mesh.tet_regions            # region tag per tet
mesh.topology               # edges, faces, incidence with signs, classification
mesh.sets()                 # {"cells": ..., "faces": ..., "edges": ...} by name
mesh.periodic_points        # (n, 2) point on a master face and its image
mesh.write_msh("cell.msh")  # physical groups from the labels and names
```

### CAD files

```python
g = rm.Geometry(maxh=2.0)
bodies = g.import_step("assembly.step")   # one solid per body of the file
mesh = g.mesh()                           # faces meshed on their true surfaces
```

STEP files (AP203 and AP214) bring planes, cylinders, cones, spheres, tori
and B-spline surfaces; coordinates stay in the file's unit. Each solid takes
the name the file gives its part, so an assembly's mesh has its parts as
named sets and physical groups.

### Sizing

```python
g.maxh = 0.5                               # global target size
g.region(2).maxh = 0.2                     # one region
g.surf(normal=(0, 0, 1)).maxh = 0.1        # the faces facing +z
g.edge(near=(0, 0, 1)).maxh = 0.05         # the edge nearest a point
g.tol = 1e-3                               # chord tolerance of curved entities
g.refine_near_points([(1, 1, 1)], 0.02)    # point size sources
mesh = g.mesh(target_elements=50_000)      # an element budget
mesh = g.mesh(min_angle=15)                # refined where tets stay below 15 degrees
```

`maxh` is the size the mesh needs for what lives on it; curved geometry
takes either a chord tolerance (`g.tol`, about ten segments round a circle
by default) or a geometric error:

```python
mesh = g.mesh(geom_error=1e-2, order=2)    # volumes and sheet areas within 1 %
                                           # on quadratic elements
```

`geom_error` bounds the error of the volume of every region and the area of
every sheet, measured on flat elements (`order=1`) or on the quadratic
elements of the second-order mesh (`order=2`). Quadratic elements follow a
curve with far fewer of them: on CAD parts read from STEP, one percent at
`order=2` takes about as many tets as the default tolerance or fewer, with a
measured error of a few hundredths of a percent.

### Second order

```python
so = mesh.second_order()       # tet10: points, tets (n, 10), faces (m, 6), volumes
so["curved_tets"]              # per tet: a mid-edge node off its chord
mesh.write_msh("part.msh", order=2)
mesh.write_inp("part.inp", order=2)
mesh.show(second_order=True)   # curved faces drawn curved
```

Every edge takes a node in its middle, on the true geometry where the edge
lies on a curved surface or a curve. Tets with no such node keep their
mid-edge nodes in the middle of their edges (an affine map, `curved_tets`
false).

### Surface meshes

`g.surface_mesh()` meshes only the surfaces (interfaces, outer boundary,
sheets) and gives its edge topology and the same sets and file output.
`rm.polygon_union` merges overlapping layout polygons into outlines for
sheets and prisms.

### What happened, how long, and where the quality is worst

```python
print(mesh.report())     # stage timings, quality with location, warnings
mesh.timings             # seconds per stage
mesh.metrics             # predicate counts, point and tet counts
mesh.quality             # min_dihedral_deg, worst_location, worst_region, regions
mesh.diagnostics         # histogram, watertightness, fidelity, located defects
mesh.log                 # [{level, stage, message, at}, ...]
```

Set `RAPIDMESH_LOG=1` to stream the log to stderr while meshing.

A geometry the mesher cannot mesh raises `rm.MeshError` (a `ValueError`)
whose message says where and what to repair: features far below the mesh
size, gaps between solids, degenerate faces.

## License

Dual licensed: GNU Affero General Public License v3.0, or a commercial
license for use in proprietary products and services without the AGPL
obligations (contact [milanrother.com/consulting](https://milanrother.com/consulting/)).
Versions up to 0.6.0 were published under the MIT license.
