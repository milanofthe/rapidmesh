# rapidmesh

Pure-Rust conforming **tetrahedral mesh generator** for 3-D electromagnetic FEM
(Maxwell, H(curl)/Nédélec), with a Python builder API.

- **Primitives + exact CSG booleans** — boxes, cylinders, spheres, cones, tori,
  prisms, sweeps, lofts; unioned/subtracted by an exact mesh arrangement (no
  float snapping, material interfaces stay exactly conforming).
- **Constrained Delaunay tetrahedralization** with exact boundary recovery.
- **Sizing-field-driven refinement** and dihedral-angle-targeted optimization.
- **Observability** — per-stage timings, statistics, a leveled log, and quality
  with location, all from Python.

## Install

```bash
pip install rapidmesh
```

## Usage

```python
import rapidmesh as rm

g = rm.Geometry(maxh=0.4)
g.box(4, 4, 2)                                   # air / substrate box
g.cylinder(radius=0.8, height=2, position=(2, 2, 0), void=True)  # a bore

mesh = g.mesh()
print(mesh)                       # Mesh(... tets, ... points, min dihedral ... deg)

mesh.points        # (n_points, 3) float64
mesh.tets          # (n_tets, 4)  uint64
mesh.tet_regions   # (n_tets,)    region tag per tet
mesh.faces         # (n_faces, 3) surface faces (region interfaces, sheets)
```

### What happened, how long, and where the quality is worst

```python
print(mesh.report())     # per-stage timings + quality-with-location + warnings
mesh.timings             # {"mesh.faces": 0.42, "mesh.refine": 0.16, ...} seconds
mesh.metrics             # predicate counts, recovery work, point/tet counts
mesh.quality             # min_dihedral_deg, worst_location, worst_region, regions[]
mesh.log                 # [{level, stage, message, at}, ...]
mesh.warnings            # the warn/error subset (budget caps, slivers)
```

Set `RAPIDMESH_LOG=1` in the environment to stream the log live to stderr
(including per-refinement-round progress, so you can see what is running and
where it spends or hangs time).

## License

MIT
