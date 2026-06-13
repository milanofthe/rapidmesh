# Conformal tessellation at intersection curves

Work package to remove seam micro-features at their source: the uncoordinated
tessellation of primitives that meet on a shared intersection curve.

## 1. Problem and where it sits

rapidmesh has three independent paper foundations (DESIGN.md):

1. **CSG kernel** (Lévy 2024, Cherchi/Attene 2022): tessellate each primitive,
   then exactly arrange (intersect) the triangle soups → a **tagged PLC**.
2. **Tet mesher** (Diazzi/Attene 2023): tagged PLC → conforming CDT.
3. **Quality pass** (HXT toolkit): sliver removal.

The Diazzi CDT (step 2) assumes an *"exactly arranged watertight tagged PLC as
input"* (docs/cdt-recovery-plan.md). It guarantees conforming validity and
bit-exact region volumes, **not** element quality on a bad PLC: a thin input
facet maps to a conforming sliver. So the CDT paper is not where the seam
problem lives — it lives one step earlier, in the **PLC generation** (step 1),
which is our own code, not covered by any of the three papers.

### Root cause (measured)

Each primitive tessellates *independently and before* the boolean is known
(prim.rs: a cylinder barrel is N axial segments, a cap is a fan around its
centre). When two surfaces meet on a shared curve, they discretize that curve
*separately*. The exact arrangement then produces intersection points that
land ~1e-4·diag next to the other surface's tessellation vertices → thin
fragments → slivers and Steiner-split overproduction.

Concrete (counterbore_plate, the diagnostic in examples/diag_dense.rs): the
step floor is the counterbore void's planar cap (a fan, radius 0.26, with
vertices only on its centre and outer ring). The narrow bore (cylinder, 14
segments, radius 0.13) pierces it on a circle. The cap fan has **no vertices on
that circle**; the arrangement cuts the fan triangles against the bore wall and
creates intersection points on the circle *between* the 14 bore vertices →
254 of 309 short PLC-PLC edges are "divergent" (endpoints on different
surfaces), 2.7e-4..1.4e-2 apart. These are exactly the slivers (min-dih 0.0)
and the bulk of the 33408 segment splits (the perf wall: 2140 PLC points →
35548).

The same pattern, different geometry, at sweep-segment kinks (serpentine) and
stacked-box coplanar boundaries (laminate). It is one root cause with several
geometric faces.

### Why downstream fixes are partial (already tried)

- **Coarsening edge-collapse** (committed): removes the Steiner *swarm* around
  seams (97% of short edges), but cannot touch the divergent PLC-PLC edges —
  their endpoints sit on different surface sets, so the patch-preserving gate
  correctly forbids the collapse.
- **Tolerance welding** (RAPIDMESH_WELD_TOL experiment): snaps near twins, but
  only a narrow window works (1e-4..3e-4: counterbore 0.6→2.7), helps no other
  model, and crashes the T-junction repair above 3e-4. A fragile patch.
- **Envelope sliver removal** (fTetWild): explicitly disqualified by DESIGN.md
  — the ε-envelope smears material interfaces, a hard Maxwell-FEM no.

The information "these two near points belong to the same curve" is lost by the
time we mesh. The fix has to keep it — i.e. act in step 1.

## 2. The asset we already have

Primitives carry their **analytic surface** (`SurfaceKind`: Plane, Cylinder,
Cone, Sphere, Torus) per facet (`Faceted.face_surface` → `Faceted.surfaces`).
So we can compute intersection *curves* analytically, and we already know which
surface every facet belongs to. The exact arrangement also already finds the
intersections — just *after* both sides are finely tessellated.

The goal: every surface that meets a shared curve uses the **same** vertex
chain on that curve.

## 3. Two architectural levels

### Level A — late-triangulated flat faces (covers the dominant case)

The dominant seam (cap × barrel, the counterbore type) comes from flat faces
(caps, box sides, sheets) being **fan-triangulated up front**, planting interior
vertices that collide with the cuts. Fix: do not pre-triangulate a flat face.
Carry it as a **planar polygon facet** (its boundary loop) and triangulate it
**once, inside the arrangement**, with every intersection curve crossing it as
a constraint. A flat face then has no interior vertices of its own — the only
vertices on a piercing circle are the piercing surface's, so there are no near
twins.

This is the larger structural change, because the arrangement and
triangulate_facet are triangle-based today (Tri-Tri intersection, per-triangle
retriangulation). Level A needs a **planar-facet** abstraction: a face is a
plane + boundary loop + the set of constraint segments the arrangement drops on
it, triangulated as a unit (triangulate_facet already does constrained
triangulation of one facet from points+constraints — it generalizes from "one
input triangle" to "one planar polygon").

Scope of Level A:
- prim.rs: emit flat faces (frustum caps, box sides, prism/sheet faces) as
  polygon facets, not fans. Curved barrels stay tessellated.
- arrange.rs / scene.rs: arrangement candidate units become planar facets
  (grouped by coplanar surface id) instead of raw triangles; intersection
  segments accumulate per facet; classification representative is per polygon.
- triangulate.rs: triangulate_facet consumes a polygon boundary + constraints
  (it is close already; the boundary becomes the outer constraint loop).

This alone removes the counterbore/perforated-plate/box-stack class
(flat-face seams), which is the majority of the corpus's slivers and the perf
overproduction.

### Level B — coordinated curved tessellation (the remainder)

Where two *curved* surfaces meet (barrel × barrel, barrel × sphere, sweep
kinks), both sides are tessellated and neither is a polygon. Here we need the
genuine CAD-kernel step: compute the analytic intersection curve
(cylinder∩cylinder = a quartic space curve, cylinder∩plane = circle/ellipse,
etc.), discretize it **once**, and constrain both surfaces' tessellation to its
vertices. This is the harder half (per surface-kind-pair intersection math) and
is deferred until Level A is proven.

A pragmatic Level-B-lite: since the arrangement already produces the exact
intersection segments, snap **each curved surface's own ring tessellation** to
sample the shared curve at common parameters where two curved surfaces are
known to meet — narrower than full analytic curves, handles the common
coaxial/orthogonal cases.

## 4. Staged plan with gates

- **WP0 — instrument & corpus baseline.** Per-model: PLC facet count, short
  PLC-PLC edge count (diag_dense already does this), split count, min-dih.
  Freeze as the before-table. Gate: numbers reproduced for all 18 models.
- **WP1 — planar-facet representation.** Introduce a `PlanarFacet` (plane id +
  boundary loop) in the CSG kernel; flat primitive faces emit one each.
  Arrangement and classification operate on planar facets; barrels unchanged.
  Gate: every existing fixture (conform suite + the 4 S-parameter validation
  meshes) bit-identical region volumes; corpus meshes still valid.
- **WP2 — late triangulation of flat faces.** triangulate_facet triangulates a
  planar facet from its boundary + accumulated intersection constraints, once.
  Gate: counterbore short PLC-PLC edges → ~0; min-dih ≥ (target, e.g. 15°) on
  flat-seam models; split count and segments-phase time drop on
  perforated_plate.
- **WP3 — curved coordination (Level B).** Analytic intersection curves for the
  curved kind-pairs, shared discretization. Gate: serpentine / chain / coaxial
  models min-dih ≥ target; no divergent PLC-PLC edges remain corpus-wide.
- **WP4 — retire the patches.** With seams gone at the source, re-evaluate the
  coarsening collapse (still useful for sizing, but no longer load-bearing) and
  delete RAPIDMESH_WELD_TOL.

## 5. Effort, risk, payoff

- **Effort.** WP1+WP2 (flat faces) are the bulk and the structural change:
  introducing a planar-facet unit through arrange/scene/triangulate. Estimate a
  focused multi-day refactor; the per-facet constrained triangulation already
  exists. WP3 (curved) is open-ended (per-pair intersection math) and may stay
  partial.
- **Risk.** Stays entirely in the exact paradigm (no envelope, no float
  snapping) — region volumes remain bit-exact, the Maxwell interface
  requirement is preserved. The arrangement's exact predicates are unchanged;
  only the *units* they operate on change (polygon vs triangle). Main risk is
  the refactor surface area in arrange.rs/scene.rs; mitigated by the bit-exact
  fixture gate at WP1.
- **Payoff.** Removes the seam-sliver class AND the split overproduction in one
  move — i.e. the quality problem (min-dih 0.0) and the perf wall (segments
  insert-bound on 33408 splits) have the same cure. It is also the step that
  makes rapidmesh a real geometry kernel rather than tessellate-and-arrange,
  and produces exactly the clean PLC the Diazzi CDT already expects.

## 6. Open questions to resolve before WP1

- Planar-facet unit: reuse the existing per-triangle arrangement by grouping
  coplanar same-surface triangles post-hoc, or a first-class polygon facet from
  prim.rs? (Leaning first-class: cleaner, but touches more of prim.rs.)
- Non-convex / holed flat faces (prism with holes, sheet_polygon): the boundary
  loop is already polygonal there — does the same path absorb them?
- Coplanar overlaps between different solids (box on substrate): the arrangement
  must still split coplanar facets against each other; planar facets must carry
  the coplanar-merge logic that the triangle path has today.

Related: docs/cdt-recovery-plan.md, DESIGN.md (§ CSG kernel, § Tet mesher).
