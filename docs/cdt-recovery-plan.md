# CDT Boundary-Recovery Rewrite

Decision (2026-06-13): the conforming-Delaunay-with-Steiner recovery in
`conform.rs` (patch marks, projected tiling balance, repair loop) is replaced
by a true constrained Delaunay tetrahedrization following Diazzi, Panozzo,
Vaxman, Attene, "Constrained Delaunay Tetrahedrization: A Robust and
Practical Approach", SIGGRAPH Asia 2023 (arXiv:2309.09805), plus the
segment-recovery theory of Si & Gaertner. No further patches land on the old
recovery; release happens with the holistic solution.

Why: the old layer RECONSTRUCTS face-to-patch membership with geometric
heuristics and produced an open-ended series of lottery bug classes (flat
in-plane slivers double-counting, interface starvation, crease split
cascades, welding micro-slivers, folded tile pairs). CDT recovery makes
conformity CONSTRUCTIVE (local cavity retetrahedrization with termination
guarantees) and membership becomes PROVENANCE: we know what we weave in.

Source discipline: paper text only. The authors' reference implementation is
GPL and must never be read (rapidmesh is proprietary).

## Pipeline (paper Alg. 2)

1. `D := Delaunay(V)` over the PLC vertices (exists: `DelaunayBuilder`).
2. Segment recovery until no PLC segment is missing (Sec. 3.3/4.2):
   encroachment reference point, acute-endpoint categories, Steiner points
   ON the segment as LNC points `t*v1 + (1-t)*v2`.
3. Face recovery until no PLC facet is missing (Sec. 3.4/4.4): cavity =
   tets incident to edges piercing the facet, split along the facet plane,
   per-half local Delaunay + boundary expansion; on the expansion failure
   mode (tet crossing the facet plane) switch to modified gift wrapping
   with the three validity conditions; cospherical ties broken by symbolic
   perturbation (memory-order parity, paper Alg. 1).
4. Region classification by flood fill (exists, stays).
5. Rounding LNC -> f64 with inversion/flattening checks; conformity-
   preserving local swaps as fallback (our optimizer plays this role).

## What we already have

- Exact kernel: staged orient3d/insphere (filter -> expansion), generic
  Ring trait, BigRational test oracle, implicit LPI/TPI points (degree 4/7;
  LNC is degree 1 and slots into the same homogeneous machinery).
- Bowyer-Watson with cavity API, neighbor pointers, hull handling.
- Exactly arranged watertight tagged PLC as input (much tamer than wild
  soups; segments/facets carry provenance already).
- Validation harness: exact region-volume gates, structural conformity
  checks, EM fixtures, density instrument, demo repros (patch antenna,
  RFIC spiral) that defeated the old layer.

## Work packages and gates

- WP1 exact: `Point3::Lnc { a, b, t }` + orient3d/insphere variants taking
  LNC arguments (polynomial in t, filtered then exact) + symbolic
  perturbation. Gate: oracle tests vs BigRational on randomized and
  degenerate (cospherical/collinear) configurations.
- WP2 segment recovery. Gate: on every test PLC each input segment is a
  union of DT edges, zero missing, exact.
- WP3 face recovery (expansion + gift-wrapping fallback). Gate: every PLC
  facet is a union of DT faces; region volumes bit-exact vs the PLC.
- WP4 rounding: LNC snap to f64 with validity checks; swaps preserving
  constrained facets where snapping degenerates. Gate: `check_structure`
  on rounded output across the suite.
- WP5 integration: `mesh_plc_with` recovery layer replaced; SurfaceFace
  built from constraint PROVENANCE (facet -> patch/surface/tags direct);
  refine_queue inserts split constraints first-class instead of mark
  bookkeeping; DELETE on_patch marks, tiling balance, inside_patch rejects,
  batch repair, adoption passes, divergence brakes. Gate: full test suite,
  EM fixtures, patch-antenna and RFIC-spiral repros mesh cleanly,
  measure_density unchanged (mean ~0.97 h), max-edge 1.5 h contract holds.

## Contracts that must not move

`TetMesh` schema (faces/surfaces/surface_owners/edges/abandoned_patches,
the latter ideally always empty afterwards), `MeshParams`, exact volume
gates, OVERSIZE_FACTOR 1.45 calibration, optimizer interfaces.
