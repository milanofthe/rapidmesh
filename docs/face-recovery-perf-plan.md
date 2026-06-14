# Flip-based facet recovery (performance)

Work package to replace the gift-wrapping facet recovery with a flip-based fast
path. This is the single largest meshing-performance lever, especially on
curved geometry.

## 1. Where the time goes (measured)

Per-phase meshing timing, current `feature/cdt-recovery` (conformal WP1/WP2),
rapidfem example geometries:

| model | tets | assemble | mesh (faces) | optimize | µs/tet |
|---|---|---|---|---|---|
| WR90 (flat box) | 2788 | 2.5ms | 69ms (18ms) | 26ms | 35 |
| iris_filter | 16685 | 12ms | 0.62s (0.28s) | 0.15s | 47 |
| horn + PML | 28912 | 4.5ms | 0.67s (0.26s) | 0.65s | 46 |
| **coax_step (curved)** | 2930 | 43ms | 2.15s (**1.37s**) | 0.31s | **850** |
| stepped_lpf (large/fine) | 88051 | 18ms | 19.6s (**11.5s**) | 6.6s | 296 |

Showcase corpus mirrors this: perforated_plate `faces 34s` (of 50s total),
lattice_cube `faces 15.7s`, bracket `faces 9.2s`.

**Diagnosis.** `assemble` (exact CSG) is negligible (<=43ms everywhere). The
bottleneck is the `faces` phase: `recover_faces` (cdt.rs), which detects edges
piercing each PLC facet and retetrahedralizes the cavity with **gift wrapping**
(`gift_wrap`, Shewchuk's paper fallback). Its inner loop is
O(front_faces x cavity_vertices) with an exact orient/insphere per candidate
pair -- inherently slow for large or thin cavities.

The clearest evidence is coax_step: only 2930 tets but 2.5s (850us/tet), ~24x
the per-element cost of the flat WR90 (35us/tet). The extra cost is entirely
the gift-wrap of the curved cylinder-barrel facets. flat/clean EM geometry is
already fast; **curved geometry pays a ~20x face-recovery tax.**

## 2. The fix: flip-based recovery with a gift-wrap fallback

The standard fast method (Si & Gaertner 2005; Shewchuk 2002) recovers a facet
by bistellar flips (2-3, 3-2, 2-2/edge-removal) that remove the tet edges
piercing it, reusing the existing neighbor adjacency -- O(local) per flip
instead of O(cavity) per gift-wrap. Flips alone do not always succeed (known
hard configurations need Steiner points or a cavity retet), so gift wrapping
stays as the **correctness-guaranteeing fallback**; flips are the fast path that
clears the common cases.

This is a fast-path-first integration: lowest risk, because the proven-correct
gift-wrap remains for anything flips cannot clear, and the conform / S-parameter
gates keep it honest.

### What already exists

- `DelaunayBuilder` (delaunay.rs): flat tet slab with per-face neighbor
  pointers, walk location, neighbor-BFS cavities, `edge_map`, exact
  predicates on explicit f64 + cheap degree-1 (LNC/PAC) Steiner points. Good
  base; the recovery data structure.
- Flip LOGIC exists in optimize.rs (2-3, 3-2 generalized to edge removal of
  k-rings, 2-2 surface flips) -- but quality-driven, on `TetMesh` with
  alive/owner maps, NOT on the `DelaunayBuilder`. It is a reference, not
  directly reusable.
- `recover_faces` / `recover_one_facet` (cdt.rs): the incremental
  creation-log/bbox skip + `face_exists` short-circuit + `boundary_intact`
  precondition are KEPT; only the cavity-surgery step (currently
  `retet_facet_cavity` -> `gift_wrap`) gains a flip fast path.

## 3. Staged plan with gates

- **P0 -- instrument.** Counters in `gift_wrap` / `recover_one_facet`: cavities
  per model, cavity sizes (front faces, verts), total gift-wrap time, flips
  attempted. Confirm coax_step / stepped_lpf / perforated_plate are
  gift-wrap-bound and freeze the before-table (coax 850us/tet, perforated_plate
  faces 34s). Gate: numbers reproduced.

- **P1 -- flip primitives on DelaunayBuilder.** Implement 2-3 and 3-2 flips
  (and the k-ring edge removal / 2-2 as needed) directly on the builder's
  neighbor-pointer structure, with exact orient3d guards and local neighbor
  rewiring. Port the logic from optimize.rs, adapt to the builder. Gate: each
  flip preserves a valid tetrahedralization (unit tests: volume invariant,
  neighbor consistency, no inverted tets) and round-trips (2-3 then 3-2).

- **P2 -- flip recovery of piercing edges.** For a facet with a piercing edge,
  apply the flip that removes the crossing (2-3 to expose the facet edge, 3-2
  to delete the piercing edge), iterating until no tet edge pierces the facet.
  Deterministic ordering to avoid cycling. Gate: on facets the current code
  recovers, the flip path produces a conforming result with the facet present;
  the recovered tet set has bit-exact region volume.

- **P3 -- gift-wrap fallback.** When flips stall (a full round with no progress
  / a hard config), fall back to the existing `gift_wrap` cavity retet for that
  facet. Gate: every facet is recovered (flip OR fallback); conform suite green.

- **P4 -- gate & measure.** conform suite (exact region volumes, conformity,
  feature edges) green; the 5 rapidfem EM validations (WR90, coax_step, horn,
  iris_filter, stepped_lpf) green with unchanged S-parameters; showcase corpus
  min-dih unchanged. Perf target: curved cases (coax_step) drop toward the flat
  per-element rate (~50us/tet, i.e. coax ~0.2s instead of 2.5s);
  perforated_plate / stepped_lpf `faces` time falls sharply.

## 4. Risk

Facet recovery is correctness-critical -- the conform gate and the rapidfem
S-parameters depend on a conforming PLC. Mitigations: (a) flips are exact
(orient3d on the builder's points); (b) the proven gift-wrap stays as the
fallback, so no facet can be left unrecovered; (c) the flip-recovery termination
(cycling) is bounded by a deterministic ordering plus a step cap that triggers
the fallback. The blast radius is confined to the cavity-surgery step; the
broadphase, skip logic, and segment recovery are untouched.

## 5. Out of scope (separate future levers)

- Refinement cost (stepped_lpf `refine 7.9s`, 6 rounds): the queue-driven point
  insertion for fine/thin regions. A separate perf item.
- optimize cost (stepped_lpf 6.6s): smoothing/flip rounds. Separate.
- Barrel-quad flats (reverted, see conformal-tessellation-plan.md S9): would cut
  the NUMBER of hard curved facets, compounding with this WP; gated on the
  cavity-sampler issue.

Related: docs/conformal-tessellation-plan.md, docs/cdt-recovery-plan.md, cdt.rs
(`recover_faces`, `gift_wrap`), optimize.rs (flip logic reference).
