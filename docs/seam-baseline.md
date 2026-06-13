# Seam baseline (WP0)

Frozen before-state for the conformal-tessellation refactor
(conformal-tessellation-plan.md). Numbers are the current `feature/cdt-recovery`
state *after* the coarsening collapse and the parallelization round, i.e. the
bar the refactor must beat on quality (min-dih) and perf (segments / splits).

Reproduce: `PYTHONPATH=python/python_src python python/examples/showcase.py <id>`
(per-model, subprocess-isolated against the known-panicking repros);
seam-edge provenance via `cargo run --release -p rapidmesh --example diag_dense`.

## The 18 meshing models (post-coarsening)

| model | tets | pts | min-dih° | class |
|---|---|---|---|---|
| sizing_field | 1961 | — | 26.3 | flat (sizing) |
| baffled_tank | 7763 | — | 22.9 | flat + sheets |
| spring | 25989 | — | 14.4 | curved (helix) |
| gear | 7356 | — | 9.0 | flat (prism) |
| bearing | 17008 | — | 8.2 | curved + flat |
| chain | 14549 | — | 7.9 | curved (tori) |
| coax_step | 2246 | — | 6.0 | curved (coaxial) |
| microstrip | 1342 | — | 4.9 | flat + sheet |
| bracket | 10711 | — | 4.7 | flat + bores |
| mold_block | 26645 | — | 4.4 | flat + mixed cuts |
| perforated_plate | 13268 | — | 3.6 | **flat, 33 bores** |
| dice | 9709 | — | 3.5 | flat + sphere dimples |
| lattice_cube | 25903 | — | 2.8 | flat + bores |
| counterbore_plate | 11666 | — | 0.6 | **flat + concentric bores** |
| rocket | 12340 | — | 0.6 | curved (cones) |
| orbs | 10800 | — | 0.3 | curved (sphere fusion) |
| serpentine | 9640 | — | 0.0 | **curved (sweep kinks)** |
| laminate | 19886 | — | 0.0 | **flat (coplanar stacks)** |

min-dih 0.0–0.6 = the seam slivers the refactor targets. Flat-seam models
(counterbore, laminate, perforated_plate) are the WP2 (Level A) targets; curved
ones (serpentine, orbs, rocket) are WP3 (Level B).

## Seam diagnostic (counterbore_plate, the reference case)

From diag_dense (own arrangement of a 4-bore counterbore plate, target h 0.15):
- PLC: 1578 edges, 312 already < t/4 (seam micro-facets in the input PLC)
- Mesh (after optimize): **309 short PLC-PLC edges, 254 DIVERGENT**
  (endpoints on different surface sets — the irreducible seam slivers), only
  55 subset-collapsible
- divergent surface-kind pairs: 254× Cylinder vs Cylinder (the bore-wall ×
  step-floor concentric-ring incoherence)

## Perf reference (perforated_plate, the segments wall)

- assemble 7.2s, mesh 53.8s (**segments 41s**, faces 11.5s), optimize 3.9s
- initial DT insert: 25ms (2140 PLC pts) → **35548 pts after recovery**
  (≈33408 Steiner splits = seam overproduction, the insert-bound cost)

## Gate targets after the refactor

- WP2 (flat): counterbore divergent PLC-PLC → ~0; min-dih ≥ ~15° on
  counterbore / laminate / perforated_plate / baffled_tank; perforated_plate
  split count and segments-time drop sharply.
- WP3 (curved): serpentine / orbs / rocket min-dih ≥ ~15°; no divergent
  PLC-PLC edges corpus-wide.
- Throughout: conform suite + 4 S-parameter validation fixtures stay green,
  region volumes bit-exact.
