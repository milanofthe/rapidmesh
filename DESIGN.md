# rapidmesh — Design

Standalone tetrahedral mesh generator for 3D EM FEM (Maxwell, H(curl)/Nédélec),
pure Rust, no gmsh/OpenCASCADE dependency. General-purpose for arbitrary RF/EM
structures (not layer-stack specialized). Proprietary for now, possibly OSS later.

Primary consumer: rapidfem (Nédélec-2 frequency-domain solver, currently gmsh-based
via `geometry.py`). rapidmesh must eventually be a drop-in alternative meshing
backend validated against the existing gmsh-meshed fixtures (same geometry, compare
S-parameters).

## Architecture

Everything converges on one central intermediate representation: a **tagged PLC**
(watertight triangle surface complex with face/region tags and back-references to
the originating analytic/NURBS surface).

```
Primitives ──► exact mesh CSG (indirect predicates,   ──┐
               multi-operand boolean expressions)        │
                                                         ▼
STEP (later) ──► parser (subset) ──► NURBS          tagged PLC ──► CDT with exact
                 tessellation                       (+ surface     boundary recovery
                                                    backrefs)          │
                                                                       ▼
                                                    Delaunay refinement (sizing field)
                                                                       │
                                                                       ▼
                                                    quality pass (edge removal,
                                                    smoothing, sliver hunt via
                                                    min dihedral angle)
                                                                       │
                                                                       ▼
                                                    optional: order-2 node snapping
                                                    onto analytic/NURBS surface
                                                    (needed for Nédélec-2 rates)
```

### Stage decisions (research-backed, 2026-06)

- **CSG kernel = mesh-based exact arrangements, NOT B-rep booleans.** Naive f64
  booleans (Cork, Blender, csgrs/BSP) break systematically on coplanar facets,
  and CSG trees produce coplanar configurations "nearly systematically" (box on
  substrate). Blueprint: Lévy, "Exact predicates, exact constructions and
  combinatorics for mesh CSG" (ACM TOG 2024, geogram, permissive license — best
  legal porting reference) and Cherchi/Pellacini/Attene/Livesu, "Interactive and
  Robust Mesh Booleans" (SIGGRAPH Asia 2022, indirect predicates). Key technique:
  intersection points represented implicitly (indirect predicates + interval
  filtering, expansion arithmetic fallback), CDT remeshing with symbolic
  perturbation for co-cyclic points. Cascaded booleans must be evaluated as
  multi-operand expressions, not pairwise-with-snapping (snapping exact points to
  float between ops breaks subsequent ops).
- **Tet mesher = CDT with exact boundary conformity, NOT envelope-based.**
  Conformal material interfaces are a hard requirement for Maxwell FEM (every tet
  inside exactly one material region); fTetWild's epsilon-envelope (default
  b/1000) smears interfaces and is therefore disqualified as the core. Blueprint:
  Diazzi/Panozzo/Vaxman/Attene, "Constrained Delaunay Tetrahedrization: A Robust
  and Practical Approach" (SIGGRAPH Asia 2023): TetGen's CDT theory, but
  floating-point-only via indirect predicates with implicitly represented Steiner
  points, parameter-free, 100% success on Thingi10k, exact boundary conformity.
  Also fixes an algorithmic (not numeric) gap in TetGen's theory.
- **Refinement = Ruppert/Shewchuk-style Delaunay refinement** with sizing field.
  Termination guarantee is tied to the radius-edge ratio criterion (provable for
  bound ~2; pushing the bound toward 1 breaks termination). Radius-edge does NOT
  bound dihedral angles — slivers survive and need a separate stage.
- **Quality pass targets the minimal dihedral angle** (the load-bearing metric for
  Nédélec conditioning; radius ratio is explicitly inadequate as a sliver
  detector). Toolkit per HXT paper (Marot & Remacle 2020): edge removal, improved
  Laplacian smoothing, Growing SPR Cavity ("mother of all flips"). A single bad
  tet measurably increases preconditioned-solver iteration counts.
- **Order-2 geometric elements matter because rapidfem is Nédélec-2:** on curved
  domains, straight tets degrade H(curl) convergence from order 2 to ~1.5;
  order-2 curved meshes (midside nodes snapped to the true surface) restore the
  full rate (arXiv 2201.00883). The PLC carries surface back-references so the
  snapping stage can project onto the analytic/NURBS surface (same idea as HFSS
  curvilinear elements).
- **Sizing**: wavelength-based default (pollution effect: h must shrink faster
  than proportional to wavelength, (kh)^{2p}·C_sol bounded — arXiv 2408.04507) +
  local feature size + user maxh + **consume external size fields** (rapidfem
  already exports residual-estimator/Dörfler size fields → HFSS-style adaptive
  loop: solve → error estimate → refine → ΔS convergence).
- **Parallelism: multithreaded CPU (rayon), not GPU.** Negative research finding:
  no significant GPU tet-meshing successor as of 2025/26; SOTA (Diazzi CDT, HXT
  at ~55M tets/s) is CPU-multithreaded.
- **No 2.5D fast path** (decision 2026-06-09). The mesher is general-purpose for
  arbitrary EM structures. The HFSS-Phi lesson (10-20x via geometry-class
  specialization) may motivate a second engine *in the same framework* later; the
  tagged-PLC representation keeps that door open but nothing is built for it now.

## Licensing constraints (proprietary codebase)

Reimplement from papers only; dependencies must be MIT/Apache.

| Source | License | Usable how |
| --- | --- | --- |
| TetGen | AGPL | paper only — never read/port the code |
| gmsh incl. HXT | GPL | paper only |
| CGAL | GPL/commercial | paper only |
| Diazzi CDT reference impl | GPL (LGPL flag) | paper only |
| fTetWild | MPL-2.0 | algorithm reference; no file copies into proprietary code |
| Lévy geogram CSG | permissive | best porting blueprint for the CSG kernel |
| Shewchuk predicates | public domain | free |
| geometry-predicates / robust crates | MIT | use as dependency |
| truck (NURBS/B-rep/STEP, ricosjp) | Apache-2.0 | use as dependency (surface lib / STEP tessellation, NOT its booleans) |
| spade (2D CDT) | MIT/Apache | use as dependency |

Caveat: Shewchuk-port predicates don't handle exponent overflow — coordinates
must be normalized to roughly [1e-142, 1e201]; normalize input geometry to a unit
box internally.

## Crate layout

```
rapidmesh/                workspace
  crates/
    rapidmesh-exact       exact arithmetic foundation: Shewchuk expansions,
                          conservative interval filter (ulp-widening), generic
                          Ring trait (same geometric code runs as filter, exact,
                          or rational test oracle), implicit points (LPI/TPI as
                          lazy homogeneous coordinates), staged-exact predicates
    rapidmesh-geom        primitives (box/cylinder/sphere/extrude/...), tagged PLC
                          type, surface back-references, transforms
    rapidmesh-csg         exact mesh arrangements + multi-operand booleans
                          (indirect predicates, interval filter, expansion fallback)
    rapidmesh-tet         CDT (boundary recovery via implicit Steiner points),
                          Delaunay refinement, sizing fields, quality pass,
                          order-2 snapping
    rapidmesh             facade: builder API, mesh export (rapidfem format,
                          .msh/.vtk for inspection)
```

## Roadmap

- **v0.1 (core loop):** primitives + exact CSG + CDT + refinement + quality pass.
  Validation gate: rapidfem fixtures meshed by rapidmesh vs gmsh, S-parameter
  comparison (closing the loop — our unique advantage over generic meshers).
- **v0.2:** order-2 curved elements (midside snapping), external size-field
  consumption (adaptive loop with rapidfem error estimator).
- **v0.3:** STEP import (pragmatic AP203/AP214 subset for MCAD solids; ruststep
  is stalled/experimental, Foxtrot proves feasibility at ~26/915 entities —
  subset is the only realistic scope).

Risk order (hardest first): exact CSG arrangements > CDT boundary recovery >
Delaunay kernel/refinement > quality pass > STEP subset.

## Key references

- Diazzi, Panozzo, Vaxman, Attene: Constrained Delaunay Tetrahedrization: A
  Robust and Practical Approach. SIGGRAPH Asia 2023. arXiv:2309.09805
- Lévy: Exact predicates, exact constructions and combinatorics for mesh CSG.
  ACM TOG 2024. arXiv:2405.12949
- Cherchi, Pellacini, Attene, Livesu: Interactive and Robust Mesh Booleans.
  SIGGRAPH Asia 2022. arXiv:2205.14151
- Marot, Remacle: Quality tetrahedral mesh generation with HXT. arXiv:2008.08508
- Hu, Schneider, Wang, Zorin, Panozzo: Fast Tetrahedral Meshing in the Wild
  (fTetWild). SIGGRAPH 2020.
- Si: TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator. ACM TOMS 2015.
- Shewchuk: Tetrahedral Mesh Generation by Delaunay Refinement. 1998.
- Aylwin, Jerez-Hanckes et al.: FE domain approximation for Maxwell on curved
  domains. arXiv:2201.00883 (order-2 geometry for order-2 elements)
- Sharp error bounds for edge-element discretisations of high-frequency Maxwell.
  arXiv:2408.04507 (pollution-aware sizing; shape regularity is the load-bearing
  quality metric; conformal material interfaces required)
- Mesh quality impact on EM/thermal FEM (hal-00414249): min dihedral angle as
  quality gate; single bad tet degrades solver convergence
