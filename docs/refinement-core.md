# refinement-core: Umbau des 3D-Volumen-Pfads

Ziel: die beste Standalone-SOTA-Lösung für EM-FEM-Meshes (3D, H(curl)/Nédélec)
und MoM-Meshes (2D). Der 2D-Pfad (`surf2d`, `mesh_2d`/`mesh_layers`) bleibt
unangetastet — er ist die Referenz für die mesher-in-the-loop-API, die 3D
bekommen soll.

## Diagnose (Stand vor dem Umbau)

Der 3D-Pfad friert eine vorab gemeshte Surface ein und hofft, dass die
restricted Delaunay der Volumen-Punktmenge sie reproduziert (Oversampling
0.7×). Für planare Facets stimmt das per Koplanarität bit-exakt; für gekrümmte
Interfaces ist es nur wahrscheinlich. Folgen (Report: 48/72 watertight, 40/72
sliver-frei; Booleans bei 0.0–0.5° min-Dihedral):

1. `scene.rs:341` rundet das exakte CSG-Arrangement auf f64, weldet mit
   `1e-12·diag` und repariert T-Junctions per Facet-Split → Nadeln
   (GitHub-Issue #1).
2. `csg/triangulate.rs` retrianguliert ohne Qualitätskontrolle → exakt-valide
   Nadeln auf Interface-Facetten.
3. `brep/build.rs::recover_curve` konstruiert `Curve::Intersection` nie;
   `analytic_circle` verlangt achsen-senkrechte Schnitte (`|n·axis|>0.99`).
   Schräge Zylinderschnitte (Ellipsen), Zyl∩Zyl etc. → Polyline mit
   Sagitta-Fehler = der Straddler-Treiber.
4. `cvt.rs::mesh_cdt` macht keine Boundary-Recovery: pro Region eine
   unconstrained Delaunay + Centroid-Ray-Cast. Gekrümmte Interfaces: beide
   Seiten können sich uneinig sein → non-manifold Faces, Löcher, Straddler
   (Tasks #50/#51, die 7 ignorierten Tests in `rapidmesh-tet/tests/conform.rs`).
5. `optimize.rs` ist streng greedy-monoton (`QUALITY_EPS` pro Einzelschritt),
   ohne Exudation/Perturbation, mit unantastbaren PLC-Vertices
   (Apex-Slivers strukturell unlösbar) und einem Fidelity-Ratchet-Guard, der
   curved-Boundary-Reparatur blockiert, während Fidelity-Snapping umgekehrt
   die PLC-Volumina verletzt.
6. Keine Local-Feature-Size-Behandlung: kurze Boolean-Kanten werden 1:1 zu
   Elementgrößen.

## Zielarchitektur

Delaunay-Refinement gegen Carrier-Oracles mit geschützten Features
(Boissonnat–Oudot restricted Delaunay + Cheng–Dey–Ramos protecting balls,
wie CGAL Mesh_3), statt frozen surface + Hoffnung.

```
Primitive/Import
  → exaktes CSG-Arrangement (bleibt; implizite Punkte durchgängig)
  → B-rep mit analytischen Carriern UND analytischen Schnittkurven
      (NEU: Curve::Intersection via alternierende Projektion, Conics)
  → 1D: protecting balls auf Feature-Kanten (Ball-Radien = LFS-Klemme
      für kurze Boolean-Kanten; Apex/Corner = Ball, kein gepinnter Vertex)
  → 3D-Delaunay-Refinement in EINER Triangulation über alle Regionen:
      - facet criteria: Größe h(x), Form (radius-ratio), Approximations-
        distanz (tol_surf·R) — verletzte restricted-Delaunay-Facets werden
        durch Einfügen ihres auf den Carrier projizierten Zirkumzentrums
        verfeinert (Steiner = explizite f64 → kein implicit-predicate-Stau,
        die Perf-Falle des alten cdt.rs, GitHub #2/#3)
      - cell criteria: radius-edge, Größe, Region aus DomainTree
      - Konformität per Definition: eine restricted-Delaunay-Facette trennt
        zwei Regionen; kein Straddler möglich, Multi-Region trivial
  → Lloyd/ODT-Relaxation (bleibt, jetzt auf der Refinement-Delaunay)
  → Sliver-Endstufe: Perturbation + Exudation (weighted Delaunay) +
      Composite-Ops mit Rollback; greedy Ops aus optimize.rs bleiben
  → Mesh3D-Bundle (rapidmesh-topo), mesher-in-the-loop-API wie 2D
```

Was bleibt unangetastet: `rapidmesh-exact`, `rapidmesh-csg`, der
Delaunay-Kernel (`delaunay.rs`), `DomainTree`/Sizing, die Charts/Oracles
(`surfchart.rs`, `project.rs`), der komplette 2D-Pfad (`surf2d`,
`mesh_2d`/`mesh_layers`), `rapidmesh-topo`-Topologie/Geometrie.

Was ersetzt wird: der frozen-surface-Fluss `surface_sites → cdt3 →
Centroid-Klassifikation` in `cvt.rs`/`cdt3.rs`/`brep_mesh.rs`. `brep_mesh`
bleibt für den Standalone-Surface-Export (`surface_mesh`) erhalten.

## Arbeitspakete

- **WP1 Kurven-Fundament** (`rapidmesh-brep`, `rapidmesh-csg`):
  `Curve::Intersection` konstruieren + evaluieren (alternierende Projektion
  auf beide Carrier, Newton-Polish); `analytic_circle` → schräge Schnitte
  (Ellipse); Issue #1: geteilter exakter Punkt an koplanaren T-Junctions
  (`Point3::lli_coplanar`) statt Weld+Repair.
- **WP2 Refinement-Kern** (`rapidmesh-tet`, neues Modul `refine.rs` +
  `protect.rs`): protecting balls auf Brep-Kanten (Radien aus LFS =
  min(Abstand zu nicht-inzidenten Features, Kurven-Krümmung, h(x)));
  restricted-Delaunay-Facet-Extraktion pro Brep-Face (Oracle: Projektion +
  Seitentest); Refinement-Loop (encroachment: Ball > Facet > Cell);
  Region-Zuordnung pro Zelle über die Facetten-Struktur (nicht Ray-Cast).
  Abnahme: die 7 `#[ignore]`-Tests in `tests/conform.rs` + Corpus-Metriken
  (watertight 48/72 → 72/72, Boolean-min-Dihedral > 15°).
- **WP3 Sliver-Endstufe** (`optimize.rs`): Exudation (weighted Delaunay),
  Perturbation, Composite-Ops mit Rollback; `INSERT_BELOW`-Lücke (10–35°)
  schließen; Ratchet-Guard + Volumen-Konflikt entfallen (Fidelity läuft über
  Refinement-Kriterien).
- **WP4 mesher-in-the-loop 3D** (`rapidmesh-topo`, `python/`): Parität zur
  2D-API — `Mesh3D` behält Szene + Params und bekommt `remesh(h)`;
  `GradedField3D` (Octree-Variante von `gradefield.rs`); Dörfler für Tets
  (`dorfler_size_points` auf `TetMesh`); Element-Budget single-pass über das
  Count-Integral `N = K·∫ 1/h³ dV` (statt 6×-Retune-Loop in
  `mesh_cdt_budgeted`); `minh`/`maxh`/`grading` wie `Mesh2DOptions`.
- **WP5 STL-Oracle**: Soup-Oracle (AABB-Projektion), Feature-Erkennung per
  Dihedral-Schwelle → protecting balls, Facet-Kriterium gegen ε-Envelope.
  Abnahme: bench spot/fandisk min-Dihedral > 15°.
- **WP6 STEP**: AP203/214-Subset-Reader auf die B-rep; `nurbs_footpoint`
  mit Newton. Nach WP2 (NURBS nur als Oracle nötig).

## API-Parität 2D ↔ 3D (WP4-Spezifikation)

| 2D (ist) | 3D (soll) |
| --- | --- |
| `mesh_2d(regions, h, opts) -> Mesh2D` | `mesh_3d(plc/scene, h, opts) -> Mesh3D` |
| `Mesh2D.remesh(h)` (Regionen+Opts retained) | `Mesh3D.remesh(h)` (Szene+Params retained) |
| `Mesh2DOptions{min_angle_deg, target_count, minh, maxh, grading}` | `Mesh3DOptions{radius_edge, target_count, minh, maxh, grading, tol_surf}` |
| `GradedField` (Grid + Fast-Sweep, O(1) eval) | `GradedField3D` (Octree + Fast-Sweep) |
| `dorfler_mark(eta, theta)` (dimensionslos, bleibt) | dito, plus `TetMesh::dorfler_size_points` |
| Budget: closed-form `∫1/h² dA`-Bisektion | Budget: closed-form `∫1/h³ dV`-Bisektion |

## Meilensteine / Verifikation

1. WP1 grün: bestehende brep/csg-Tests + neue Kurven-Tests; Corpus-Lauf
   zeigt weniger Sagitta-Straddler.
2. WP2 grün: alle 7 `#[ignore]` in `tests/conform.rs` aktiviert und grün;
   `bench/results.json` regeneriert; Vergleichslauf gegen gmsh/tetgen
   (`bench/compare_meshers.py`).
3. WP3: Corpus sliver-frei (min-Dihedral ≥ 15° überall, Ziel > 20°).
4. WP4: rapidfem-AMR-Loop (Dörfler → remesh) als Integrationstest.
