# Bestandsaufnahme mesh_cdt (Stand 2026-06, ohne lfs/raycast)

Korpus: 67 Standard + 4 EM-Zielgeometrien (Spulen/Trafos). Mesher = `mesh_cdt`.
Quelle: `report/validation/benchmark_cdt.json`, EM via `rapidfem_geometries.py`.

## Kopfzahlen (71 Geometrien: 67 Standard + 4 EM)
- 70/71 meshen (1 Fail: pipe_junction). Volumen: 61.
- not-watertight: 37 | mit Straddlern: 17 | mit Slivers: 35.
- EM (rf_solenoid/cored_inductor/transformer/toroid_core): alle meshen; Slivers 0–12;
  watertight-false (Artefakt); Straddler 0 + dev 0 (maskiert durch C3b-Tagging, s.u.).

## Was solide funktioniert
- **Einzelne Solids, planar UND gekrümmt**: box, slab, sphere, cylinder, frustum, torus,
  hex/tri/prism_l/star_prism, wedge, ellipsoidish, nested_spheres, sizing_field — sauber,
  watertight, min-Dih gut (18–25°). Lone-Torus (1 Region): watertight, dev=0.0062 (echte
  Krümmung korrekt aufgelöst). **Der Mesher-Kern ist gesund.**
- **Einfache CSG ohne gekrümmte Verschneidung**: diff_box_sphere, sphere_minus_box,
  union_box_cyl, gear, nested_shells, via, baffled_tank — clean.
- **Planare Multi-Material (RF)**: microstrip, rf_iris_filter, rf_patch_antenna,
  rf_microstrip_line — Mesh sauber (nur watertight-Metrik-Artefakt, s.u.).

## Problem 1 — STRADDLER (gekrümmte CSG-Verschneidungen): 17 Geometrien
Schwere (Straddler-Zahl): pressure_vessel 11869, pipe_cross 9164, orbs 3457, tube 2321,
cross_cyl 1409, bearing 1192, chain 738, fused_unequal 643, fused_three 625, fused_deep 187,
capsule 61, diff_cyl_box 5, coax_step 4, dice 1.
- **Ursache (bewiesen)**: die restricted Delaunay **überbrückt** die konkave Verschneidungs-
  kehle, statt der Fläche in die Kehle zu folgen (dev/h = 3–6, echte geometrische Brücken).
  Die Schnittkurven-Punkte sind da, aber **nicht als Kanten erzwungen** → Faces spannen quer.
- **Bewiesen ausgeschlossen**: globale Dichte (lfs) → schlechter; geometrisches Crossing-
  Einfügen (raycast) → trifft Brücken nicht (Innen-Tet ragt nicht raus). Boissonnat-Oudot:
  restricted Delaunay ist nur mit **geschützten Features** mannigfaltigkeits-korrekt.
- **Fix = P1b Feature-Protection** (protecting balls auf den Schnittkurven → Kurvenpunkte
  werden Delaunay-Kanten → keine Brücken). Der einzige prinzipielle Hebel.

## Problem 2 — SLIVERS (Randschicht / dünn), straddlerfrei: ~15 Geometrien
perforated_plate 5883, lattice_cube 2907, drilled_block 2108, bracket 1482, serpentine 1107,
diff_box_cyl 953, square_to_round 436, spring 238, cone 145, mold_block 149, box_minus_2sph 140,
counterbore_plate 30, house 18, union_box_sphere 9, rf_dielectric_resonator 3.
- **Ursache**: flache Rand-Slivers (Vertices am Rand gepinnt → unerreichbar für Smoothing).
  `optimize` räumt Innen-Slivers (37→0), Rand-Rest bleibt.
- **Fix = P2 Sliver-Exudation** (gewichtete/reguläre Delaunay, Vertices fix → exakte Volumina).

## Problem 3 — FAIL: 1 Geometrie
- pipe_junction: Panic in `rapidmesh-geom/scene.rs:669` ("edge caps only slivers; no triangle
  reproduces its base edges") — **CSG-Arrangement upstream**, nicht der Mesher. Separat.

## EM-Zielgeometrien (Spulen/Trafos) — 4, alle meshen
- toroid_core 36k tets, solenoid 64k, cored_inductor 80k, transformer 91k.
- Slivers an den engen Spule/Kern-Interfaces (1–12); min-Dih 0–10°.
- watertight=false → **Artefakt** (Multi-Region, s.u.), keine echten Löcher.
- Straddler=0 + dev=0 → **maskiert** durch C3b-Tagging-Lücke (s.u.): die gekrümmten
  Interface-Flächen werden als Plane getaggt, also gar nicht auf Krümmung geprüft.

## Querschneidende REPORTING-Lücken (keine Mesh-Bugs, verzerren aber die Diagnose)
- **Diag5 — watertight-Metrik**: zählt Tripelkurven-/Interface-Kanten (>2 Faces global,
  aber pro-Region mannigfaltig) als non-manifold. Beweis: lone-Torus (1 Region) = watertight;
  dieselbe Fläche in 2-Region-Szene = "not watertight". → die 33 "not watertight" sind stark
  überzeichnet (viele Multi-Region-Artefakte). Billig zu fixen.
- **C3b — gekrümmtes Boundary-Tagging**: Interface-Kurvenflächen teils als Plane getaggt →
  dev/Straddler auf Multi-Region-gekrümmt unterzeichnet (dev=0 bei torus-in-air). Macht die
  Metriken auf genau den EM-Zielgeometrien blind.

## Empfohlene Priorisierung
1. **Diag5** (billig): watertight multi-region-korrekt → echte Leak-Zahl sichtbar machen.
2. **C3b** (Tagging): sonst sind Straddler/dev auf den EM-Zielen blind.
3. **P1b Feature-Protection**: der große Straddler-Block (17), der bewiesene Fix.
4. **P2 Exudation**: Rand-Slivers (~15 + EM-Interfaces).
5. **pipe_junction** CSG-Fail (upstream, separat).
