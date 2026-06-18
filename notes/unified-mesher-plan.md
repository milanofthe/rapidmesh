# rapidmesh — Konsolidierungsplan: einheitlicher chart-getriebener Mesher

Stand-Handoff (überlebt /compact). Branch: `feature/cvt-mesher`. Der Report
(`report/report.tex`) ist NUR für Ergebnisse + finale Version, NICHT für Planung.

## Ziel

`mesh_cdt` (constrained, boundary-conforming) wird der **einzige** Mesher; der
alte `cvt::mesh` (unconstrained + Oversampling + statistische Rand-Erholung) wird
**gelöscht**. Surface-Meshing wird **ein** chart-getriebener Algorithmus über alle
Flächen-Arten. Toleranz (`tol_surf`/`tol_edge`, rel. Sagitta) + gröbste-Stelle-
Insert sind besonders für **gekrümmte** Flächen zentral.

## Architektur-Entscheidung: Oracle statt globaler Karte (2026-06-17)

Die globale 2D-Karte kann **prinzipiell nicht** der universelle Mechanismus sein:
Hairy-Ball (kein reguläres Gitter ohne Pol auf S²) + Theorema Egregium (K≠0 ⇒ keine
Isometrie zur Ebene). Karten sind exakt nur für **Developables** (K=0: Ebene/Zyl/
Kegel/Extrudat).

Der vereinheitlichende Entwurf (SOTA: CGAL Mesh_3, restricted Delaunay/CVT) ist:

- **Repräsentation = dualer Carrier** (haben wir bereits in `site.rs`):
  - Float-Oracle `project`/`frame`/`curvature_radius` → Verteilung/Relaxation/Sizing.
  - Exakter Konstruktor `Carrier::exact()` → Konformität + exaktes Volumen. Exakt
    nur auf **rationalen** Trägern (Ebene `Pac`, Gerade `Lnc`); gekrümmt ist per
    Konstruktion relativ-tol (transzendent — kein exakter rationaler Punkt auf S²).
- **Mechanismus = intrinsisches CVT + Delaunay**, lokale **Tangentialebene als
  universelle Karte pro Punkt** (Exponentialmap). Globale Developable-Karte = der
  Spezialfall, in dem alle lokalen Frames zu einer Karte zusammenkleben.
- **Nicht-Oracle, bleibt explizit** (korrekt, kein Hack): der B-Rep-Feature-Komplex
  (Ecken/Feature-Kurven/Trim-Loops/Region-Adjazenz) als harte Constraints, und das
  `inside`-Oracle fürs Volumen. Genau hier scheitern reine implizite Mesher; SOTA
  fügt das als „feature protection" wieder hinzu — wir haben es als B-Rep.

Watertight bleibt: wir extrahieren Flächen-Tris und **frieren sie als harte
Constraints** für `cdt3` ein (nicht statistisches Volumen-Readback wie der alte Pfad).

Folge für die Kugel: **flächen-nativ statt radialer Karte**. Geschlossene Kugel →
geodätische Icosphere (isotrop, polfrei, nahtlos). Getrimmter Cap (≤ Hemisphäre) →
azimutaler Chart, Frame aus **Facet-Centroid** (Randmittel ist bei Hemisphäre
nulldeutig). Developables/Ebene → Chart-Fastpath (Ebene exakt).

## Die zwei Ansätze (warum es aktuell zwei Pfade gibt)

- **Alt `cvt::mesh`** (aktueller Default via `mesh_plc_with`): unconstrained
  Delaunay über Surface+Innenpunkte, Surface OVERSAMPLED feiner als Volumen, Rand
  statistisch zurückgelesen (restricted Delaunay, Centroid-Klassifikation).
  Konformität = Sampling-Dichte-Wette → Slivers/Straddler/falsche Volumina an
  gekrümmten/dünnen/komplexen Stellen. Benchmark: Median min-Dihedral **7.5°**,
  8 Geometrien bei ~0°.
- **Neu `mesh_cdt`** (hinter `RAPIDMESH_CDT`-Env-Flag): frozen Surface = HARTE
  Constraint (boundary-constrained Delaunay), keine Straddler, watertight by
  construction, exakte Volumina als Theorem (`prop:watertight`). Box/Sphere:
  20–26° min-Dihedral. Das ist der richtige, spec-konforme Weg.

Zwei Pfade = Übergangs-Scaffolding (neuen neben funktionierendem altem reifen
lassen), kein Dauerdesign.

## Schon erledigt (committet auf feature/cvt-mesher)

- **Stage 2 — 2D constrained Delaunay** (`surf2d.rs`: `Cdt`, `triangulate_constrained`,
  Sloan-Flip + Außen/Loch-Filter). Konkave/gelochte Plates korrekt.
- **Konstruktiver Carrier** (`site.rs`: `Carrier::exact()` → exakte `Point3::Lnc`
  auf Linie / `Pac` auf Ebene; `Site::at`). Geschlossen unter Recovery, bit-exakt
  auf achsenausgerichteten Ebenen.
- **cdt3.rs** — boundary-constrained Tetraedrisierung:
  - 3D-Flips `flip23`/`flip32` via `DelaunayBuilder::replace_cavity` (exakt vorgeprüft).
  - `piercing_edge` + `recover_facet` (Flip-basiert, Steiner-Notnagel) — `Recover::{Done,NeedSteiner}`.
  - `tetrahedralize_constrained(verts: &[Site], tris, tri_carrier, interior, lo, hi)`:
    exakt einfügen, gekrümmte Facets recovern, planare übersprungen (koplanar).
  - `classify_regions(tets, points, surface_face_oracle)` — Flood-Fill, kein Centroid.
- **Per-Entität-Sizing-API** (hierarchisch): `report/corpus.py`-fern in Rust+Python:
  - `MeshParams`: `tol_edge`/`tol_surf` (rel. Sagitta, Default 1e-2), `maxh_edge`/
    `maxh_surf`/`maxh_vol`, per-Entität `edge_maxh`/`edge_tol`/`surf_maxh`/`surf_tol`
    (keyed by brep-ID); Resolver `edge_maxh_for`/`surf_maxh_for`/… (Dimension-Fallback).
  - B-Rep-Topologie-Read-Modell: `rapidmesh-brep/src/topology.rs::extract_topology`
    (Region→Faces→Kanten + Geometrie + Inzidenz).
  - Python: `g.region(sel).surf(sel).edge(sel).maxh/.tol`, Selektoren
    `id/tag/normal/near/between`, unfiltered = Dimension-global, spezifischste gewinnt.
    (`python_src/rapidmesh/geometry.py` `_Scope`/`_Topology`; Binding `python/src/lib.rs`
    `topology()` + `PyTopology` + mesh()-Params.)  12 Permutations-Tests grün
    (`python/tests/test_sizing.py`).
  - `point_size` trägt per-Entität-Größe (Surface-Punkte), damit `optimize` die
    Verfeinerung nicht coarsent.
- **Adaptive Volumen-Seeding** (gröbste-Stelle-Insert) in BEIDEN Lloyd-Loops
  (`cvt::mesh` UND `mesh_cdt`): wo Tet-Kante > lokales `h` → freie Site am Mittelpunkt,
  separationsgeschützt, klar von Surface; dann re-relax. Fixt `per_region_sizing`.
- **Stage 1a Surface-Vereinheitlichung (TEILWEISE)**: `brep_mesh::surface_sites`
  trianguliert jetzt **planare** Faces (constrained 2D, Site-Indizes via `edge_sidx`),
  liefert `tris` + `tri_carrier`. Gekrümmte Faces: noch KEINE Tris. Alter Pfad
  ignoriert `tris`; `mesh_cdt` nutzt noch `frozen_surface`/`surface_mesh`.
- **Einheitlicher Benchmark + Gallery**: `report/corpus.py` (67 Geometrien:
  validate 39 + showcase 23 + RF 5), `bench()` (Quality/Timing, panik-robust),
  `report/render_gallery.py` (PNG je Geometrie + Sizing-Permutation; `--cdt` opt-in).
  Viewer-Render: enger gerahmt (`canvas3d.ts::fitCamera` 0.5·diag) + 2× (DSF).

## Stand-Messung (Gate)

- Default `cvt::mesh`: conform 10/2 (rot: `box_feature_edges`, `cylinder_via`);
  benchmark 66/67 (rot: `pipe_junction` Assembly-Panic, vorbestehend in `scene.rs`).
- `mesh_cdt` als Default (Experiment): conform 7/12, ABER besser bei
  `box_feature_edges` + `per_region_sizing`. Rot: per-Entität (`per_edge`/`face_maxh`/
  `surface_maxh`, weil mesh_cdt `frozen_surface` statt `surface_sites` nutzt),
  `cylinder_via` (Multi-Region), `torus` (Curved-Recovery), und **6× langsamer**
  (509 s vs 83 s — `piercing_edge` O(Tets)).

## Algorithmus 1 — einheitliches chart-getriebenes Face-Meshing

Einziges Variable über alle Surface-Arten: die Karte `Φ`. Sonst EIN Codepfad.

```
MeshFace(face, h_max, tol):
  Φ ← Chart(face.surface)        # Plane: In-Plane-Frame (trivial)
                                 # Cyl/Cone/Extruded: isometrischer Unroll (exakt)
                                 # Sphere: azimutal-äquidistant; NURBS: metrik-skaliert
                                 # geschlossen/periodisch: Revolution-Karte (rim-aligned)
  B ← Φ(frozene Kantenpunkte aller Loops)             # FIXE 2D-Boundary
  h(uv) = min(h_max, surf_maxh(face), √(8·tol·R_min(uv)))   # eq:surf_defl
          R_min(uv) = Φ.curvature_radius(uv)                # Toleranz via Sizing-Feld
  scatter Innen ~ h (greedy graded)
  repeat:                                              # Relax-Insert (eq:disc_centroid)
     2D-Delaunay(B ∪ innen); Innenpunkte → dichtegew. Centroid (ρ=h⁻²), in Loops, B fix
     gröbste-Stelle-Insert: Dreieck mit längster Kante ℓ>h(mid) → mid einfügen
  bis keine Kante>h UND max move<τ
  T ← triangulate_constrained(B ∪ innen, Loops)        # constrained 2D
  lift: p = Φ.to_xyz(uv)                               # exakt auf Developables
  return T (getaggt) + Carrier(face)
```

Kernpunkt: Toleranz geht über das Sizing-Feld ein (`h=√(8·tol·R)`). Ebene: `R=∞ →
h=h_max` (nur Größe). Krümmung: `h` automatisch fein wo es sich biegt; Insert füllt
bis zu diesem krümmungs-bewussten `h` → konstante Sehnen-Deflection (skaleninvariant).
Größe + Toleranz sind EIN Kriterium.

## Algorithmus 2 — Kanten (1D, schon vorhanden via `curve.rs::distribute`)

```
h(s)=min(h_max, √(8·tol·R(s))), R(s)=1/κ(s); n=⌈∫ds/h⌉ Punkte äquidistribuiert ∫1/h
```
Direkte Platzierung = konvergiertes 1D-CVT (gröbste-Stelle-äquiv., kein Spiking).
`tol_edge`/`maxh_edge` verdrahtet. ✓

## Phasenplan

**Phase A — der eine chart-/oracle-getriebene Surface-Pfad**
- A1: `PlaneChart` als `SurfaceChart` (In-Plane-Frame, exakt isometrisch). ✓ committet
- A2: in `surface_sites` EIN chart-getriebener Pfad (Plane/Developable/curved);
  Scatter → Lloyd → `triangulate_constrained` → Lift; planar = Spezialfall. ✓ committet
- A3: Sizing `h(uv)=min(Caps, √(8·tol·R))`, `surf_tol_for`/`surf_maxh_for`. ✓ committet
- A-Kugel: geschlossen → geodätische Icosphere; Cap → Chart aus Facet-Centroid;
  Round-trip-Filter projiziert erst auf die Fläche (Facettierung ≠ Chart-Fehler). ✓ committet
- A4 (offen): gröbste-Stelle-Insert (`ℓ>h`) im Relax-Loop — NUR für gekrümmte (planar
  nicht, sonst bricht die Oversample-Balance des alten Pfades / exakte Volumina).
- A5 (offen): Voll-Revolution-Barrels (Zyl/Kegel/Torus) + Extrudat-Barrels: aktuell
  `revolution_grid`/pinned-Fallback → später nahtloser periodischer Pfad.
- Torus/NURBS (offen): nicht-konvex → kein Hüllen-Trick; near-isometrischer Chart bzw.
  tangential-CVT + restricted Delaunay. Aktuell Chart bzw. pinned-Fallback.
- Verif.: Sphere ✓ (icosphere isotrop+watertight, Cap on-surface≤tol). Cyl/Torus offen.

**Phase B — `mesh_cdt` nutzt die eine Surface**
- `surface_sites` = vollständiger FrozenSurface-Producer (alle Faces, Carrier,
  per-Entität). `mesh_cdt` konsumiert ihn; `frozen_surface`/`surface_mesh`-Patch-Pfad weg.
- Verif.: mesh_cdt Box/Sphere/Cyl watertight + exakt + `per_edge`/`face_maxh`/
  `surface_maxh` grün unter mesh_cdt.

**Phase C — constrained Volumen robust + schnell**
- C1: Octree-beschleunigtes `piercing_edge` (O(Tets)-Scan weg → 509 s).
- C2: Multi-Region-Interface-Recovery (`via`).
- C3: Curved-Recovery robust (`torus` watertight). Ggf. Cavity-Retet statt nur Flip.

**Phase D — umschalten + löschen**
- `mesh_plc_with` → `mesh_cdt`; `cvt::mesh` + `surface_mesh`-Patch-Pfad + RAPIDMESH_CDT-Flag löschen.
- Verif.: conform voll grün; `benchmark.json` vorher/nachher über alle 67.

Reihenfolge: A liefert die einheitliche, per-Entität-aware Surface (Voraussetzung
für B). C macht mesh_cdt über den ganzen Korpus tragfähig. D macht Default + löscht.
Benchmark (67) durchgehend als Gate.

## B2-GATE: GO (2026-06-17)

`mesh_cdt` konsumiert die einheitliche `surface_sites` (B1). Messung via mesh_cdt:
Box 300 Tets/0.02s, UV-Kugel(radial) 909/0.04s, fein 5521/0.27s — alle **Steiner=0**,
watertight, exaktes/≤tol-Volumen, Einzelregion. Die geschlossene gekrümmte Constraint
(Icosphere) wird rein per Flip rekonstruiert (geodätische Seedung → nah-Delaunay →
keine Steiner nötig), in ms statt den befürchteten 509 s. **Tiefste Unsicherheit der
Vision aufgelöst → GO.** Zylinder läuft (1746 Tets), aber Barrel noch nicht
watertight-by-construction (keine Constraint-Tris) → A5. Perf-Befund: der einfache
Spine ist schnell; die 509 s betreffen den vollen 67er-Korpus → C1 misst sich am
Benchmark, nicht an diesen Fällen.

## PIVOT (2026-06-18): Konformität pro Carrier-Typ, keine forcierte gekrümmte Recovery

Befund aus C1/C3-Messung: das einzige harte Problem des neuen Meshers ist die
Oberflächenkonformität. Forcierte gekrümmte Recovery (Edge-Removal/Steiner, TetGen-
Stil) ist der falsche Kampf: der Zylinder-Barrel braucht steiner=4192, weil eine
*willkürliche* gekrümmte Triangulierung erzwungen wird. Die Kugel war steiner=0, weil
ihre Tris schon Delaunay-Flächen waren.

SOTA-Einsicht für UNSEREN Fall (wir erzeugen die Surface selbst, gekrümmt ist ohnehin
tol): Konformität **pro Carrier-Typ** lösen:

- **Planar**: koplanare Punkte auf exaktem `Point3::Pac`-Carrier → die Delaunay tilet
  die Ebene von selbst, Region-Trennung exakt → **bit-exakte rationale Volumina**.
  Recovery uebersprungen (schon heute). Das ist die Staerke, die wir behalten.
- **Gekrümmt** (tol, nie exakt): KEINE forcierte Recovery. Die gekrümmte Boundary IST
  die restricted Delaunay der Surface-Punkte (= Flaechen zwischen Innen/Aussen-Tets,
  via Inside-Oracle/Centroid-Klassifikation, die mesh_cdt schon macht). Bei Unter-
  Sampling verfeinern (Surface-Punkt einfuegen) — das einzige, was sauber sein muss,
  und genau das, was der alte Pfad falsch machte (blind oversamplen statt verfeinern).

Folge: gekrümmte Faces brauchen fuers Volumen nur **Punkte** (an Sizing-Dichte), keine
erzwungene Triangulierung. Die Surface-Tris (Icosphere/Band/Cap) bleiben als Surface-
Level-Referenz + Tagging-Hint, sind aber keine harten Volumen-Constraints mehr. Das
macht Edge-Removal/Cavity-Retet ueberfluessig und vereinfacht A5 (Barrel = Punkte
genuegen) und C2 (gekrümmte Interfaces gleich behandelt).

Watertight: planar als Theorem (exakt), gekrümmt unter adaequatem Sampling + Refinement
(SOTA-Standard, CGAL Mesh_3 / Boissonnat-Oudot). Gekrümmte Carrier waren nie bit-exakt
(float `Point3`), also kein Exaktheits-Verlust.

### C3 neu (ersetzt Edge-Removal/Cavity-Retet):
- **C3a (#108) — FERTIG**: forcierte Recovery aus `tetrahedralize_constrained` raus;
  gekrümmte Boundary = restricted Delaunay (Region-Differenz). Zylinder steiner
  4192→**0**, 87s→**0.18s**; Kugel/Box unveraendert; Lib gruen. 366 Zeilen Recovery-
  Subsystem (flip23/32, recover_facet, piercing_edge, insert_steiner, Steiner-Budget,
  Recover, Vert.carrier) GELOESCHT. Der Pivot ist vollzogen.
- **C3b (#111)**: robustes Tagging der gekrümmten Boundary-Flaechen (per-Punkt
  plc_surface-Id statt nur Input-Tri-Match), damit Output-Surface-Tags stimmen.
- **C3c (#112)**: Straddler-Refinement-Kriterium — eine Boundary-Flaeche mit einem
  Innen-(Volume)-Vertex oder Deflection > tol heisst unter-sampled → Surface-Punkt auf
  dem Carrier einfuegen, neu bauen, iterieren. Nur falls der Korpus Straddler zeigt.

## TERRAIN-KARTE (Diag4, 2026-06-18): mesh_cdt ueber 67 Geometrien

66/67 gemeshed (pipe_junction = vorbestehender CSG-Panic). 57 Volumen: ~24 sauber
(Box/Kugel/Zyl/Torus/Frustum/Prismen/meiste RF: watertight, min-Dih 12-25°, null
Straddler/Sliver, schnell). Drei Lücken aufgedeckt:
1. **Straddler (17 Geom)** auf gekrümmten CSG-Verschneidungen (pipe_cross 9164,
   pressure_vessel 11869, orbs 3457) — restricted-Delaunay-Unter-Sampling. → **C3c nötig**.
2. **Slivers (32 Geom)** breit, auch planar-lastig (perforated_plate 5883, lattice_cube
   2907) und teils watertight. Der alte Pfad hatte optimize.rs-Quality-Pass; mesh_cdt
   nicht. → **Q1 (#117) Quality-Pass** — von der Karte aufgedeckt, war nicht im Plan.
3. **watertight-Metrik über-flaggt Multi-Region** (Triple-Linien sind korrekt). →
   **Diag5 (#118)** Metrik pro Region-Paar.
Fazit: Pivot reif fuer saubere Geometrie, noch nicht korpustauglich — fehlen C3c + Q1,
Metrik muss multi-region-korrekt (Diag5). Alles absehbare Ingenieursarbeit. Artefakt:
`report/validation/benchmark_cdt.json`.

## Roadmap zur Vision (Tasks #102-110, kritischer Pfad)

Ziel: `mesh_cdt` einziger Mesher, alter Pfad geloescht, 67er-Korpus + conform
watertight/exakt bei vertretbarer Perf. Reihenfolge = Risiko zuerst, dann Perf
(macht iterieren bezahlbar), dann Breite, dann Umschalten.

1. **B1 (#102)** — `mesh_cdt` konsumiert `surface_sites` (tris+tri_carrier+per-Entitaet)
   statt `frozen_surface`. Schliesst die Schleife; raeumt Sizing-Blocker ab.
2. **B2 (#103) — GATE** — Box (exakt+watertight) + geschlossene Kugel (<=tol+watertight,
   Steiner beschraenkt). **GO** → weiter. **NO-GO** → Eskalation (Cavity-Retet in cdt3).
   Das billigste Experiment, das die teuerste Unsicherheit (gekruemmte Recovery) aufloest.
3. **C1 (#104) — fertig** — `piercing_edge` lokale Stern-Suche (O(star) statt O(tets),
   Full-Scan-Fallback). Zylinder-Recovery 87s→14s. BEFUND: die Restkosten sind KEIN
   piercing_edge-Problem mehr, sondern **C3** — der Zylinder-Barrel erzeugt steiner=4192
   (= Budget), weil `recover_facet` nur 3-2-Flips kann und das nicht-near-Delaunay-Band
   nicht reinflippt. Die Icosphere war steiner=0 (near-Delaunay). → C3 ist der echte
   Engpass fuer gekruemmte Nicht-Kugel-Flaechen; rueckt VOR C2.
   Design-Fork fuer C3: (A) Recovery staerken (2-3/4-4-Flips, Cavity-Retet) — generell,
   hilft auch Torus; (B) Surface-Tris near-3D-Delaunay erzeugen (z.B. konvexe Huelle
   fuer konvexe Solids) — steiner~0, aber nicht fuer konkav/Torus. (A) ist holistisch.
4. **A5 (#105)** — Barrel-PUNKTE an Sizing-Dichte (Zyl/Kegel/Torus/Extrudat). Nach dem
   Pivot brauchen gekrümmte Faces fuers Volumen nur Punkte; das Tri-Band ist Surface-
   Level/Tagging. revolution_grid liefert die Punkte schon. Cone-Apex/Torus = Punkte
   genuegen ebenfalls. Damit weitgehend durch C3a abgedeckt.
5. **C3a (#108)** — gekrümmte Recovery zum No-Op; Boundary = restricted Delaunay.
   Messen: Zylinder steiner=0/schnell/watertight. DER zentrale Pivot-Schritt.
6. **C3b (#111)** — robustes gekrümmtes Boundary-Tagging (per-Punkt plc_surface).
7. **C2 (#107)** — Multi-Region-Interfaces (`cylinder_via`, `em_scene`); gleiches
   Prinzip (planar forciert-exakt, gekrümmte Interfaces via restricted Delaunay).
8. **C3c (#112)** — Straddler-Refinement-Kriterium (nur falls Korpus es zeigt).
9. **A4 (#106)** — adaptiver groebste-Stelle-Insert (nur gekrümmt). Qualitaet, spaet.
10. **D1 (#109)** — `mesh_plc_with`→`mesh_cdt`; `cvt::mesh` + Patch-Pfad + Flag loeschen.
11. **D2 (#110)** — 67er-Benchmark/Gallery vorher/nachher; Ergebnisse in `report.tex`.

Abhaengigkeiten: B1→B2(GATE)→C1→**C3a**→{C3b, A5-Punkte}→{C2, C3c, A4}→D1→D2.
C3a ist nach dem Pivot der Dreh- und Angelpunkt (loest die gekrümmte Konformitaet).

## Schlüsseldateien

- `crates/rapidmesh-tet/src/cvt.rs` — `mesh` (alt), `mesh_cdt` (neu), `surface_mesh`/
  `frozen_surface` (Patch-Pfad, soll weg), Lloyd-Loops (adaptiv).
- `crates/rapidmesh-tet/src/brep_mesh.rs` — `surface_sites` (brep-basiert, per-Entität;
  planare Tris da, gekrümmte TODO). `SurfaceSites{..,tris,tri_carrier}`.
- `crates/rapidmesh-tet/src/cdt3.rs` — Flip-Recovery + `tetrahedralize_constrained` + `classify_regions`.
- `crates/rapidmesh-tet/src/surfchart.rs` — `SurfaceChart`-Trait + `build_chart` (Cyl/Cone/Sphere/Torus/Extruded). A1: PlaneChart hier.
- `crates/rapidmesh-tet/src/surf2d.rs` — `Cdt`/`triangulate_constrained`/`cvt_fill`.
- `crates/rapidmesh-tet/src/site.rs` — `Carrier`/`Site`/`exact()`.
- `crates/rapidmesh-tet/src/conform.rs` — `MeshParams` (alle Sizing-Felder + Resolver), `mesh_plc_with` (Switch-Punkt).
- `crates/rapidmesh-brep/src/{lib,topology,surface}.rs` — B-Rep + Topologie.
- `report/corpus.py` (Benchmark+Gallery-Quelle), `report/render_gallery.py`, `report/viewer.py`.
- Tests: `crates/rapidmesh-tet/tests/conform.rs` (exakte Volumina, check_structure), `python/tests/test_sizing.py`.

## Bekannte Issues (nicht von dieser Arbeit)

- `pipe_junction` paniке im CSG-Assembly (`scene.rs:669`, „caps only slivers") — vorbestehend.
- `dice` Render-Timeout (Viewer 30 s) — kein Mesh-Problem.
- `mesh_cdt` Perf: `piercing_edge` O(Tets) → Octree (Phase C1).

## Konventionen

- Deutsch antworten; keine Long-Dashes im Report; single-line commits; Claude nicht als Co-Author.
- `cargo -j 4` (CPU drosseln); `maturin develop --release` aus `python/` vor numerischen Python-Läufen.
- Branch `feature/cvt-mesher`, inkrementelle Commits pro Schritt.
