# Path A (Recovery + Flood + Frozen-Faces) — Validierungsbericht

Exhaustive Validierung des „force the frozen surface as boundary"-Ansatzes (Path A)
für straddler-freie Volumen-Meshes. Ergebnis: **zwei unabhängige strukturelle Wände**;
Path A ist als sauberer Default nicht erreichbar ohne ein noch fehlendes Teilstück.

## Committet & validiert (echter Wert)

- **Stufe 0 — Konformitäts-Metrik** (`n_nonconformal_faces`, diagnostics.rs): treuer
  Nordstern, `missing + extra` gegen die gefrorenen `surf_faces`. Keine False-Zeros
  (jede Straddler-Geometrie hat nc>0). Über Python/corpus exponiert.
- **Stufe 1.1+1.2 — Facet Recovery** (recover.rs): pierced-tet Cavity (trennt benigne
  Band-Flips von echten Blockern, 660× schneller als der alte Vertex-Star-Ball) +
  bounded gezielte Verify-Enlarge (giftwrap meldet die stuck face, absorbiere den Tet
  dahinter). Panic-frei, korruptionsfrei (`try_replace_cavity` validiert jeden Swap
  exakt). **Erfolgsrate (FROZEN+FLOOD+RECOVER):**
  - `failed=0`: cylinder, rf_toroid_core, rocket, chain
  - Teil-recovert: fused_unequal(47/4), orbs(736/142), mold_block(45/34),
    bearing(173/469), rf_dielectric(172/326)

## Wand 1 — Vollständige Recovery ist research-hard

Die residualen `failed` sind `giftwrap_none` (Gift-Wrap ist in 3D beweisbar unvollständig,
George & Borouchaki). Zwei robustere Primitive versucht, beide ungenügend:

- **Steiner-Split** (Diazzi-Stil, Punkt auf dem Carrier): **kaskadiert** — mold_block
  failed 34→371, surf_tris 1201→18562, watertight kaputt. Exakt die „unbounded Steiner
  für gekrümmte Bänder"-Warnung.
- **Split-Pointset Delaunay + fillcavity** (TetGen `delaunizecavity`): zweimal gebaut
  (inkl. Konvergenz-Fix `missing` gegen volle DT statt getrimmten Rand). **Divergiert
  empirisch** (Trace: missing 2→33 bei wachsender Cavity). Wurzel: meine pierced-Cavity
  und die Split-by-Plane-Rekonstruktion sind geometrisch inkonsistent (die Facette ist
  nicht der „Äquator" der Cavity). TetGen vermeidet das mit Crossing-Tet-Cavity +
  Single-DT mit markierter Missing-Region — ein größerer, anderer Cavity-Aufbau.

→ Vollständige robuste Recovery = TetGens komplette `formcavity`/`delaunizecavity`/
`fillcavity`/`carvecavity`-Pipeline (>1000 Zeilen, viele Sonderfälle). Eigenes WP.

## Wand 2 — Benign-Band-Flood-Leak (unabhängig von Recovery)

Selbst mit `recovery_failed==0` regrediert Flood+Frozen als Default:
**watertight 47→32, min_dih kollabiert breit (cylinder 14.41→0.25, cone 0.65→0).**

Wurzel definitiv validiert: Eine **benigne Band-Facette** (koplanares Quad, Diagonal-Wahl)
ist *keine* Tet-Fläche — Recovery lässt sie korrekt in Ruhe (Erzwingen → flacher Sliver).
Der Flood-Oracle kennt aber nur Frozen-Faces als Wände; die benigne Frozen-Diagonale ist
keine Tet-Fläche → das Quad hat **keine Wand** → Flood leckt → falsche Tets behalten →
Slivers + non-watertight. `recovery_failed==0` garantiert also *nicht* Flood-Exaktheit.

→ Flood+Frozen wird nur exakt, wenn der Wand-Satz zur *tatsächlichen* Volumen-
Triangulierung **rekonziliert** wird (benigne Frozen-Diagonale durch die Volumen-Diagonale
ersetzen). Dieses **„Wall-Reconciliation"-Teilstück fehlt** — es ist der saubere Fix für
Wand 2.

## Validierte Schlussfolgerung

Path A braucht ZWEI noch fehlende Teilstücke: (1) TetGen-vollständige `fillcavity`-Recovery,
(2) Wall-Reconciliation für benigne Bänder. Beide sind machbar, aber substanziell.

Die Recherche legte für einen Oracle-Mesher **Path B** nahe (Restricted-Delaunay-Rand
direkt aus dem Oracle): er umgeht *beide* Wände, weil der Rand aus dem Volumen abgeleitet
statt mit einer separaten gefrorenen Surface rekonziliert wird. Strategische Entscheidung
für den Nutzer.

## Nachtrag — Wand 2 gelöst (Wall-Reconciliation gebaut)

Wand 2 ist **behoben**: `reconcile_benign_bands` (cvt.rs) ersetzt eine benigne Frozen-
Diagonale durch die tatsächliche Volumen-Diagonale (beide Tet-Flächen vorhanden) — gleiche
Geometrie, gleiche Region, kein Sliver. Damit ist jede benigne Band-Facette eine Tet-Fläche,
der Flood leckt dort nicht mehr.

**Opt-in Path-A-Modus** (`RAPIDMESH_PATHA`): Recovery + Reconciliation + Flood + Frozen-
Boundary, aktiv pro Geometrie nur wenn *jede* Frozen-Facette eine Tet-Fläche ist. Validiert
(compare_base, gegen Baseline watertight 47 / straddler-free 61):

- **Gefixt (watertight + straddler-frei):** fused_unequal (15→0), rf_toroid_core (137→0),
  chain (10→0). Verbessert: pipe_cross 278→249, rf_dielectric 158→124, bearing 218→192.
- watertight bleibt 47, straddler-free 61→63.
- **Default unverändert** (ohne `RAPIDMESH_PATHA` exakt die alte Baseline).

**Verbleibend bis Path-A-Default:**
1. **Boundary-Slivers** (Stufe 4): einige gekrümmte Körper bekommen unter Path A min_dih→0
   (randfeste Slivers; cylinder 14.41→0). Klassisches Fixed-Boundary-Sliver-Problem.
2. **Non-manifold Reconciliation** auf wenigen Kreuzungs-Körpern (cross_cyl, frustum,
   union_box_cyl, cyl_coarse_interior verlieren watertight) — Reconciliation an Region-
   Interfaces verfeinern.
3. **WP #41** (fillcavity) für die unrecoverten Creases (orbs/bearing-Residuen).
