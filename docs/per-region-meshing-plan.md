# Per-Region-Volumenvernetzung — Straddler an der Wurzel eliminieren

Architektur-Plan für den Übergang von „ein Whole-Domain-CDT + Centroid-Klassifikation"
zu „N manifolde Per-Region-CDTs, die sich die eingefrorene Interface-Triangulierung
teilen". Ziel: Straddler **per Konstruktion** ausschließen, nicht per Klassifikation
nachträglich flicken.

## 1. Diagnose — warum Straddler heute entstehen

`mesh_cdt` (cvt.rs) macht **eine** Tetraedrisierung der gesamten Multi-Region-Domäne und
klassifiziert danach jeden Tet per Centroid (`domain.region_at(c)`, cvt.rs:614).

Der eigentliche Befund steht in `cdt3::tetrahedralize_constrained` (cdt3.rs:149–155):
**gekrümmte Facetten werden gar nicht erzwungen.** Planare Interfaces werden durch
Koplanarität konform (die Delaunay kachelt eine Ebene von selbst), aber gekrümmte
Interfaces bleiben die *restricted Delaunay der Surface-Punkte* und werden nur downstream
per Region-Differenz extrahiert. Wo ein Tet das gekrümmte Interface durchschneidet, landet
sein Centroid in der falschen Hälfte → **Straddler**. Deshalb sind alle Straddler-Geometrien
gekrümmte Verschneidungs-Körper (bearing, orbs, pipe_cross, rf_dielectric, rf_toroid,
fused_unequal), nie planare.

Der Grund, warum gekrümmte Facetten *nicht* erzwungen werden, ist im Code dokumentiert:
ein erzwungenes gekrümmtes Faceting (ein Zylinder-Barrel-Band) ist nicht near-Delaunay und
bräuchte unbeschränkte Steiner-Punkte im **Non-Manifold-Whole-Domain-Setting**. Genau diese
Schwierigkeit löst der Per-Region-Ansatz auf.

## 2. Zielarchitektur

Die eingefrorene Stage-2-Surface (`FrozenSurface`, cvt.rs:1498) ist bereits die **geteilte**
Interface+Rand-Triangulierung. Jede `SurfaceFace` trägt `regions: [RegionTag; 2]`
(conform.rs:43) — die zwei Regionen, die sie trennt. Damit:

**Phase A — geteilte Surface (existiert schon).** Die Frozen-Surface ist die einzige,
gemeinsame Triangulierung aller Interfaces und Außenränder, region-paar-getaggt.

**Phase B — Per-Region-Randextraktion.** Für jede Region `R > 0`: sammle alle Frozen-Faces,
deren Paar `R` enthält, orientiere jede so, dass die Normale **nach `R` hinein** zeigt. Das
ist die geschlossene, **manifolde** Randfläche `∂R`. (Außenränder haben Paar `(0, R)`,
innere Interfaces `(R, R')` mit beiden ≠ 0.)

**Phase C — Per-Region-Tetraedrisierung mit erzwungenem Rand (das neue Kernstück).** Für
jede Region `R`: seede Innenpunkte in `R` (gefilterte/relaxierte Lloyd-Punkte) und
tetraedrisiere mit `∂R` als **harter Constraint** — inklusive **Boundary-Recovery der
gekrümmten Facetten**. Weil `∂R` ein einzelnes geschlossenes Manifold ist, ist das das
klassische, terminierende „CDT eines Polyeders" (Si/TetGen, Cheng-Dey-Shewchuk
conforming Delaunay): Steiner-Punkte dürfen auf dem Rand eingefügt werden und die
Konvergenz ist für ein Manifold garantiert — die „unbeschränkte Steiner"-Sorge galt nur
dem Erzwingen *eines* nicht-Delaunay-Facetings im Non-Manifold-Komplex.

**Phase D — Stitch.** Interface-Vertices sind über Regionen hinweg **derselbe Frozen-Vertex**
(identische Koordinaten) → ein globales Vertex-Array; jeder Tet behält den Region-Tag seines
Sub-Meshes; Randflächen sind die (bereits region-getaggten) Frozen-Faces. **Keine
Centroid-Klassifikation mehr** — die Region-Tags sind per Konstruktion exakt.

## 3. Warum das Straddler grundsätzlich vermeidet

Ein Straddler ist definitionsgemäß ein Tet, der ein Interface kreuzt. Wird jede Region in
ihrem eigenen geschlossenen Rand vernetzt, kann **kein Tet ein Interface kreuzen** — es gibt
keinen Klassifikationsschritt, der etwas falsch zuordnen könnte. Konformität zwischen
Regionen ist automatisch, weil benachbarte Regionen **dieselben** Interface-Dreiecke teilen.

Zusätzlich löst es die zweite strukturelle Wand der früheren Path-A-Analyse auf: jede Region
ist ein **sauberes Manifold**, also entfällt die Non-Manifold-/Benign-Band-Reconciliation
innerhalb eines Meshes vollständig. Non-Manifold-heit lebt nur noch *zwischen* Regionen und
wird durchs *Teilen* der Triangulierung erledigt.

## 4. Der ehrliche harte Kern — Interface-Steiner-Sharing

Fügt die Recovery von Region `R` einen Steiner-Punkt **auf einem geteilten Interface-Face**
ein, muss der Nachbar `R'` denselben Punkt bekommen, sonst konformieren die zwei Meshes auf
dem Interface nicht.

**Lösung: Boundary-Recovery als geteilte Vorab-Phase auf der Frozen-Surface.** Refine/recover
die Surface-Triangulierung **global zuerst** (füge nötige Steiner-Punkte in die geteilte
`FrozenSurface` ein, aktualisiere beide Region-Sichten), sodass die Surface danach von beiden
Seiten „as-is tetraedrisierbar" ist. Die Per-Region-Innen-Tetraedrisierung braucht dann
keinen weiteren Rand-Steiner. In der Praxis ist die Frozen-Surface bereits eine
sizing-graded near-Delaunay-Triangulierung → die meisten Geometrien brauchen **null**
Interface-Steiner; nur die harten gekrümmten Creases (bearing/orbs) brauchen ihn.

Dies ist genau der WP, der vorher als „Curved Facet-Recovery in cdt3" (#34) und
„TetGen-vollständige fillcavity-Recovery" (#41) notiert war — jetzt aber im **gut gestellten
Manifold-Per-Region-Rahmen**, wo er terminiert, statt im Non-Manifold-Whole-Domain-Rahmen,
wo er divergierte.

## 5. Degradationsverhalten (bewusst)

Centroid-Klassifikation degradiert „weich": bei unvollständiger Recovery liefert sie ein
best-effort-Mesh *mit* Straddlern. Per-Region ist prinzipientreuer, aber strenger: scheitert
die Recovery einer Region, scheitert *diese Region* hart. **Mitigation:** pro Region ein
Fallback auf den heutigen Whole-Domain-Classify-Pfad, falls die Recovery nicht konvergiert —
so ist Per-Region strikt besser auf der sauberen Mehrheit und nie schlechter als heute auf
den harten ~6.

## 6. Implementierungs-Schritte (konkret, am Code)

1. **`fn region_boundaries(fs: &FrozenSurface) -> Vec<RegionBoundary>`** (neue Datei
   `crates/rapidmesh-tet/src/region_split.rs`): pro Region `R` die nach innen orientierten
   Frozen-Faces + ihre Vertex-Menge + Carrier. Reine Funktion über die vorhandenen Tags.
2. **Per-Region-Innenpunkte:** die existierende Lloyd-Relaxation (`mesh_cdt` Stage 3)
   liefert schon Innenpunkte; pro Region via `domain.region_at` filtern (Übergangslösung)
   oder die Relaxation pro Region laufen lassen (sauberer, parallelisierbar).
3. **`fn tetrahedralize_region(boundary, interior, lo, hi) -> Constrained`** (Erweiterung
   von cdt3): wie `tetrahedralize_constrained`, aber mit **echtem Boundary-Recovery** für
   die gekrümmten Facetten (Flip-/Cavity-Retetraedrisierung auf Manifold-Input; die alte
   `recover.rs`-Idee, jetzt im wohlgestellten Rahmen).
4. **Shared-Steiner-Vorabphase** (Phase 4 nur falls nötig): globale Surface-Recovery vor den
   Per-Region-Läufen.
5. **`fn stitch_regions(Vec<Constrained>, fs) -> TetMesh`**: Weld an geteilten
   Frozen-Vertices, Region-Tags aus dem Sub-Mesh, Randflächen aus den Frozen-Faces.
6. **Verdrahtung:** neuer `mesh_cdt`-Zweig (oder Ersatz), Default-Auswahl wie gehabt über
   `method`. Centroid-Klassifikation entfällt im neuen Pfad.

## 7. Spike (Validierung vor Vollausbau)

- **Schritt 1 — planar:** `stacked_two_region` (zwei Regionen, planares Interface). Beweist
  Split + Stitch konform + straddler-frei, **ohne** dass Recovery nötig ist (Koplanarität).
- **Schritt 2 — gekrümmt:** `nested_spheres` (Kugel-in-Kugel, gekrümmtes Interface). Der echte
  Test des Recovery-Kernstücks. Erfolgskriterium: Straddler = 0, watertight = Y, geteilte
  Interface-Vertices, Qualität ≥ Centroid-Baseline.
- **Schritt 3 — hart:** `bearing` / `orbs`. Zeigt, ob die Manifold-Recovery die Creases
  schafft, wo die Whole-Domain-Recovery divergierte.

## 8. Erwarteter Gewinn

Auf der sauberen Mehrheit: straddler-frei + manifold + konform per Konstruktion, plus die
Qualitäts- und Sliver-Vorteile des CDT-Pfads bleiben. Auf den harten gekrümmten Körpern:
mindestens so gut wie heute (Fallback), im Erfolgsfall straddler-frei wo es heute 15–278
Straddler sind. Und der ganze Non-Manifold-Reconciliation-Komplex entfällt.
