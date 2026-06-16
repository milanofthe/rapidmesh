# rapidmesh-brep: a boundary-representation layer

> Detaillierte Spec. Source of Truth bleibt der Code; dieses Dokument leitet jede
> Datenstruktur und jeden Builder-Schritt aus dem ab, was `rapidmesh-csg` /
> `rapidmesh-geom` heute tatsaechlich produzieren.

## 1. Warum (das Problem an der Wurzel)

Der Mesher konsumiert heute den **`TaggedPlc`** -- eine wasserdichte
Dreieckssuppe mit Tags:

```
TaggedPlc {
  vertices:      Vec<[f64;3]>,
  triangles:     Vec<[u32;3]>,
  surface_refs:  Vec<SurfaceRef>,        // -> surfaces[i]: SurfaceKind
  region_tags:   Vec<[RegionTag;2]>,     // [front, back] Material je Dreieck
  face_tags:     Vec<FaceTag>,           // Port/PEC/ABC
  surfaces:      Vec<SurfaceKind>,       // Plane|Cylinder|Sphere|Cone|Torus|Extruded
  surface_owners:Vec<u32>,               // Szenen-Solid je Surface (SHEET_OWNER fuer Sheets)
}
```

Diese Facettierung ist ein Artefakt der CSG-Arrangement-Stufe **und** der
Eingangstesselierung (`n_seg`). Sie zu meshen heisst, gegen das Artefakt zu
kaempfen -- und genau das tut `cvt.rs` heute mit einer Reihe von Hilfskonstrukten:

- **`feature_edge_chains`** rekonstruiert Kantenzuege aus den Patch-Randkanten,
  splittet an Ecken (Turn > 45 deg).
- **`ExtrudedEdgeCurve`** holt die analytische Profilkurve zurueck, weil eine aus
  der Facettierung resampelte Polylinie das Eingangs-`n_seg` erbt statt der
  echten Kruemmung.
- **`chart_groups`** gruppiert gekruemmte Facetten und testet Bijektivitaet/Seam
  -- geschlossene/umlaufende Flaechen (Kugel, voller Zylinder) fallen auf den
  facettierten Pfad zurueck.
- **`rep[]` Union-Find** kollabiert die stumpfe Hinterkante (zwei fast-koinzidente
  Profilendpunkte) auf einen Eckpunkt.
- Die Flaechenkonformitaet entsteht **implizit** ueber "Oberflaeche feiner seeden
  als Volumen + restricted Delaunay rekonstruiert den Rand". Bei einem duennen
  Profil (Airfoil-Nase/Hinterkante) liegen Ober- und Unterseite naeher als ein
  Element zusammen -> der restricted Delaunay kann sie nicht trennen -> ~300
  nicht-geschlossene Kanten an der Vorderkante.

Ein **B-Rep** entfernt das Artefakt an der Wurzel: Flaechen sind getrimmte
analytische Surfaces, Kanten sind analytische Kurven (inkl. der neuen
Schnittkurven aus den Booleans), Vertices sind Punkte. Der Mesher meshed *aus der
Geometrie* -- verteilt auf jeder Kantenkurve, meshed jede getrimmte Flaeche in
ihrem (u,v)-Parameterraum, fuellt das Volumen -- unabhaengig von jeder
Eingangstesselierung. Das ist exakt der Plan "NURBS als allgemeine Kanten und
Flaechen konsumieren".

## 2. SOTA (recherchiert)

- **Half-edge / DCEL** (truck, Fornjot): kompakt, elegant, aber **nur
  2-mannigfaltig** -- kann kein Multi-Material-Interface (Kante, an der 3+
  Materialien treffen) und kein eingebettetes Sheet darstellen. rapidmeshs
  Kerndomaene ist genau multi-material.
- **Radial-edge (Weiler / "NMG")**: fuer **nicht-mannigfaltige** Geometrie gebaut
  -- eine Kante verlinkt radial ALLE Flaechen entlang ihr; jede Flaeche traegt
  front/back-Materiallabels. Das ist die richtige Topologie.
- **OpenCASCADE `TopoDS`** (industrieller SOTA): saubere **Topologie (+) Geometrie**
  -Trennung; der Schluessel ist die **PCurve** -- die Kante als parametrische
  Kurve im (u,v)-Raum jeder Nachbarflaeche, was das Meshen einer getrimmten
  Flaeche im Parameterraum wohldefiniert macht.
- "Topology-First B-Rep Meshing" (arXiv): Topologie treibt das Mesh, Geometrie
  folgt -- passt zu unserem Bottom-up-Ansatz.

## 3. Die Datenstruktur

Geometrie und Topologie getrennt (OpenCASCADE-Stil), Topologie nicht-mannigfaltig
(Weiler), Arena-allokiert (Vec + id-newtypes), gebaut FROM dem exakten
CSG-Arrangement.

### 3.1 Geometrie (analytisch)

```
Surface = SurfaceKind     // wiederverwendet aus rapidmesh-geom (Plane..Extruded)
Curve   = Line { p0, dir }
        | Circle { center, axis, radius, x }
        | Nurbs { curve: Arc<NurbsCurve>, t:[f64;2] }
        | Intersection { a: SurfaceId, b: SurfaceId }   // LAZY
PCurve  = { face: FaceId, uv: Vec<[f64;2]> }            // Kante im (u,v) der Flaeche
```

**Drei Entscheidungen, die das elegant halten:**

1. **Lazy `Curve::Intersection{a,b}`.** Eine Schnittkante ist nur "Surface A trifft
   Surface B"; der Mesher projiziert bei Bedarf auf A∩B (alternierende
   Projektion / Newton, wiederverwendet `project::closest_on_surface`). Fuer die
   haeufigen Faelle wird die geschlossene Form *erkannt* und gespeichert, damit
   die Kruemmung exakt ist:
   - `Plane ∩ Plane` -> `Line`
   - `Plane ∩ Sphere`, `Plane⊥axis ∩ Cylinder/Cone` -> `Circle`
   - `Plane⊥axis ∩ Extruded` -> die Profil-`Nurbs` bei fester Hoehe (heutiger
     `ExtrudedEdgeCurve`)
   - `Plane‖axis ∩ Extruded/Cylinder` -> `Line` (Extrusions-Ruling)
   - alles andere -> `Intersection{a,b}`, Punkte per Dual-Projektion einer
     facettierten Startpolylinie; Kruemmung diskret (osculating circle, wie
     `PolylineCurve`).
   So vermeiden wir ein schweres allgemeines Schnittkurven-Subsystem und decken
   trotzdem jede heutige Primitive exakt ab.

2. **PCurve pro Flaeche.** Eine getrimmte Flaeche wird in ihrem eigenen
   (u,v)-Raum gemeshed (2D-Lloyd, Trim-Loops als Rand), dann geliftet. Das fixt
   die Airfoil-Vorder-/Hinterkante: die Flaeche ist *per Konstruktion* getrimmt,
   keine restricted-Delaunay-Inferenz. Vorder-/Hinterkante sind Loop-Ecken im
   (t,h)-Raum, keine fast-koinzidenten 3D-Vertices.

3. **Built FROM dem exakten CSG-Arrangement.** Die Dreieckssuppe gibt die exakte
   TOPOLOGIE (welche Surfaces treffen, Regionslabels, Vertex-Positionen); der
   B-Rep legt analytische Kantenkurven + Trim-Loops obendrauf. Das exakte CSG
   bleibt Source of Truth -> Exaktheit erhalten.

### 3.2 Topologie (nicht-mannigfaltig, Weiler radial-edge)

```
Vertex   { point: Point3, pos: [f64;3] }              // exakte Ecke
Edge     { curve: Curve, ends:[VertexId;2], t:[f64;2], radial: Vec<HalfEdgeId> }
HalfEdge { edge, forward: bool, loop_: LoopId, pcurve: PCurve }   // ein "co-edge"
Loop     { coedges: Vec<HalfEdgeId>, face: FaceId }   // outer + holes, orientiert
Face     { surface: SurfaceId, loops: Vec<LoopId>, regions:[RegionTag;2] }
Shell    { faces: Vec<FaceId> }
Region   { shells: Vec<ShellId>, tag: RegionTag }
Brep     { vertices, edges, halfedges, loops, faces, shells, regions, surfaces }
```

- **Edge.radial**: ALLE Verwendungen (HalfEdges) der Kante -- bei einer
  Box-Kante 2, an einem 3-Material-Interface 3+, an einem eingebetteten Sheet-Rand
  beliebig. Das ist die Nicht-Mannigfaltigkeit.
- **Face.regions = [front, back]**: kommt 1:1 aus `TaggedPlc.region_tags`. Front =
  Seite, in die die Flaechennormale zeigt. **Sheet** = Face mit `front == back`
  (beide nichtnull, dieselbe Region); **Aussenwand** = eine Seite ist `RegionTag(0)`
  (Hintergrund).
- **periodisch**: ergaenzend `Surface`/`Face` traegt `period: [Option<f64>;2]`
  (Zylinder/Kugel/Torus umlaufend). Der Face-Mesher kachelt den periodischen
  Streifen statt am Seam zu zerreissen -- loest den `chart_groups`-Seam-Fallback.

## 4. Der Builder: `TaggedPlc -> Brep`

Reine Funktion `brep::from_plc(&TaggedPlc) -> Brep`. Kein CSG-Umbau; der
STEP-Importpfad (der ebenfalls auf `TaggedPlc` konvergiert) ist automatisch
abgedeckt. Schritte:

**B1 -- Faces gruppieren.** Verbundene Komponenten der Dreiecke unter dem
Schluessel `(surface_ref, {region_lo, region_hi}, face_tag)`. Das ist exakt der
`chart_groups`/`build_patches`-Key von heute, vereinheitlicht: planare Patches
UND gekruemmte Gruppen werden zum selben `Face`-Typ. Jede Komponente -> eine
`Face` mit `surface`, `regions`.

**B2 -- Vertices.** Die PLC-Vertices sind bereits global dedupliziert. Ecken
(`Vertex`) sind die Endpunkte der Kantenzuege (B3); innere Facetten-Vertices sind
keine B-Rep-Vertices. `Vertex.point` = exakter `Point3` (aus dem VertexPool, wo
verfuegbar; sonst `Point3::Explicit`).

**B3 -- Edges + Kurven.** Randkanten je Face-Gruppe (Kanten, die nur einer Facette
der Gruppe gehoeren), ueber Gruppen hinweg gesammelt. Eine B-Rep-Kante = maximaler
Zug solcher Kanten zwischen zwei Ecken, wobei eine Ecke entsteht bei: Grad != 2,
Turn > Schwelle, ODER Wechsel des angrenzenden Surface-Paars. Das ist
`feature_edge_chains`, plus das Surface-Paar-Kriterium. Pro Kante:
- die zwei (oder mehr) angrenzenden Surfaces bestimmen -> `Curve` via der
  Erkennungstabelle aus 3.1(1);
- `ends` = die zwei Ecken; `t` = Parameterbereich der Kurve;
- die fast-geschlossene Kette (stumpfe Hinterkante) wird auf eine Ecke gemerged
  (heutiges `rep[]`), jetzt sauber als Topologie-Operation statt PLC-Weld.

**B4 -- Loops + PCurves.** Pro Face die Randkanten in orientierte Zyklen ordnen
(outer + holes), Orientierung aus der Facettennormale. Pro Co-Edge die `PCurve`:
die Kantenpunkte via `surfchart::to_uv` in den (u,v)-Raum DIESER Flaeche
abgebildet. (Analytische PCurves koennen die gesampelte spaeter ersetzen.)

**B5 -- Radial-Verlinkung.** Pro Kante alle HalfEdges (eine je angrenzendem Face)
in `Edge.radial`. Nicht-mannigfaltige Interfaces fallen hier automatisch heraus.

**B6 -- Shells + Regions.** Faces, die eine Region begrenzen, zu Shells gruppieren
(eine Region kann mehrere Shells haben: Aussenhaut + Hohlraum). `Region.tag` aus
den Region-Tags. Region 0 (Hintergrund) wird nicht materialisiert.

**Exaktheit:** Der Builder snappt nichts. Vertex-Positionen, Regionslabels und die
Inzidenz kommen unveraendert aus dem exakten Arrangement; nur analytische Kurven +
Trim-Loops kommen oben drauf.

## 5. Param-Space-Charts + getrimmtes Face-Meshing

Wiederverwendet `surfchart.rs` (`to_uv`/`to_xyz`/`curvature_radius` je
`SurfaceKind`), generalisiert um Periodizitaet. (u,v) pro Kind:

| SurfaceKind | (u,v) | Periodisch | Singularitaet |
|---|---|---|---|
| Plane | Frame-2D (drop-axis) | -- | -- |
| Cylinder | (theta, z) | u: 2*pi*R | -- |
| Cone | (theta, s) | u | Apex |
| Sphere | (theta, phi) | u | Pole |
| Torus | (theta_maj, theta_min) | u, v | -- |
| Extruded | (t_profil, h) | -- | -- |

Der Face-Mesher:
1. Loop-PCurves bilden ein getrimmtes Gebiet in (u,v).
2. `surf2d::cvt_fill` fuellt es (Loops als Rand, Size-Field als Target) -- der
   bestehende 2D-Lloyd.
3. Jedes (u,v) wird via `to_xyz` nach 3D geliftet.
4. **Planare Flaeche**: Lift auf die EXAKTE Ebene (`Site::on_plane`, `Pac`/`Lnc`
   Carrier) -> in-plane-Moves erhalten das exakte Regionsvolumen (der
   Planar-Konformitaets-Gate der Tests bleibt gruen). Gekruemmte Flaeche:
   `Site::on_surface`, float on-surface (Toleranz-Fixtures).
5. **Periodisch/umlaufend**: der Streifen wird mit periodischem Rand gekachelt
   statt am Seam abzubrechen -> Kugel/voller Zylinder werden ECHT gemeshed, kein
   facettierter Fallback mehr.

**Konformitaet zwischen Nachbarflaechen** entsteht jetzt *per Konstruktion*: eine
geteilte Kante wird EINMAL verteilt (1D `curve::distribute` auf der analytischen
`Curve`); beide Faces konsumieren dieselben 3D-Kantenpunkte als Loop-Rand (jeweils
durch ihr eigenes `to_uv` abgebildet). Die geteilten Randknoten fallen in 3D exakt
zusammen -> wasserdicht ohne Oversampling.

## 6. Wie der Mesher den B-Rep konsumiert (chirurgisch, Stufen bleiben)

Die zentrale Erkenntnis: **der B-Rep aendert nur die QUELLE der
Oberflaechenpunkte.** Volumen-Lloyd, Regionsklassifikation und
restricted-Delaunay-Extraktion bleiben unveraendert.

```
brep = brep::from_plc(plc)
Stufe 1 (Kanten):  fuer jede Edge -> curve::distribute(&Curve, deflection, cap, grad)
                   -> geteilte Kantenpunkte (1x, beide Faces teilen sie)
Stufe 2 (Faces):   fuer jede Face -> getrimmtes (u,v)-Meshing (Abschn. 5)
                   -> Oberflaechenpunkte EXAKT auf der Surface, konform ueber Kanten
Stufe 3+ (Volumen):UNVERAENDERT -- vol_field aus Oberflaechenpunkten, 3D-Lloyd,
                   region_at-Klassifikation, restricted-Delaunay-Rand-Extraktion
```

- Warum die Volumenstufe unveraendert bleibt: Die getrimmten Faces liefern GUTE,
  konforme, dichte Oberflaechenpunkte (exakt auf der Surface). Der bestehende
  Volumen-Delaunay + die restricted-Extraktion rekonstruieren daraus den Rand --
  sie scheitern heute NUR, weil die chart_group-Punkte an der duennen Nase
  spaerlich/nicht-konform sind. Mit korrektem Param-Space-Meshing sind sie dicht
  und konform -> die Extraktion greift.
- `feature_edges()` und das Output-Tagging (`surface`/`face_tag`/owner) kommen
  jetzt direkt aus dem `Brep` statt aus `point_tile` -- sauberer.

Integration: `rapidmesh-tet` bekommt eine Dep auf `rapidmesh-brep`; `cvt::mesh`
ruft `from_plc` und ersetzt die Bloecke `feature_edge_chains` /
`ExtrudedEdgeCurve` / `chart_groups` / planar-patch-fill durch die zwei B-Rep-
getriebenen Stufen 1+2. Die `Site`-Carrier (exakt planar / float curved) bleiben.

## 7. Eingebettete Sheets funktionieren bereits NATIV

Wichtige Korrektur: In der Bottom-up-Architektur sind eingebettete Sheets KEIN
Sonderfall. Beleg: `tests/conform.rs::face_maxh_refines_tagged_sheet` (ein
`sheet_rect`/`FaceTag(7)` in einer Box-Region) laeuft gruen -- `n_sheet > 20`,
`check_structure` passt. Der Mechanismus ist vollstaendig vorhanden:

1. Ein Sheet ist nur eine weitere Face -> Stufe 2 seedet seine Punkte FIX auf der
   Ebene (`Site::on_plane`), genau wie jede Aussenwand.
2. Der Volumen-Lloyd haelt beidseitig Abstand (die generische Clearance
   `vol_field.at(p)` aus `cvt.rs` -- ein Volumen-Seed innerhalb einer lokalen
   Groesse vom Sheet wuerde ein Tet erzeugen, das durch das Sheet greift; die
   Clearance verhindert das auf BEIDEN Seiten, weil sie den naechsten
   Oberflaechenpunkt misst, egal welche Seite).
3. Der restricted Delaunay rekonstruiert die dichte ebene Punktschicht als
   Tet-Faces (eine dichte planare Schicht zwingt Tets, sich anzuschmiegen).
4. Die Extraktion emittiert sie bereits: der `else if ra != 0`-Zweig in
   `cvt::mesh` (Sheet = gleiche Region beidseitig + getaggt + `patch_of_face`).

Der B-Rep aendert daran nichts Grundsaetzliches: eine Sheet-Face ist
`regions == [r, r]` (front == back). Sie laeuft durch dieselben Stufen 1+2 wie
jede Face, der getrimmte Param-Space-Mesher macht sie nur SAUBERER (analytisch
getrimmter Rand statt facettiert).

### Was bei `air_dielectric_pec` tatsaechlich offen ist (nicht "Sheets")

Das ignorierte Fixture ist NICHT am Sheet-Konzept gescheitert, sondern an einer
**geometrischen Koinzidenz**: eines seiner Sheets liegt bei `z = 2.0` -- exakt
koplanar mit dem air/diel-Regionsinterface (der Oberkante der Dielektrikum-Box).
An dieser Stelle faellt die Tet-Face in den `ra != rb`-Interface-Zweig, traegt
also das Regionspaar `[air, diel]`, waehrend der Sheet-`FaceTag` verloren geht --
eine `SurfaceFace` traegt heute nur EIN Label. Im B-Rep ist das sauber: an der
koinzidenten Stelle ist es EINE Face, die BEIDES traegt (`regions = [air, diel]`
als Interface UND den Sheet-`face_tag`); die radial-edge-Topologie fuehrt mehrere
Labels an einem Ort. Das ist der eigentliche (kleine) Rest, nicht die
Sheet-Faehigkeit.

## 8. Die wirklich harten Teile (ehrlich benannt)

- **Schnittkurven-Projektion** fuer `Intersection{a,b}`: Robustheit der
  alternierenden Projektion an flachen Schnittwinkeln. Mitigation: facettierte
  Startpolylinie als Initialschaetzung; bei Nichtkonvergenz Fallback auf die
  `PolylineCurve`.
- **Periodische Seams**: korrekte Branch-Cut-Wahl, sodass das getrimmte Gebiet
  nicht ueber den Seam laeuft (oder der periodische Streifen sauber kachelt).
- **Exakter Planar-Gate**: planare Faces MUESSEN exakte on-plane-Carrier behalten,
  sonst brechen die rationalen Volumen-Fixtures. Test zuerst.

## 9. Build-Plan (inkrementell, Tests am Ende -- "wir bauen, dann testen")

1. **Builder** `from_plc` (B1-B6): `Brep` mit Vertices, Edges+Curves (Erkennung +
   Lazy-Intersection), Loops+PCurves, Radial, Regions. Unit-Tests: Box (12 Edges,
   6 Faces, 8 Verts), Sphere (1 Surface periodisch), Airfoil (Extruded-Profil-Edge).
2. **Param-Charts**: `surfchart` um Periodizitaet/Pole generalisieren; PCurve-
   Lifting mit exaktem on-plane-Carrier fuer Ebenen.
3. **Getrimmter Face-Mesher** (Abschn. 5): ersetzt chart_group + planar-patch-fill.
4. **Verdrahten**: `cvt::mesh` auf B-Rep-Stufen 1+2 umstellen, Stufe 3+
   wiederverwenden. Korpus + `conform.rs` gruen halten (planar exakt zuerst).
5. **Sheets reaktivieren** (`front==back`-Faces): `air_dielectric`,
   `surface_owners` Fixtures.

Nach jedem Schritt: Landing-Page-Showcase neu generieren (Nutzer sieht
Fortschritt), `cargo test -p rapidmesh-tet -j 4`, `maturin develop --release` vor
Python-Laeufen.
