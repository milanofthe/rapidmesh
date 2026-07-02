# rapidmesh — Arbeitsregeln

## Benchmarking

- **Der Corpus ist die einzige Benchmark-Wahrheit.** Jede Mesher-Änderung wird
  gegen den KOMPLETTEN Corpus gebencht: `python report/corpus.py`
  (mesht alle Geometrien, schreibt Metriken, rendert die Gallery).
  Keine Ad-hoc-Test-Suiten, keine Einzelgeometrie-Probes als Beleg.
- Neue Problemgeometrien werden als reguläre Corpus-Einträge aufgenommen
  (`report/validate.py` CASES bzw. `report/corpus.py`), nie als Wegwerf-Beispiele.
- **Jeder Bench-Lauf wird datumsversioniert archiviert**: `bench/history/<datum>_<sha>.json`
  (Quality, Defekte, Timing, git-Metadaten). Der Runner druckt die Trajektorie
  gegen den vorherigen Lauf — Regressionen müssen dort auffallen, nicht später.

## Renders

- Ergebnisse werden IMMER mit gerendert (Normal- + Debug-View mit Defekt-Markern),
  über die bestehende Pipeline: `report/render_gallery.py` bzw. den Render-Teil
  von `report/corpus.py`. Kein zweiter Renderer, keine Parallel-Skripte.
- Debug-View-Marker: Sliver = amber, Straddler = magenta, non-manifold = rot.

## Referenz-Mesher

- gmsh/tetgen-Vergleich über `bench/compare_meshers.py` (gleiche Geometrien,
  gleiches Viewer-Schema).
