"""STEP import through the Python API."""
from pathlib import Path

import rapidmesh as rm

STEP = Path(__file__).resolve().parents[2] / "crates" / "rapidmesh-step" / "fixtures"


def test_step_assembly_gives_one_solid_per_body():
    g = rm.Geometry(maxh=1.5)
    solids = g.import_step(STEP / "assembly.step")
    assert len(solids) == 3
    m = g.mesh()
    assert sorted(set(m.tet_regions.tolist())) == sorted(s.region for s in solids)
    assert m.diagnostics["watertight"]
