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


def test_step_read_once_bodies_one_by_one():
    step = rm.read_step(STEP / "assembly.step")
    assert step.metres_per_unit == 1e-3
    assert sorted(step.names) == ["base", "boss", "plate"]
    g = rm.Geometry(maxh=1.5)
    plate = g.add_body(step, step.names.index("plate"))
    g.add_body(step, step.names.index("boss"), void=True)
    m = g.mesh()
    assert set(m.tet_regions.tolist()) == {plate.region}
    assert m.diagnostics["watertight"]
