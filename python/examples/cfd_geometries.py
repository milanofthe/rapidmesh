"""CFD geometries: flow domains with the refinement a solver needs near walls
and in wakes. The fluid is the meshed region; bodies in the flow are voids
unless the case is conjugate (heat sink), where the solid is a second region.

    python python/examples/cfd_geometries.py
"""

import math

import rapidmesh as rm


def naca_wing_tunnel() -> rm.Mesh:
    """A finite NACA 0012 wing (chord 1, span 2) in a wind tunnel section,
    tip inside the tunnel so the tip vortex region is resolved, wake refined
    behind the trailing edge."""
    c = 1.0
    g = rm.Geometry(maxh=0.35)
    g.label(g.box(8 * c, 4 * c, 3 * c, position=(-2 * c, -2 * c, -1.5 * c)), "fluid")
    wing = g.airfoil_naca0012(c, 2 * c, position=(0, 0, -1.5 * c), span_axis=(0, 0, 1), void=True)
    g.refine_surface(wing, 0.03 * c)
    wake = [(c + 0.5 * k * c, 0.0, z) for k in range(1, 7) for z in (-1.0, 0.0, 0.5)]
    g.refine_near_points(wake, [0.05 * c * (1 + 0.5 * (i // 3)) for i in range(len(wake))])
    return g.mesh()


def dfg_cylinder() -> rm.Mesh:
    """The 3D DFG benchmark channel (Schaefer and Turek, 3D-2Z): a cylinder of
    diameter 0.1 spanning a 2.5 x 0.41 x 0.41 channel at (0.5, 0.2), fine
    around the cylinder and along its wake."""
    H, L, D = 0.41, 2.5, 0.1
    g = rm.Geometry(maxh=0.04)
    g.label(g.box(L, H, H), "fluid")
    cyl = g.cylinder(D / 2, H, position=(0.5, 0.2, 0.0), segments=48, void=True)
    g.refine_surface(cyl, 0.008)
    g.refine_near_points([(0.5 + 0.1 * k, 0.2, 0.2) for k in range(1, 9)], 0.015)
    return g.mesh(grading=0.3)


def backward_facing_step() -> rm.Mesh:
    """Backward-facing step: an inlet channel of height 1 expanding to height
    2, span 4, refined along the step edge where the shear layer separates."""
    g = rm.Geometry(maxh=0.3)
    inlet = g.box(4.5, 4.0, 1.0, position=(-4.0, 0.0, 1.0))
    g.label(g.union(inlet, g.box(16.0, 4.0, 2.0, position=(0.0, 0.0, 0.0))), "fluid")
    g.refine_near_points([(0.0, y, 1.0) for y in (0.5, 1.5, 2.5, 3.5)], 0.06)
    return g.mesh(grading=0.3)


def pipe_bend() -> rm.Mesh:
    """A 90 degree pipe bend (diameter 1, bend radius 1.5 D) with straight
    legs of 3 D: secondary flow in the bend, recovery downstream."""
    D, Rb, leg = 1.0, 1.5, 3.0
    path = [(-leg, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for k in range(1, 17):
        a = 0.5 * math.pi * k / 16
        path.append((Rb * math.sin(a), Rb * (1 - math.cos(a)), 0.0))
    path.append((Rb, Rb + leg, 0.0))
    g = rm.Geometry(maxh=0.12)
    g.label(g.sweep(path, D / 2, segments=32), "fluid")
    return g.mesh()


def heat_sink() -> rm.Mesh:
    """Conjugate heat transfer: an aluminium heat sink (base 50 x 50 x 4 mm,
    nine 1.5 mm fins, 20 mm tall) in an air duct; fluid and solid are two
    regions sharing the finned interface."""
    mm = 1e-3
    g = rm.Geometry(maxh=4 * mm)
    g.label(g.box(120 * mm, 60 * mm, 40 * mm, position=(-60 * mm, -30 * mm, 0)), "air")
    base = g.box(50 * mm, 50 * mm, 4 * mm, position=(-25 * mm, -25 * mm, 0), maxh=1.5 * mm)
    t, n = 1.5 * mm, 9
    pitch = (50 * mm - t) / (n - 1)
    fins = [
        g.box(50 * mm, t, 20.5 * mm, position=(-25 * mm, -25 * mm + k * pitch, 3.5 * mm), maxh=1.5 * mm)
        for k in range(n)
    ]
    g.label(g.union(base, *fins), "heat sink")
    return g.mesh()


EXAMPLES = {
    "naca_wing_tunnel": naca_wing_tunnel,
    "dfg_cylinder": dfg_cylinder,
    "backward_facing_step": backward_facing_step,
    "pipe_bend": pipe_bend,
    "heat_sink": heat_sink,
}


if __name__ == "__main__":
    for name, build in EXAMPLES.items():
        s = build().stats
        print(f"{name:22} {s['n_tets']:7} tets  min-dih {s['min_dihedral_deg']:5.1f}  {s['millis']:6} ms")
