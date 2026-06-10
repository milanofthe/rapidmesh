"""The rapidfem example geometries, rebuilt with the rapidmesh builder API
and exported into the comparison viewer.

Each function is a straight port of the geometry section of the
corresponding ``rapidfem.examples.fd_*`` script (materials and physics
stripped: regions keep their meaning through the region tags, PEC traces
through sheet tags). Run from the repo root:

    python python/examples/rapidfem_geometries.py
"""

from pathlib import Path

import rapidmesh as rm

mm = 1e-3
C0 = 299_792_458.0

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer" / "public" / "meshes"


def lambda_maxh(f_max: float, er_max: float = 1.0, n: int = 10) -> float:
    """Target edge length of n elements per wavelength at f_max."""
    return C0 / (f_max * n * er_max**0.5)


def coax_step() -> rm.Mesh:
    """Coaxial impedance step (50 -> 75 ohm): outer air dielectric with two
    stepped inner conductors (their volumes become separate regions whose
    surfaces a solver marks PEC)."""
    ri1, ro = 1.50 * mm, 3.45 * mm
    ri2 = 0.99 * mm
    l1, l2 = 15.0 * mm, 15.0 * mm
    g = rm.Geometry(maxh=lambda_maxh(f_max=10.0e9))
    g.cylinder(radius=ro, height=l1 + l2, position=(0, 0, 0))
    g.cylinder(radius=ri1, height=l1, position=(0, 0, 0), maxh=lambda_maxh(10.0e9) / 2)
    g.cylinder(radius=ri2, height=l2, position=(0, 0, l1), maxh=lambda_maxh(10.0e9) / 2)
    return g.mesh()


def microstrip_line() -> rm.Mesh:
    """50 ohm microstrip on RO4003C: substrate + air, PEC trace on the
    interface (sheet tag 7)."""
    sub_h, er_sub = 0.508 * mm, 3.55
    line_w, line_l = 1.13 * mm, 30.0 * mm
    sub_w, air_h = 20.0 * mm, 10.0 * mm
    g = rm.Geometry(maxh=lambda_maxh(f_max=3.3e9))
    g.box(sub_w, line_l, air_h + sub_h, position=(-sub_w / 2, 0, 0))
    g.box(
        sub_w,
        line_l,
        sub_h,
        position=(-sub_w / 2, 0, 0),
        maxh=lambda_maxh(f_max=3.3e9, er_max=er_sub) / 4,
    )
    g.xy_plate(line_w, line_l, position=(-line_w / 2, 0, sub_h), tag=7)
    return g.mesh()


def dielectric_resonator() -> rm.Mesh:
    """Shielded dielectric resonator: a high-permittivity puck on a low-loss
    support post inside a metallic cavity."""
    inch = 25.4 * mm
    w, s = 2.0 * inch, 2.03 * inch
    d_sup, l_sup, er_sup = 0.56 * inch, 0.80 * inch, 10.0
    d_res, l_res, er_res = 1.176 * inch, 0.481 * inch, 34.0
    f = 3.0e9
    g = rm.Geometry(maxh=lambda_maxh(f_max=f))
    g.box(w, w, s, position=(-w / 2, -w / 2, 0))
    # segments=28: the default tessellation of this size combination trips a
    # known recovery-bookkeeping issue on curved patches (tracked upstream).
    g.cylinder(radius=d_sup / 2, height=l_sup, segments=28, maxh=lambda_maxh(f, er_sup))
    g.cylinder(
        radius=d_res / 2,
        height=l_res,
        position=(0, 0, l_sup),
        segments=28,
        maxh=lambda_maxh(f, er_res),
    )
    return g.mesh()


def iris_filter() -> rm.Mesh:
    """WR90 X-band iris bandpass filter: three inductive iris pairs in a
    rectangular waveguide (iris volumes as separate regions, like the
    rapidfem original where their faces become PEC)."""
    a, b = 22.86 * mm, 10.16 * mm
    apertures = [10.0 * mm, 8.0 * mm, 10.0 * mm]
    spacing, iris_t = 15.0 * mm, 1.0 * mm
    input_len, output_len = 12.0 * mm, 12.0 * mm
    length = input_len + (len(apertures) - 1) * spacing + 2 * iris_t + output_len
    g = rm.Geometry(maxh=lambda_maxh(f_max=12.4e9))
    g.box(a, b, length, position=(-a / 2, -b / 2, 0))
    z_centers = [input_len + iris_t / 2 + k * spacing for k in range(len(apertures))]
    for zc, slot in zip(z_centers, apertures):
        strip_w = (a - slot) / 2
        for x0 in (-a / 2, slot / 2):
            g.box(
                strip_w,
                b,
                iris_t,
                position=(x0, -b / 2, zc - iris_t / 2),
                maxh=lambda_maxh(f_max=12.4e9) / 2,
            )
    return g.mesh()


def patch_antenna() -> rm.Mesh:
    """Inset-fed 2.4 GHz patch antenna: FR-4 substrate in an air volume,
    patch and ground plane as tagged sheets (PML shells of the rapidfem
    original omitted: they are absorber bookkeeping, not geometry)."""
    sub_w, sub_l, sub_h, er_sub = 60 * mm, 60 * mm, 1.6 * mm, 4.4
    patch_w, patch_l = 38 * mm, 29 * mm
    pad_xy, pad_z = 25 * mm, 30 * mm
    f = 2.8e9
    total_w, total_l = sub_w + 2 * pad_xy, sub_l + 2 * pad_xy
    g = rm.Geometry(maxh=lambda_maxh(f_max=f))
    g.box(
        total_w,
        total_l,
        sub_h + pad_z,
        position=(-total_w / 2, -total_l / 2, 0),
    )
    g.box(
        sub_w,
        sub_l,
        sub_h,
        position=(-sub_w / 2, -sub_l / 2, 0),
        maxh=lambda_maxh(f_max=f, er_max=er_sub) / 2,
    )
    g.xy_plate(patch_w, patch_l, position=(-patch_w / 2, -patch_l / 2, sub_h), tag=7)
    g.xy_plate(sub_w, sub_l, position=(-sub_w / 2, -sub_l / 2, 0), tag=7)
    return g.mesh()


EXAMPLES = {
    "coax_step": coax_step,
    "microstrip_line": microstrip_line,
    "dielectric_resonator": dielectric_resonator,
    "iris_filter": iris_filter,
    "patch_antenna": patch_antenna,
}


if __name__ == "__main__":
    for name, build in EXAMPLES.items():
        mesh = build()
        s = mesh.stats
        print(
            f"{name:22} {s['n_tets']:7} tets  {s['n_points']:6} pts  "
            f"min-dih {s['min_dihedral_deg']:5.1f}  r/e {s['max_radius_edge']:6.2f}  "
            f"{s['millis']:5} ms"
        )
        mesh.save_viewer_json(name, VIEWER_DIR)
    print(f"-> {VIEWER_DIR}")
