"""Surface-only export (Geometry.surface_mesh) demo: writes OBJ files of the
conforming boundary surface for a few curved cases, so the curved Lloyd
(analytic-chart relaxation with curvature/volume-error bias) can be inspected
in any 3D viewer. No volume mesh is built."""

from pathlib import Path
import rapidmesh as rm


def write_obj(sm, path: Path) -> None:
    lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in sm.points]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in sm.faces]
    path.write_text("\n".join(lines))
    print(f"{path.name}: {len(sm.points)} verts, {len(sm.faces)} faces")


def main() -> None:
    out = Path(__file__).parent / "surface_out"
    out.mkdir(exist_ok=True)

    # Two partially overlapping spheres: three regions (A\B, B, outside) meet on
    # the intersection circle; each curved group is relaxed on its sphere.
    g = rm.Geometry(maxh=0.4)
    g.sphere(1.0, (0, 0, 0), segments=24, rings=16)
    g.sphere(1.0, (1.2, 0, 0), segments=24, rings=16)
    write_obj(g.surface_mesh(), out / "overlapping_spheres.obj")

    # A hemisphere (sphere minus a half-space box void): a BOUNDED curved group,
    # so the chart engages. Meshed coarse (maxh >> radius), the curvature/
    # volume-error bias still keeps the dome round and even.
    g = rm.Geometry(maxh=5.0)
    g.sphere(1.0, (0, 0, 0), segments=24, rings=16)
    g.box(4, 4, 2, position=(-2, -2, -2), void=True)  # carves z < 0
    write_obj(g.surface_mesh(), out / "hemisphere_curvature_bias.obj")

    # A cylinder (closed barrel -> falls back to its facets) plus its flat caps.
    g = rm.Geometry(maxh=0.4)
    g.cylinder(0.8, 3.0, position=(0, 0, 0), axis=(0, 0, 1))
    write_obj(g.surface_mesh(), out / "cylinder.obj")

    # NACA 0012 airfoil extruded into a 3D wing section: the curved skin is one
    # analytic extruded-spline surface; the curvature bias refines the sharp
    # leading edge and coarsens the gentle aft, with vertices exactly on the
    # profile. Trailing edge is a flat blunt face.
    g = rm.Geometry(maxh=0.25)
    g.airfoil_naca0012(chord=1.0, span=0.4, n_seg=140)
    write_obj(g.surface_mesh(), out / "naca0012.obj")

    print(f"\nOBJ files written to {out}")


if __name__ == "__main__":
    main()
