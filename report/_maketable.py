"""Emit report/validation/quality_table.tex from the campaign results.json.

A grouped tabularx: per case, the quality metric (min dihedral for volumes, min
planar angle for 2D plates) at coarse/medium/fine density, plus the fine element
count. Regenerable; the report \\inputs the produced file.
"""
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
results = json.loads((REPO / "report" / "validation" / "results.json").read_text())

CAT_ORDER = ["2D", "Primitive", "Boolean"]
CAT_LABEL = {"2D": "2D plates (min planar angle)",
             "Primitive": "3D primitives (min dihedral)",
             "Boolean": "Booleans (min dihedral)"}
NICE = {  # display names
    "plate_rect": "rectangle plate", "disc": "disc", "hexagon": "hexagon",
    "l_polygon": "L-polygon", "annulus": "annulus",
    "box": "box", "sphere": "sphere", "cylinder": "cylinder", "cone": "cone",
    "frustum": "frustum", "torus": "torus", "wedge": "wedge", "prism_l": "L-prism",
    "union_box_sphere": "box $\\cup$ sphere", "diff_box_cyl": "box $-$ cylinder",
    "diff_box_sphere": "box $-$ sphere", "fused_two": "2 fused spheres",
    "fused_three": "3 fused spheres", "capsule": "capsule (cyl $\\cup$ 2 sph)",
    "union_box_cyl": "box $\\cup$ cylinder", "diff_cyl_box": "cylinder $-$ slab",
    "drilled_block": "drilled block ($-$2 cyl)",
}

by = collections.OrderedDict()
for x in results:
    if "error" in x:
        continue
    by.setdefault((x["category"], x["name"]), []).append(x)

lines = [
    r"\begin{table}[H]\centering",
    r"\caption{From-scratch validation: rapidmesh's own element quality across "
    r"three mesh densities (coarse / medium / fine), and the fine element count. "
    r"Min planar angle for the 2D plates, min dihedral angle for the volumes.}",
    r"\label{tab:validation}",
    r"\begin{tabularx}{\textwidth}{@{}l L c c c r@{}}",
    r"\toprule",
    r"Case & test & coarse & medium & fine & \#elems\\",
    r"\midrule",
]
for cat in CAT_ORDER:
    items = [(k, v) for k, v in by.items() if k[0] == cat]
    if not items:
        continue
    lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\itshape {CAT_LABEL[cat]}}}\\")
    for (c, name), recs in items:
        recs = sorted(recs, key=lambda z: -z["h"])  # coarse -> fine
        q = [f"{z['quality_deg']:.1f}" for z in recs]
        while len(q) < 3:
            q.append("--")
        elems = recs[-1]["n_elems"]
        disp = NICE.get(name, name.replace("_", r"\_"))
        kindtxt = "plate" if cat == "2D" else ("union/cut" if cat == "Boolean" else "solid")
        lines.append(rf"\quad {disp} & {kindtxt} & {q[0]} & {q[1]} & {q[2]} & {elems}\\")
    lines.append(r"\addlinespace")
if lines[-1] == r"\addlinespace":
    lines.pop()
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

out = REPO / "report" / "validation" / "quality_table.tex"
out.write_text("\n".join(lines))
print(f"wrote {out}  ({sum(1 for x in results if 'error' not in x)} rows from results.json)")
