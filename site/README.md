# rapidmesh showcase (mesh.rapidpassives.org)

A fullscreen, auto-cycling 3D gallery of meshes produced by the rapidmesh
tet-mesher. Built with SvelteKit + `adapter-static`, deployed as plain files to
GitHub Pages on the custom domain `mesh.rapidpassives.org`.

## What it does

- The WebGL viewer fills the entire viewport and slowly orbits each model.
- Models auto-cycle every 8 s with a fade-through-black transition.
- A row of tabs along the bottom switches model immediately; a manual click (or
  any drag / zoom) pauses auto-cycling and it resumes after 30 s of inactivity.
- A `rapidmesh` wordmark sits top-left with a per-model stats line.

All timing and motion constants live in `src/lib/constants.ts`. Colors, fonts
and sizes come from the shared design tokens in `src/lib/theme.ts` /
`src/app.css` (no one-off values in components).

## Local development

```sh
npm install
npm run dev        # http://localhost:5200
```

Open the printed localhost URL in a browser to view the site.

## Production build

```sh
npm run build      # static output in build/
npm run preview    # serve the built output locally to verify
```

`build/` contains `index.html`, hashed JS/CSS, the `meshes/` data and a `CNAME`
file (`mesh.rapidpassives.org`), ready to publish to GitHub Pages. DNS for the
custom domain is handled by the repository owner.

## Shared viewer code

The WebGL renderer and camera bus are shared with the dev viewer at
`rapidmesh/viewer` (the source of truth):

- `src/lib/render/canvas3d.ts`
- `src/lib/viewbus.ts`
- `src/lib/theme.ts`
- `src/lib/mesh_types.ts`

These are **verbatim copies** of the matching files under
`rapidmesh/viewer/src/lib/` (each carries a header noting the source of truth).
If the renderer changes there, re-copy them here. The plain ES modules carry no
SvelteKit dependency, so the copy works as-is. The Svelte wrapper
`src/lib/components/MeshViewer.svelte` is a trimmed, chrome-free derivative of
the viewer's `MeshPanel.svelte` (fullscreen canvas, dim surface fill + bright
wireframe, idle orbit).

## Mesh data and the manifest

Mesh JSON files live in `static/meshes/`. The model list is **data-driven**:
`static/meshes/manifest.json` lists each model's `id`, display `name`, mesh
`file` and optional `stats` (`n_tets`, `n_points`, `min_dihedral_deg`). To add a
model, drop its JSON in `static/meshes/` and add an entry to the manifest; no
code changes are needed.

### Syncing baked meshes from the dev viewer

The current models were copied from the dev viewer's pre-baked exports
(`rapidmesh/viewer/public/meshes/`) and renamed to stable showcase ids:

| showcase file              | source export                          |
| -------------------------- | -------------------------------------- |
| `meshes/wr90.json`         | `rapidmesh_wr90_default.json`          |
| `meshes/coax.json`         | `rapidmesh_coax_step_default.json`     |
| `meshes/iris.json`         | `rapidmesh_iris_filter.json`           |
| `meshes/microstrip.json`   | `rapidmesh_microstrip_line.json`       |
| `meshes/stepped.json`      | `rapidmesh_stepped_lpf_default.json`   |

To refresh or add benchmark models, copy the desired `rapidmesh_*.json` export
into `static/meshes/` under its showcase id and update `manifest.json`.
