/**
 * MeshJson (rapidmesh exporter schema, $lib/mesh_types) → MeshData (the
 * flat-typed-array shape the ported rapidfem MeshViewer consumes, $lib/msh).
 *
 * The rapidfem viewer was built for NAMED physical groups (air, dielectric,
 * conductor, pec, port) and colours geometry by classifying those names. The
 * showcase meshes carry only NUMERIC ids (per-tet region ids + per-face
 * surface tags), so this adapter synthesises rapidfem-style group names that
 * drive the viewer's exact palette while keeping each region/surface a
 * distinct, toggleable legend entry.
 *
 * ── FIELD MAPPING ────────────────────────────────────────────────────────
 *   MeshJson.points  [x,y,z][]            → MeshData.nodes      Float64Array (flat, metres)
 *   MeshJson.tets    [a,b,c,d][]          → MeshData.tets       Uint32Array (flat)
 *   MeshJson.tet_regions number[]         → (NOT copied to tet_phys; see below)
 *   MeshJson.faces   {tri,tag,regions}[]  → MeshData.tris       Uint32Array (flat, one per face)
 *                                           MeshData.tri_phys   Int32Array (synthetic surface tag)
 *   MeshJson.edges   [a,b][]              → MeshData.edges      Uint32Array (FLATTENED to [a0,b0,...])
 *   MeshJson.stats.min_dihedral_deg       → MeshData.stats.min_dihedral_deg
 *   (derived) edge count                  → MeshData.stats.n_edges
 *   phys_names / phys_dim                 → synthesised below
 *
 * ── WHY FACES (not tet hulls) DRIVE THE FILLS ────────────────────────────
 *   rapidfem normally fills VOLUME hulls reconstructed from tets and only
 *   surfaces NAMED tris. Here MeshJson.faces already IS the exact boundary +
 *   interface triangulation, and the viewer's surface-wireframe layer is built
 *   only from tris with a non-zero tri_phys. So we map EVERY face to a tris
 *   entry with a non-zero synthetic tag: this makes the Surf fill AND the Wire
 *   overlay cover the whole model for every mesh (waveguides have no named
 *   surfaces at all), with no duplicate geometry. tet_phys is left all-zero so
 *   the viewer's volume-hull pass produces nothing (no double fills / no
 *   z-fighting). The tets are still uploaded — the interior tet-wireframe layer
 *   and the crinkle clip iterate them directly, independent of tet_phys.
 *
 * ── COLOURING (rapidfem palette, verbatim) ───────────────────────────────
 *   • Faces with tag === 0 (region boundaries/interfaces) → name `dielectric_<r>`
 *     where r is the adjacent interior region. classify() → 'dielectric', so
 *     color_for() cycles the rapidfem DIELECTRIC_CYCLE per region: distinct
 *     hues, one per region, matching the dev viewer's per-region colours.
 *   • Faces with tag > 0 (named metal surfaces: traces, conductor walls) →
 *     name `pec_<tag>`. classify() → 'conductor', so color_for() returns the
 *     signature accentSecondary yellow — exactly how rapidfem paints metal.
 */

import type { MeshJson } from '$lib/mesh_types';
import type { MeshData } from '$lib/msh';

// Synthetic surface-tag bases. Both are > 0 (rapidfem treats tag 0 as
// unnamed/skip) and disjoint so region-boundary and named-metal groups never
// collide. Region r → REGION_TAG_BASE + r; face tag t → NAMED_TAG_BASE + t.
const REGION_TAG_BASE = 1000;
const NAMED_TAG_BASE = 2000;

export function adaptMesh(j: MeshJson): MeshData {
	const np = j.points.length;
	const nodes = new Float64Array(np * 3);
	const min: [number, number, number] = [Infinity, Infinity, Infinity];
	const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
	for (let i = 0; i < np; i++) {
		const p = j.points[i];
		nodes[i * 3] = p[0];
		nodes[i * 3 + 1] = p[1];
		nodes[i * 3 + 2] = p[2];
		for (let k = 0; k < 3; k++) {
			if (p[k] < min[k]) min[k] = p[k];
			if (p[k] > max[k]) max[k] = p[k];
		}
	}

	// Faces → tris + tri_phys, building phys_names / phys_dim as we go.
	const nf = j.faces.length;
	const tris = new Uint32Array(nf * 3);
	const tri_phys = new Int32Array(nf);
	const phys_names = new Map<number, string>();
	const phys_dim = new Map<number, number>();
	for (let f = 0; f < nf; f++) {
		const face = j.faces[f];
		tris[f * 3] = face.tri[0];
		tris[f * 3 + 1] = face.tri[1];
		tris[f * 3 + 2] = face.tri[2];
		let tag: number;
		if (face.tag > 0) {
			tag = NAMED_TAG_BASE + face.tag;
			if (!phys_names.has(tag)) {
				phys_names.set(tag, `pec_${face.tag}`);
				phys_dim.set(tag, 2);
			}
		} else {
			// Colour by the adjacent interior region: prefer the first positive
			// of the two region ids (0 = exterior/void). Falls back to region 1.
			let reg = face.regions[0] > 0 ? face.regions[0] : face.regions[1];
			if (reg <= 0) reg = 1;
			tag = REGION_TAG_BASE + reg;
			if (!phys_names.has(tag)) {
				phys_names.set(tag, `dielectric_${reg}`);
				phys_dim.set(tag, 2);
			}
		}
		tri_phys[f] = tag;
	}

	// Tets → flat tets; tet_phys left all-zero (see header: no volume hulls).
	const nt = j.tets.length;
	const tets = new Uint32Array(nt * 4);
	for (let t = 0; t < nt; t++) {
		const tet = j.tets[t];
		tets[t * 4] = tet[0];
		tets[t * 4 + 1] = tet[1];
		tets[t * 4 + 2] = tet[2];
		tets[t * 4 + 3] = tet[3];
	}
	const tet_phys = new Int32Array(nt); // all 0 → viewer's hull pass is a no-op

	// Feature edges: MeshJson stores [a,b][] pairs; MeshData wants a flat
	// [a0,b0,a1,b1,...] Uint32Array. Empty/absent → undefined (renders as none).
	let edges: Uint32Array | undefined;
	if (j.edges && j.edges.length > 0) {
		edges = new Uint32Array(j.edges.length * 2);
		for (let i = 0; i < j.edges.length; i++) {
			edges[i * 2] = j.edges[i][0];
			edges[i * 2 + 1] = j.edges[i][1];
		}
	}

	return {
		nodes,
		tris,
		tri_phys,
		tets,
		tet_phys,
		phys_names,
		phys_dim,
		bbox: { min, max },
		edges,
		stats: {
			n_edges: edges ? edges.length / 2 : 0,
			min_dihedral_deg: j.stats?.min_dihedral_deg,
		},
	};
}
