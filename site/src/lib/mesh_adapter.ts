/**
 * MeshJson (rapidmesh exporter schema, $lib/mesh_types) → MeshData (the
 * flat-typed-array shape the verbatim rapidfem MeshViewer consumes, $lib/msh).
 *
 * The adapter mirrors what rapidfem's own serializer emits, so the verbatim
 * viewer behaves exactly like a real rapidfem preview:
 *
 *   MeshJson.points  [x,y,z][]      → MeshData.nodes    Float64Array (flat)
 *   MeshJson.tets    [a,b,c,d][]    → MeshData.tets     Uint32Array (flat)
 *   MeshJson.tet_regions number[]   → MeshData.tet_phys Int32Array; each
 *       region becomes a 3D physical group named `dielectric_<r>` (the
 *       object-API material convention "<class>_<idx>"), so the viewer's
 *       volume-hull pass reconstructs and fills the per-material boundaries
 *       itself, with the dielectric hue cycle keyed by the region index,
 *       exactly as in rapidfem previews.
 *   MeshJson.faces with tag > 0     → MeshData.tris + tri_phys, named
 *       `pec_<tag>` (2D groups): named metal surfaces in conductor yellow.
 *       Untagged interface faces are NOT emitted; the volume hulls already
 *       cover them, like in rapidfem (avoids duplicate fills / z-fighting).
 *   MeshJson.edges   [a,b][]        → MeshData.edges flat Uint32Array
 *   MeshJson.stats.min_dihedral_deg → MeshData.stats.min_dihedral_deg
 */

import type { MeshJson } from '$lib/mesh_types';
import type { MeshData } from '$lib/msh';

// 2D group ids for named faces, kept clear of region ids (3D groups).
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

	const phys_names = new Map<number, string>();
	const phys_dim = new Map<number, number>();

	// Volumes: one 3D group per region, material-style names.
	const nt = j.tets.length;
	const tets = new Uint32Array(nt * 4);
	const tet_phys = new Int32Array(nt);
	for (let t = 0; t < nt; t++) {
		const tet = j.tets[t];
		tets[t * 4] = tet[0];
		tets[t * 4 + 1] = tet[1];
		tets[t * 4 + 2] = tet[2];
		tets[t * 4 + 3] = tet[3];
		const r = j.tet_regions?.[t] ?? 1;
		tet_phys[t] = r;
		if (!phys_names.has(r)) {
			phys_names.set(r, `region_${r}`);
			phys_dim.set(r, 3);
		}
	}

	// Named surfaces only (tag > 0): 2D pec groups.
	const named = j.faces.filter((f) => f.tag > 0);
	const tris = new Uint32Array(named.length * 3);
	const tri_phys = new Int32Array(named.length);
	for (let f = 0; f < named.length; f++) {
		const face = named[f];
		tris[f * 3] = face.tri[0];
		tris[f * 3 + 1] = face.tri[1];
		tris[f * 3 + 2] = face.tri[2];
		const tag = NAMED_TAG_BASE + face.tag;
		tri_phys[f] = tag;
		if (!phys_names.has(tag)) {
			phys_names.set(tag, `pec_${face.tag}`);
			phys_dim.set(tag, 2);
		}
	}

	// Feature edges: flat [a0,b0,a1,b1,...].
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
