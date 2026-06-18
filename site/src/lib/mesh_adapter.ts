/**
 * MeshJson (rapidmesh exporter schema, $lib/mesh_types) → MeshData (the
 * flat-typed-array shape the verbatim rapidfem MeshViewer consumes, $lib/msh).
 *
 * Grouping uses the exporter's surface provenance and solid labels:
 *
 *   Volumes — one 3D physical group per region; regions whose solids share
 *       a label merge into one group (the eight bearing balls are one
 *       legend entry). Group names are the labels themselves.
 *   Sheets (faces with tag > 0) — one 2D group per tag, named by the
 *       exporter's tag label (`sheet <tag>` without one).
 *   Cavity walls (untagged faces owned by a void solid) — one 2D group per
 *       void label (`cavities` without one). Voids carry no region, so
 *       these groups are the only way their carved surfaces get their own
 *       color, like the coax inner conductors.
 *   Other untagged faces are NOT emitted; the viewer's volume passes cover
 *       them (avoids duplicate fills / z-fighting).
 *   MeshJson.edges  [a,b][]  → MeshData.edges flat Uint32Array
 */

import type { MeshJson } from '$lib/mesh_types';
import type { MeshData } from '$lib/msh';

// 2D group id ranges, kept clear of 3D group ids (small ints).
const SHEET_TAG_BASE = 2000;
const VOID_TAG_BASE = 3000;

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
	const solids = j.solids ?? [];

	// Volumes: regions merge into one 3D group per display name.
	const region_name = new Map<number, string>();
	for (let i = 0; i < solids.length; i++) {
		const s = solids[i];
		if (s.region > 0 && !region_name.has(s.region)) {
			region_name.set(s.region, s.label ?? `solid ${i + 1}`);
		}
	}
	const gid_by_name = new Map<string, number>();
	const gid_by_region = new Map<number, number>();
	let next_gid = 1;
	const region_gid = (r: number): number => {
		let gid = gid_by_region.get(r);
		if (gid !== undefined) return gid;
		const name = region_name.get(r) ?? `region ${r}`;
		gid = gid_by_name.get(name);
		if (gid === undefined) {
			gid = next_gid++;
			gid_by_name.set(name, gid);
			phys_names.set(gid, name);
			phys_dim.set(gid, 3);
		}
		gid_by_region.set(r, gid);
		return gid;
	};

	const nt = j.tets.length;
	const tets = new Uint32Array(nt * 4);
	const tet_phys = new Int32Array(nt);
	for (let t = 0; t < nt; t++) {
		const tet = j.tets[t];
		tets[t * 4] = tet[0];
		tets[t * 4 + 1] = tet[1];
		tets[t * 4 + 2] = tet[2];
		tets[t * 4 + 3] = tet[3];
		tet_phys[t] = region_gid(j.tet_regions?.[t] ?? 1);
	}

	// 2D groups: sheets by tag, cavity walls by void label.
	const owners = j.surface_owners ?? [];
	const void_gids = new Map<string, number>();
	let next_void = VOID_TAG_BASE;
	const face_group = (face: MeshJson['faces'][number]): number | null => {
		if (face.tag > 0) {
			const gid = SHEET_TAG_BASE + face.tag;
			if (!phys_names.has(gid)) {
				phys_names.set(gid, j.tag_labels?.[String(face.tag)] ?? `sheet ${face.tag}`);
				phys_dim.set(gid, 2);
			}
			return gid;
		}
		const owner = face.surface !== undefined ? owners[face.surface] : undefined;
		if (owner === undefined || owner < 0) return null;
		const s = solids[owner];
		if (!s || s.region > 0) return null; // covered by the volume passes
		const name = s.label ?? 'cavities';
		let gid = void_gids.get(name);
		if (gid === undefined) {
			gid = next_void++;
			void_gids.set(name, gid);
			phys_names.set(gid, name);
			phys_dim.set(gid, 2);
		}
		return gid;
	};

	const kept: { f: MeshJson['faces'][number]; gid: number }[] = [];
	for (const f of j.faces) {
		const gid = face_group(f);
		if (gid !== null) kept.push({ f, gid });
	}
	const tris = new Uint32Array(kept.length * 3);
	const tri_phys = new Int32Array(kept.length);
	for (let i = 0; i < kept.length; i++) {
		const { f, gid } = kept[i];
		tris[i * 3] = f.tri[0];
		tris[i * 3 + 1] = f.tri[1];
		tris[i * 3 + 2] = f.tri[2];
		tri_phys[i] = gid;
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
		defects: j.defects,
	};
}
