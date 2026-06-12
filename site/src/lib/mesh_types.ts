/* SHARED RENDERING MODULE — DUPLICATED COPY.
 * Source of truth: rapidmesh/viewer/src/lib/mesh_types.ts
 * This file is copied verbatim into the mesh.rapidpassives.org showcase
 * (rapidmesh/site). The dev viewer at rapidmesh/viewer is canonical; keep
 * the two in sync if the renderer changes. Plain ES module, no SvelteKit deps.
 */
/** Mesh JSON schema shared by all mesher exporters, and the viewer's
 *  display settings. */

export interface MeshJson {
	name: string;
	mesher: string;
	points: [number, number, number][];
	tets: [number, number, number, number][];
	tet_regions: number[];
	faces: { tri: [number, number, number]; tag: number; regions: [number, number] }[];
	/** Feature edges (geometric creases) as vertex-index pairs. May be absent
	 *  or empty for meshes without exported feature edges. */
	edges?: [number, number][];
	stats: {
		n_points: number;
		n_tets: number;
		min_dihedral_deg: number;
		max_radius_edge: number;
		max_edge: number;
		millis: number;
	};
}

export interface ViewSettings {
	surface_wire: boolean;
	tet_fill: boolean;
	tet_wire: boolean;
	clip_enable: boolean;
	clip_axis: 0 | 1 | 2;
	clip_t: number;
}
