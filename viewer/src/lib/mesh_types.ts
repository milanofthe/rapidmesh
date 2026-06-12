/** Mesh JSON schema shared by all mesher exporters, and the viewer's
 *  display settings.
 *
 *  SOURCE OF TRUTH. A verbatim copy lives in
 *  rapidmesh/site/src/lib/mesh_types.ts (the mesh.rapidpassives.org showcase).
 *  Keep the two in sync if this changes. */

export interface MeshJson {
	name: string;
	mesher: string;
	points: [number, number, number][];
	tets: [number, number, number, number][];
	tet_regions: number[];
	faces: { tri: [number, number, number]; tag: number; regions: [number, number] }[];
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
