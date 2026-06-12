/* SHARED RENDERING MODULE — DUPLICATED COPY.
 * Source of truth: rapidfem/python/python_src/rapidfem/ui/frontend-src/src/lib/msh.ts
 * This file is copied verbatim into the mesh.rapidpassives.org showcase
 * (rapidmesh/site). The rapidfem mesh-preview viewer is canonical; keep the
 * two in sync if the renderer changes. Plain ES module, no SvelteKit deps.
 *
 * `MeshData` is the flat-typed-array shape the rapidfem MeshViewer consumes.
 * The showcase meshes are exported by rapidmesh in the `MeshJson` schema
 * (see $lib/mesh_types); $lib/mesh_adapter converts MeshJson → MeshData.
 */

/** Common mesh-data shape consumed by the 3D viewer. Built directly from
 *  the WASM mesher's output (`mesh_from_spec`). */

export interface MeshData {
	nodes: Float64Array;        // [x0,y0,z0, ...] in METERS — kept f64 for clean
	                             // analytical normals on coplanar triangles
	                             // (μm-scale geometry suffers from f32 cross-product noise)
	tris: Uint32Array;
	tri_phys: Int32Array;
	tets: Uint32Array;
	tet_phys: Int32Array;
	phys_names: Map<number, string>;
	phys_dim: Map<number, number>;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	/** Flat uint pairs [a0,b0, a1,b1, ...] of feature edges (geometric creases).
	 *  Old payloads without this field render identically (treated as empty). */
	edges?: Uint32Array;
	/** Optional mesh quality statistics from the backend. */
	stats?: {
		n_edges?: number;
		min_dihedral_deg?: number;
	};
}
