<script lang="ts">
	/**
	 * Full rapidfem mesh-preview viewer, ported 1:1 into the
	 * mesh.rapidpassives.org showcase.
	 *
	 * Source of truth:
	 *   rapidfem/python/python_src/rapidfem/ui/frontend-src/src/lib/components/MeshViewer.svelte
	 *
	 * This is the EXACT viewer rapidfem uses for mesh previews — same toolbar
	 * (zoom in/out, fit view, rotate, save PNG), same inspection layer toggles
	 * (Surf / Wire / Edge / Tets), same crinkle clip (Clip + X/Y/Z axis + slider,
	 * prefix-sorted O(log n) draw-count clipping), same per-group legend with
	 * click-to-hide toggles, same HUD, and the same colours / markup / styles.
	 *
	 * INTENTIONAL ADAPTATIONS for the showcase (everything else is verbatim):
	 *   • Removed the field / time-domain visualisation paths (FD |E| point
	 *     cloud, TD trajectory animation, video recording, colourbar, E/J/H +
	 *     lin/log toolbars). Those require the rapidfem FEM kernel ($lib/api,
	 *     viz worker), which the static showcase does not ship and never feeds
	 *     to this component. The mesh-inspection feature set the owner asked for
	 *     is preserved completely.
	 *   • Input is the showcase's MeshJson payload; it is converted to the
	 *     rapidfem MeshData shape via $lib/mesh_adapter (documented mapping).
	 *   • Added the showcase's idle slow orbit + `oninteract` callback (the shell
	 *     pauses its auto-cycle on any viewer interaction, toolbar included).
	 */
	import { onMount, untrack } from 'svelte';
	import {
		initGL, disposeGL, clearMeshes, addMesh, addLineMesh, setBBox,
		render3D, fitCamera, setTagVisible,
		type GLState, type Camera
	} from '$lib/render/canvas3d';
	import { buildTriSoupF64 } from '$lib/render/mesh_scene';
	import type { MeshData } from '$lib/msh';
	import { palette, canvas as canvasTheme } from '$lib/theme';
	import { adaptMesh } from '$lib/mesh_adapter';
	import { ORBIT_SPEED } from '$lib/constants';
	import type { MeshJson } from '$lib/mesh_types';

	// ── Inspection layer tags (negative to avoid colliding with group ids) ──
	const TAG_WIRE_SURF  = -11;  // surface wireframe overlay
	const TAG_FEAT_EDGES = -12;  // feature edges from payload
	const TAG_TET_WIRE   = -13;  // interior tet wireframe

	// Axis descriptors for the crinkle-clip toolbar buttons.
	const CLIP_AXES: { lbl: string; ax: 0 | 1 | 2 }[] = [
		{ lbl: 'X', ax: 0 },
		{ lbl: 'Y', ax: 1 },
		{ lbl: 'Z', ax: 2 },
	];

	let {
		data = null as MeshJson | null,
		orbit = true,
		oninteract
	}: {
		data?: MeshJson | null;
		orbit?: boolean;
		oninteract?: () => void;
	} = $props();

	// MeshJson → MeshData. $derived (NOT $state) so the large typed arrays are
	// never deep-proxied; recomputes only when the `data` prop reference changes.
	const mesh = $derived<MeshData | null>(data ? adaptMesh(data) : null);

	let canvas = $state<HTMLCanvasElement | null>(null);
	let container = $state<HTMLDivElement | null>(null);
	// Plain (non-$state) — the renderer/GL object and camera hold large buffers
	// and are walked every animation frame; proxying them would tax each frame.
	let gl_state: GLState | null = null;
	let camera: Camera = { theta: Math.PI / 4, phi: Math.PI / 4, distance: 1, target: [0, 0, 0] };
	let z_flip = 1;
	let mounted = false;
	let needs_rebuild = true;
	let cursor_world = $state({ x: 0, y: 0 });
	// Stored as "hidden" so that newly-built meshes default to visible without
	// losing the explicit hides the user picked from the legend.
	let hidden_tags = $state(new Set<number>());

	// ── Inspection layer toggles (internal toolbar state) ───────────────
	let layer_surface = $state(true);
	let layer_wire    = $state(false);
	let layer_edges   = $state(false);
	let layer_tets    = $state(false);

	// ── Crinkle clip (prefix-sort trick from rapidmesh MeshPanel) ───────
	let clip_enable = $state(false);
	let clip_axis   = $state<0 | 1 | 2>(0);
	let clip_t      = $state(1.0);

	// Each entry tracks one mesh/lineMesh in the GL state. Sorted by centroid
	// along the active clip axis so the slider only needs a binary search to set
	// the draw count. No re-uploads during drags.
	interface PrefixMesh {
		idx:        number;        // index into state.meshes or state.lineMeshes
		line:       boolean;
		vals:       Float64Array;  // centroid values sorted ascending along built_axis
		vpu:        number;        // vertices per unit (3 for tri, 2 for edge)
		full_count: number;        // total vertex count when clip is disabled
	}
	let prefix_meshes: PrefixMesh[] = [];
	let built_axis  = -1;
	// Non-null when the sorted geometry is already uploaded for a given mesh+axis.
	let last_built_for: { mesh: MeshData; axis: number } | null = null;

	function notify_interact() { oninteract?.(); }

	function toggle_tag(tag: number) {
		if (!gl_state) return;
		notify_interact();
		const next = new Set(hidden_tags);
		if (next.has(tag)) next.delete(tag); else next.add(tag);
		hidden_tags = next;
		// In inspection mode, layer_surface gates all surface fills.
		setTagVisible(gl_state, tag, layer_surface && !next.has(tag));
	}
	let is_dragging = false;
	let is_right_drag = false;
	let last_mouse = { x: 0, y: 0 };

	// ── Camera animation (ease-out cubic) ──────────────────────────────
	let anim_id = 0;
	let anim_target: Camera | null = null;
	function effective_camera(): Camera { return anim_target ?? camera; }
	function animate_camera(target: Camera, durationMs = 300) {
		anim_target = target;
		const start = { ...camera, target: [...camera.target] as [number, number, number] };
		const t0 = performance.now();
		const id = ++anim_id;
		function step() {
			if (!mounted || id !== anim_id) return;
			const t = Math.min(1, (performance.now() - t0) / durationMs);
			const e = 1 - Math.pow(1 - t, 3);
			camera = {
				theta: start.theta + (target.theta - start.theta) * e,
				phi: start.phi + (target.phi - start.phi) * e,
				distance: start.distance + (target.distance - start.distance) * e,
				target: [
					start.target[0] + (target.target[0] - start.target[0]) * e,
					start.target[1] + (target.target[1] - start.target[1]) * e,
					start.target[2] + (target.target[2] - start.target[2]) * e
				]
			};
			if (t < 1) requestAnimationFrame(step);
			else anim_target = null;
		}
		requestAnimationFrame(step);
	}

	// ── Imperative toolbar actions ─────────────────────────────────────
	export function zoom_in() {
		notify_interact();
		const b = effective_camera();
		animate_camera({ ...b, target: [...b.target] as [number, number, number], distance: b.distance / 1.3 }, 200);
	}
	export function zoom_out() {
		notify_interact();
		const b = effective_camera();
		animate_camera({ ...b, target: [...b.target] as [number, number, number], distance: b.distance * 1.3 }, 200);
	}
	export function fit_view() {
		notify_interact();
		if (!mesh) return;
		animate_camera(fitCamera(mesh.bbox.min, mesh.bbox.max), 350);
	}
	export function rotate_90() {
		notify_interact();
		const b = effective_camera();
		animate_camera({ ...b, target: [...b.target] as [number, number, number], theta: b.theta + Math.PI / 2 }, 400);
	}
	export function save_png() {
		notify_interact();
		if (!canvas) return;
		render_frame();
		canvas.toBlob((blob) => {
			if (!blob) return;
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'rapidmesh.png';
			a.click();
			URL.revokeObjectURL(url);
		}, 'image/png');
	}

	// ── Mesh classification & coloring (verbatim rapidfem) ──────────────
	type Kind = 'dielectric' | 'conductor' | 'port' | 'gnd';

	/** Strip a trailing `_<digits>` index suffix so "air_1", "dielectric_2"
	 *  classify the same as "air", "dielectric". Returns the base name. */
	function base(name: string): string {
		return name.replace(/_\d+$/, '');
	}

	/** Render a physical-group name into a human-readable legend label. */
	function pretty_label(name: string): string {
		const b = base(name);
		const m = name.match(/_(\d+)$/);
		const idx = m ? parseInt(m[1], 10) : 0;
		const display: Record<string, string> = {
			air: 'Air',
			conductor: 'Conductor',
			dielectric: 'Dielectric',
			anisotropic: 'Anisotropic',
			port: 'Port',
			pec: 'PEC',
			pmc: 'PMC',
			abc: 'ABC',
			pml: 'PML',
			surfaceimpedance: 'Surface Impedance',
			lumpedelement: 'Lumped Element',
		};
		const d = display[b];
		if (!d) return name;
		const repeatable = b === 'port' || b === 'dielectric' || b === 'anisotropic' || b === 'pml';
		return repeatable && idx > 0 ? `${d} ${idx}` : d;
	}

	function classify(name: string): Kind | null {
		const b = base(name);
		if (b === 'air' || b === 'dielectric' || b === 'anisotropic') return 'dielectric';
		if (b === 'conductor') return 'conductor';
		if (b === 'port') return 'port';
		if (b === 'pec' || b === 'pmc' || b === 'surfaceimpedance' || b === 'lumpedelement') return 'conductor';
		if (b === 'abc') return null;
		if (b === 'pml') return 'dielectric';
		if (name.startsWith('_mat_')) return null;
		if (name === 'substrate' || name === 'oxide') return 'dielectric';
		if (name.endsWith('_gnd') || name === 'gnd' || name === 'ground') return 'gnd';
		if (name === 'p1' || name === 'p2' || /^p\d+$/.test(name) || name.endsWith('_port')) return 'port';
		return 'conductor';
	}

	// Dielectric cycle — distinct hues so multiple dielectrics are
	// distinguishable. Keep air on its own neutral gray channel.
	const DIELECTRIC_CYCLE = ['#4a9ec2', '#6bbf8a', '#7b5e8a', '#a78bd9', '#c4c46b'];

	function color_for(kind: Kind, name: string): [number, number, number] {
		const b = base(name);
		if (b === 'air') return hex('#5a5a62');
		if (b === 'conductor') return hex(palette.accentSecondary);
		if (b === 'dielectric' || b === 'anisotropic') {
			const m = name.match(/_(\d+)$/);
			const idx = m ? Math.max(0, parseInt(m[1], 10) - 1) : 0;
			return hex(DIELECTRIC_CYCLE[idx % DIELECTRIC_CYCLE.length]);
		}
		if (b === 'pml') return hex('#7b5e8a');
		if (kind === 'port') return hex(palette.accent);
		if (kind === 'conductor') return hex(palette.accentSecondary);
		if (kind === 'gnd') return hex('#5aad78');
		if (kind === 'dielectric') return hex('#5a5a62');
		const fixed: Record<string, string> = {
			met5: '#e8944a', met4: '#f0b86a', met3: '#c4c46b',
			met2: '#9bc28b', met1: '#7b9fb8', li1: '#5a8caa',
			via5: '#d9513c', via4: '#e5634f', via3: '#bf4233',
			via2: '#9d3526', via1: '#7c281b', mcon: '#aa6b40',
		};
		return hex(fixed[name] ?? palette.accentSecondary);
	}

	function hex(s: string): [number, number, number] {
		return [
			parseInt(s.slice(1, 3), 16) / 255,
			parseInt(s.slice(3, 5), 16) / 255,
			parseInt(s.slice(5, 7), 16) / 255
		];
	}

	// Wire layer colors follow the rapidfem convention: surface wire uses the
	// crosshair token, interior tet wire uses the dimmer grid token, and feature
	// edges use accentSecondary for visual distinction from wireframe.
	const WIRE_SURF_COLOR:  [number, number, number] = hex(canvasTheme.crosshair);
	const WIRE_INT_COLOR:   [number, number, number] = hex(canvasTheme.grid);
	const FEAT_EDGE_COLOR:  [number, number, number] = hex(palette.accentSecondary);

	/** Volume hull from tets — per volume independently. With the showcase
	 *  adapter tet_phys is all-zero, so this returns an empty map (no hulls);
	 *  kept verbatim for fidelity with the rapidfem source. */
	function build_volume_boundaries(m: MeshData): Map<number, number[]> {
		const enc = (a: number, b: number, c: number): bigint => {
			const s = [a, b, c].sort((x, y) => x - y);
			return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
		};
		const per_vol = new Map<number, number[]>();
		const ntets = m.tet_phys.length;
		for (let t = 0; t < ntets; t++) {
			const v = m.tet_phys[t];
			if (!v) continue;
			let arr = per_vol.get(v);
			if (!arr) { arr = []; per_vol.set(v, arr); }
			arr.push(t);
		}

		const orient_outward = (
			a: number, b: number, c: number, o: number
		): [number, number, number] => {
			const ax = m.nodes[a * 3], ay = m.nodes[a * 3 + 1], az = m.nodes[a * 3 + 2];
			const bx = m.nodes[b * 3], by = m.nodes[b * 3 + 1], bz = m.nodes[b * 3 + 2];
			const cx = m.nodes[c * 3], cy = m.nodes[c * 3 + 1], cz = m.nodes[c * 3 + 2];
			const ox = m.nodes[o * 3], oy = m.nodes[o * 3 + 1], oz = m.nodes[o * 3 + 2];
			const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
			const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
			const nx = e1y * e2z - e1z * e2y;
			const ny = e1z * e2x - e1x * e2z;
			const nz = e1x * e2y - e1y * e2x;
			const dx = ox - ax, dy = oy - ay, dz = oz - az;
			if (nx * dx + ny * dy + nz * dz > 0) return [a, c, b];
			return [a, b, c];
		};

		const out = new Map<number, number[]>();
		for (const [vol, tet_indices] of per_vol.entries()) {
			const seen = new Map<bigint, { count: number; tri: [number, number, number] }>();
			for (const t of tet_indices) {
				const a = m.tets[t * 4], b = m.tets[t * 4 + 1], c = m.tets[t * 4 + 2], d = m.tets[t * 4 + 3];
				const tri_descs: [[number, number, number], number][] = [
					[[a, b, c], d],
					[[a, b, d], c],
					[[a, c, d], b],
					[[b, c, d], a]
				];
				for (const [f, opp] of tri_descs) {
					const k = enc(f[0], f[1], f[2]);
					const prev = seen.get(k);
					if (!prev) {
						seen.set(k, { count: 1, tri: orient_outward(f[0], f[1], f[2], opp) });
					} else {
						prev.count++;
					}
				}
			}
			const arr: number[] = [];
			for (const e of seen.values()) {
				if (e.count === 1) arr.push(e.tri[0], e.tri[1], e.tri[2]);
			}
			if (arr.length) out.set(vol, arr);
		}
		return out;
	}

	// ── Crinkle clip helpers ────────────────────────────────────────────

	/** Binary search: first index where vals[i] > d. */
	function upper_bound(vals: Float64Array, d: number): number {
		let lo = 0, hi = vals.length;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (vals[mid] <= d) lo = mid + 1;
			else hi = mid;
		}
		return lo;
	}

	/** Adjust draw counts for all prefix meshes. Called on every slider drag
	 *  without re-uploading buffers. */
	function apply_clip_counts(state: GLState, enable: boolean, axis: 0 | 1 | 2, t: number) {
		if (!mesh) return;
		const d = enable
			? mesh.bbox.min[axis] + t * (mesh.bbox.max[axis] - mesh.bbox.min[axis])
			: Infinity;
		for (const pm of prefix_meshes) {
			const n = enable ? upper_bound(pm.vals, d) : pm.full_count / pm.vpu;
			const target = pm.line ? state.lineMeshes[pm.idx] : state.meshes[pm.idx];
			if (target) target.count = Math.round(n) * pm.vpu;
		}
	}

	/** Apply layer visibility and hidden_tags to the current GL state.
	 *  Prunes stale hidden_tags entries that no longer exist in the geometry. */
	function apply_layer_visibility(state: GLState) {
		const all_tags = new Set<number>();
		for (const m of state.meshes) all_tags.add(m.tag);
		const cur = untrack(() => hidden_tags);
		const next = new Set<number>();
		for (const t of cur) if (all_tags.has(t)) next.add(t);
		if (next.size !== cur.size) hidden_tags = next;
		const eff = next.size !== cur.size ? next : cur;
		for (const entry of state.meshes) {
			entry.visible = layer_surface && !eff.has(entry.tag);
		}
		setTagVisible(state, TAG_WIRE_SURF,  layer_wire);
		setTagVisible(state, TAG_FEAT_EDGES, layer_edges);
		setTagVisible(state, TAG_TET_WIRE,   layer_tets);
	}

	/** Upload sorted geometry for the given clip axis. Called once per axis
	 *  change (or mesh change). All layers (surface fills, surface wire, feature
	 *  edges, interior tet wire) are built and sorted by centroid along `axis`
	 *  so the crinkle-clip slider only needs to adjust draw counts. */
	function build_for_axis(state: GLState, m: MeshData, axis: 0 | 1 | 2) {
		clearMeshes(state);
		prefix_meshes = [];

		const np   = m.nodes;
		const nf   = m.tri_phys.length;
		const nt   = m.tet_phys.length;

		const face_cv = (fi: number): number =>
			(np[m.tris[fi*3]*3+axis] + np[m.tris[fi*3+1]*3+axis] + np[m.tris[fi*3+2]*3+axis]) / 3;
		const tet_cv = (ti: number): number =>
			(np[m.tets[ti*4]*3+axis] + np[m.tets[ti*4+1]*3+axis] +
			 np[m.tets[ti*4+2]*3+axis] + np[m.tets[ti*4+3]*3+axis]) / 4;

		// ---- Surface fills: explicit tri groups per physics tag ----
		const by_surf = new Map<number, number[]>();
		for (let f = 0; f < nf; f++) {
			const tag = m.tri_phys[f];
			if (!tag || (m.phys_dim.get(tag) ?? 2) !== 2) continue;
			let arr = by_surf.get(tag);
			if (!arr) { arr = []; by_surf.set(tag, arr); }
			arr.push(f);
		}
		for (const [tag, fis] of by_surf) {
			const name = m.phys_names.get(tag) ?? '';
			const kind = classify(name);
			if (!kind) continue;
			fis.sort((a, b) => face_cv(a) - face_cv(b));
			const flat_idx: number[] = new Array(fis.length * 3);
			for (let i = 0; i < fis.length; i++) {
				const fi = fis[i];
				flat_idx[i*3]   = m.tris[fi*3];
				flat_idx[i*3+1] = m.tris[fi*3+1];
				flat_idx[i*3+2] = m.tris[fi*3+2];
			}
			const pm_idx = state.meshes.length;
			const vals = Float64Array.from(fis, fi => face_cv(fi));
			prefix_meshes.push({ idx: pm_idx, line: false, vals, vpu: 3, full_count: fis.length * 3 });
			const { positions, normals } = buildTriSoupF64(np, flat_idx);
			addMesh(state, positions, normals, color_for(kind, name), tag);
		}

		// ---- Volume hulls (implicit surfaces from tet connectivity) ----
		const vol_b = build_volume_boundaries(m);
		for (const [vtag, idx] of vol_b.entries()) {
			const name = m.phys_names.get(vtag) ?? '';
			if (!name) continue;
			const kind = classify(name);
			if (!kind) continue;
			const ntri = idx.length / 3;
			const order = Array.from({ length: ntri }, (_, i) => i);
			order.sort((a, b) => {
				const ca = (np[idx[a*3]*3+axis] + np[idx[a*3+1]*3+axis] + np[idx[a*3+2]*3+axis]) / 3;
				const cb = (np[idx[b*3]*3+axis] + np[idx[b*3+1]*3+axis] + np[idx[b*3+2]*3+axis]) / 3;
				return ca - cb;
			});
			const sorted_idx: number[] = new Array(ntri * 3);
			const vals = new Float64Array(ntri);
			for (let i = 0; i < ntri; i++) {
				const t = order[i];
				sorted_idx[i*3]   = idx[t*3];
				sorted_idx[i*3+1] = idx[t*3+1];
				sorted_idx[i*3+2] = idx[t*3+2];
				vals[i] = (np[idx[t*3]*3+axis] + np[idx[t*3+1]*3+axis] + np[idx[t*3+2]*3+axis]) / 3;
			}
			const kind_offset: [number, number] | undefined = kind === 'dielectric' ? [2, 2] : undefined;
			const pm_idx = state.meshes.length;
			prefix_meshes.push({ idx: pm_idx, line: false, vals, vpu: 3, full_count: ntri * 3 });
			const { positions, normals } = buildTriSoupF64(np, sorted_idx);
			addMesh(state, positions, normals, color_for(kind, name), vtag, kind_offset);
		}

		// ---- Surface wireframe: edges from explicit surface tris ----
		const surf_edge_val = new Map<bigint, { a: number; b: number; val: number }>();
		for (let f = 0; f < nf; f++) {
			if (!m.tri_phys[f]) continue;
			const fv = face_cv(f);
			const ea = m.tris[f*3], eb = m.tris[f*3+1], ec = m.tris[f*3+2];
			const add_se = (u: number, w: number) => {
				const lo = u < w ? u : w, hi = u < w ? w : u;
				const k = (BigInt(lo) << 32n) | BigInt(hi);
				const cur = surf_edge_val.get(k);
				if (!cur) surf_edge_val.set(k, { a: u, b: w, val: fv });
				else if (fv < cur.val) cur.val = fv;
			};
			add_se(ea, eb); add_se(eb, ec); add_se(ec, ea);
		}
		const surf_edges = [...surf_edge_val.values()].sort((x, y) => x.val - y.val);
		{
			const pos = new Float32Array(surf_edges.length * 6);
			for (let i = 0; i < surf_edges.length; i++) {
				const e = surf_edges[i];
				pos[i*6]   = np[e.a*3];   pos[i*6+1] = np[e.a*3+1]; pos[i*6+2] = np[e.a*3+2];
				pos[i*6+3] = np[e.b*3];   pos[i*6+4] = np[e.b*3+1]; pos[i*6+5] = np[e.b*3+2];
			}
			const pm_idx = state.lineMeshes.length;
			const vals = Float64Array.from(surf_edges, e => e.val);
			prefix_meshes.push({ idx: pm_idx, line: true, vals, vpu: 2, full_count: surf_edges.length * 2 });
			addLineMesh(state, pos, WIRE_SURF_COLOR, TAG_WIRE_SURF);
		}

		// ---- Interior tet wireframe: tet edges NOT on surface ----
		const surf_edge_set = new Set<bigint>(surf_edge_val.keys());
		const int_edge_val = new Map<bigint, { a: number; b: number; val: number }>();
		for (let ti = 0; ti < nt; ti++) {
			const tv = tet_cv(ti);
			const v0 = m.tets[ti*4], v1 = m.tets[ti*4+1], v2 = m.tets[ti*4+2], v3 = m.tets[ti*4+3];
			const tet_verts = [v0, v1, v2, v3];
			for (let i = 0; i < 4; i++) {
				for (let j = i + 1; j < 4; j++) {
					const u = tet_verts[i], w = tet_verts[j];
					const lo = u < w ? u : w, hi = u < w ? w : u;
					const k = (BigInt(lo) << 32n) | BigInt(hi);
					if (surf_edge_set.has(k)) continue;
					const cur = int_edge_val.get(k);
					if (!cur) int_edge_val.set(k, { a: u, b: w, val: tv });
					else if (tv < cur.val) cur.val = tv;
				}
			}
		}
		const int_edges = [...int_edge_val.values()].sort((x, y) => x.val - y.val);
		{
			const pos = new Float32Array(int_edges.length * 6);
			for (let i = 0; i < int_edges.length; i++) {
				const e = int_edges[i];
				pos[i*6]   = np[e.a*3];   pos[i*6+1] = np[e.a*3+1]; pos[i*6+2] = np[e.a*3+2];
				pos[i*6+3] = np[e.b*3];   pos[i*6+4] = np[e.b*3+1]; pos[i*6+5] = np[e.b*3+2];
			}
			const pm_idx = state.lineMeshes.length;
			const vals = Float64Array.from(int_edges, e => e.val);
			prefix_meshes.push({ idx: pm_idx, line: true, vals, vpu: 2, full_count: int_edges.length * 2 });
			addLineMesh(state, pos, WIRE_INT_COLOR, TAG_TET_WIRE);
		}

		// ---- Feature edges from payload (not clipped, always full draw) ----
		if (m.edges && m.edges.length >= 2) {
			const ne = (m.edges.length / 2) | 0;
			const pos = new Float32Array(ne * 6);
			for (let i = 0; i < ne; i++) {
				const ea = m.edges[i*2], eb = m.edges[i*2+1];
				pos[i*6]   = np[ea*3];    pos[i*6+1] = np[ea*3+1]; pos[i*6+2] = np[ea*3+2];
				pos[i*6+3] = np[eb*3];    pos[i*6+4] = np[eb*3+1]; pos[i*6+5] = np[eb*3+2];
			}
			addLineMesh(state, pos, FEAT_EDGE_COLOR, TAG_FEAT_EDGES);
		}

		built_axis = axis;
	}

	function rebuild() {
		if (!gl_state) return;
		const m = mesh;
		if (!m) {
			clearMeshes(gl_state);
			last_built_for = null;
			needs_rebuild = false;
			return;
		}
		setBBox(gl_state, m.bbox.min, m.bbox.max);
		const needsGeoRebuild = last_built_for?.mesh !== m || last_built_for?.axis !== clip_axis;
		if (needsGeoRebuild) {
			build_for_axis(gl_state, m, clip_axis);
			last_built_for = { mesh: m, axis: clip_axis };
		}
		apply_clip_counts(gl_state, clip_enable, clip_axis, clip_t);
		apply_layer_visibility(gl_state);
		needs_rebuild = false;
	}

	// ── Frame loop / sizing ─────────────────────────────────────────────
	function get_size(): { w: number; h: number } {
		if (!container) return { w: 0, h: 0 };
		const r = container.getBoundingClientRect();
		return { w: Math.round(r.width), h: Math.round(r.height) };
	}
	function sync_canvas(): { w: number; h: number } {
		const { w, h } = get_size();
		if (w <= 0 || h <= 0 || !canvas) return { w, h };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
		if (canvas.width !== bw || canvas.height !== bh) {
			canvas.width = bw;
			canvas.height = bh;
			canvas.style.width = w + 'px';
			canvas.style.height = h + 'px';
		}
		return { w: bw, h: bh };
	}
	function render_frame() {
		if (!gl_state || !canvas) return;
		const { w, h } = sync_canvas();
		if (w <= 0 || h <= 0) return;
		if (needs_rebuild) rebuild();
		render3D(gl_state, camera, w, h, z_flip);
	}

	// Single always-on rAF loop: advances the idle orbit (when enabled, not
	// dragging, and no toolbar animation in flight) and renders every frame.
	let raf = 0;
	let last_t = 0;
	function loop(t: number) {
		if (gl_state) {
			if (orbit && !is_dragging && anim_target === null && last_t) {
				camera = { ...camera, theta: camera.theta + (ORBIT_SPEED * (t - last_t)) / 1000 };
			}
			last_t = t;
			render_frame();
		}
		raf = requestAnimationFrame(loop);
	}

	// ── Pointer / wheel handlers (orbit/pan/zoom) ───────────────────────
	function on_wheel(e: WheelEvent) {
		e.preventDefault();
		notify_interact();
		const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
		camera = { ...camera, distance: camera.distance * factor };
	}
	function on_pointer_down(e: PointerEvent) {
		notify_interact();
		anim_id++;          // cancel any in-flight camera animation
		anim_target = null;
		is_dragging = true;
		is_right_drag = e.button === 2;
		last_mouse = { x: e.clientX, y: e.clientY };
		canvas?.setPointerCapture(e.pointerId);
	}
	function on_pointer_move(e: PointerEvent) {
		// HUD coords (project to z=target plane)
		if (canvas) {
			const r = canvas.getBoundingClientRect();
			const mx = e.clientX - r.left, my = e.clientY - r.top;
			const { w, h } = get_size();
			const halfH = camera.distance * Math.tan(Math.PI / 12);
			const halfW = halfH * (w / h || 1);
			const nx = (mx / w - 0.5) * 2;
			const ny = -(my / h - 0.5) * 2;
			const ct = Math.cos(camera.theta), st = Math.sin(camera.theta);
			cursor_world = {
				x: camera.target[0] + nx * halfW * ct + ny * halfH * st * Math.sin(camera.phi),
				y: camera.target[1] - nx * halfW * st + ny * halfH * ct * Math.sin(camera.phi)
			};
		}
		if (!is_dragging) return;
		const dx = e.clientX - last_mouse.x;
		const dy = e.clientY - last_mouse.y;
		last_mouse = { x: e.clientX, y: e.clientY };
		if (is_right_drag) {
			const panScale = camera.distance * 0.0007;
			const ct = Math.cos(camera.theta), st = Math.sin(camera.theta);
			camera = {
				...camera,
				target: [
					camera.target[0] + (dx * ct - dy * st * Math.sin(camera.phi)) * panScale,
					camera.target[1] - (dx * st + dy * ct * Math.sin(camera.phi)) * panScale,
					camera.target[2] + dy * Math.cos(camera.phi) * panScale
				]
			};
		} else {
			camera = {
				...camera,
				theta: camera.theta + dx * 0.005,
				phi: Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.005))
			};
		}
	}
	function on_pointer_up() { is_dragging = false; is_right_drag = false; }
	function on_context_menu(e: Event) { e.preventDefault(); }
	function on_dbl_click() { fit_view(); }

	// ── Lifecycle ───────────────────────────────────────────────────────
	onMount(() => {
		mounted = true;
		if (!canvas) return;
		gl_state = initGL(canvas);
		if (!gl_state) return;

		const ro = new ResizeObserver(() => { /* loop renders every frame */ });
		if (container) ro.observe(container);

		if (mesh) {
			camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
			needs_rebuild = true;
		}
		raf = requestAnimationFrame(loop);

		return () => {
			mounted = false;
			if (raf) cancelAnimationFrame(raf);
			ro.disconnect();
			if (gl_state) disposeGL(gl_state);
			gl_state = null;
		};
	});

	// React to mesh / layer / clip-axis changes (full geometry rebuild).
	// clip_t is excluded — the fast clip_t effect handles slider drags.
	$effect(() => {
		mesh;
		layer_surface; layer_wire; layer_edges; layer_tets;
		clip_enable; clip_axis;
		if (!mounted || !gl_state) return;
		needs_rebuild = true;
	});

	// Fast clip-t path: only adjusts draw counts on the already-sorted buffers.
	$effect(() => {
		const ct = clip_t;
		if (!gl_state || !mesh || !mounted) return;
		if (prefix_meshes.length === 0 || built_axis < 0) return;
		const ce = untrack(() => clip_enable);
		const ca = untrack(() => clip_axis);
		if (ca !== built_axis) return;
		apply_clip_counts(gl_state, ce, ca, ct);
	});

	// Refit the camera when the visible mesh changes.
	$effect(() => {
		const m = mesh;
		if (!mounted || !m) return;
		camera = fitCamera(m.bbox.min, m.bbox.max);
		needs_rebuild = true;
	});

	// ── Legend (per-group toggles) ──────────────────────────────────────
	const tag_legend = $derived.by(() => {
		const m = mesh;
		if (!m) return [] as { name: string; color: string; kind: Kind; rank: number; tag: number }[];
		const seen = new Set<number>();
		const items: { name: string; color: string; kind: Kind; rank: number; tag: number }[] = [];
		const add = (tag: number, kind: Kind) => {
			if (seen.has(tag)) return;
			seen.add(tag);
			const name = m.phys_names.get(tag) ?? '';
			if (!name) return;
			if (classify(name) === null) return;
			const c = color_for(kind, name);
			const rank = kind === 'conductor' ? 0 : kind === 'port' ? 1 : kind === 'gnd' ? 2 : 3;
			items.push({
				name: pretty_label(name),
				color: `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`,
				kind, rank, tag
			});
		};
		for (let i = 0; i < m.tri_phys.length; i++) {
			const tag = m.tri_phys[i];
			if (!tag || (m.phys_dim.get(tag) ?? 2) !== 2) continue;
			const k = classify(m.phys_names.get(tag) ?? '');
			if (k) add(tag, k);
		}
		for (let i = 0; i < m.tet_phys.length; i++) {
			const tag = m.tet_phys[i];
			const name = m.phys_names.get(tag) ?? '';
			const k = classify(name);
			if (k) add(tag, k);
		}
		items.sort((a, b) => a.rank - b.rank);
		return items;
	});
</script>

<div class="viewer" bind:this={container}>
	<canvas
		bind:this={canvas}
		onwheel={on_wheel}
		onpointerdown={on_pointer_down}
		onpointermove={on_pointer_move}
		onpointerup={on_pointer_up}
		oncontextmenu={on_context_menu}
		ondblclick={on_dbl_click}
	></canvas>

	<div class="overlay-stack">
		{#if tag_legend.length > 0}
			<div class="overlay-panel">
				<div class="op-title">Geometry</div>
				<div class="op-body">
					{#each tag_legend as l}
						<button
							class="legend-item"
							class:hidden={hidden_tags.has(l.tag)}
							onclick={() => toggle_tag(l.tag)}
							title="Click to toggle"
						>
							<span class="swatch" style="background: {l.color};"></span>
							<span class="legend-name">{l.name}</span>
						</button>
					{/each}
				</div>
			</div>
		{/if}
	</div>

	<div class="viewer-toolbar">
		<button class="tb" onclick={zoom_in}><span class="tip">Zoom in<kbd>+</kbd></span>+</button>
		<button class="tb" onclick={zoom_out}><span class="tip">Zoom out<kbd>−</kbd></span>−</button>
		<button class="tb" onclick={fit_view}>
			<span class="tip">Fit view<kbd>F</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<polyline points="1,5 1,1 5,1" /><polyline points="11,1 15,1 15,5" />
				<polyline points="15,11 15,15 11,15" /><polyline points="5,15 1,15 1,11" />
				<rect x="5" y="5" width="6" height="6" rx="0.5" />
			</svg>
		</button>
		<button class="tb" onclick={rotate_90}>
			<span class="tip">Rotate<kbd>R</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
				<polyline points="15.3 2.7 15.3 6.7 11.3 6.7" />
				<path d="M13.66 10a6 6 0 1 1-1.41-6.24L15.3 6.7" />
			</svg>
		</button>
		<button class="tb" onclick={save_png}>
			<span class="tip">Save PNG<kbd>Ctrl+S</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M2 10v3h12v-3" /><path d="M8 2v8" /><path d="M5 7l3 3 3-3" />
			</svg>
		</button>
		{#if mesh}
			<span class="tb-sep" aria-hidden="true"></span>
			<button class="tb tb-label" class:active={layer_surface}
				onclick={() => { notify_interact(); layer_surface = !layer_surface; }}>
				<span class="tip">Toggle filled surface</span>Surf
			</button>
			<button class="tb tb-label" class:active={layer_wire}
				onclick={() => { notify_interact(); layer_wire = !layer_wire; }}>
				<span class="tip">Toggle surface wireframe</span>Wire
			</button>
			<button class="tb tb-label" class:active={layer_edges}
				onclick={() => { notify_interact(); layer_edges = !layer_edges; }}>
				<span class="tip">Toggle feature edges</span>Edge
			</button>
			<button class="tb tb-label" class:active={layer_tets}
				onclick={() => { notify_interact(); layer_tets = !layer_tets; }}>
				<span class="tip">Toggle interior tet wireframe</span>Tets
			</button>
			<span class="tb-sep" aria-hidden="true"></span>
			<button class="tb tb-label" class:active={clip_enable}
				onclick={() => { notify_interact(); clip_enable = !clip_enable; }}>
				<span class="tip">Crinkle clip (whole-tet by centroid)</span>Clip
			</button>
			{#if clip_enable}
				{#each CLIP_AXES as { lbl, ax }}
					<button class="tb tb-label" class:active={clip_axis === ax}
						onclick={() => { notify_interact(); clip_axis = ax; }}>
						<span class="tip">Clip along {lbl} axis</span>{lbl}
					</button>
				{/each}
				<div class="clip-row">
					<input type="range" class="clip-slider" min="0" max="1" step="0.001"
						bind:value={clip_t} oninput={notify_interact} />
				</div>
			{/if}
		{/if}
	</div>

	<div class="hud">
		<span class="coord">x {(cursor_world.x * 1e6).toFixed(1)} µm</span>
		<span class="coord">y {(cursor_world.y * 1e6).toFixed(1)} µm</span>
		{#if mesh}
			<span class="coord stats">{(mesh.nodes.length / 3) | 0}n · {(mesh.tris.length / 3) | 0}t · {(mesh.tets.length / 4) | 0}T</span>
		{/if}
		{#if mesh?.stats?.n_edges != null}
			<span class="coord stats">{mesh.stats.n_edges}e</span>
		{/if}
		{#if mesh?.stats?.min_dihedral_deg != null}
			<span class="coord stats">min&#8736; {mesh.stats.min_dihedral_deg.toFixed(1)}&#176;</span>
		{/if}
	</div>

</div>

<style>
	.viewer {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 320px;
		background: var(--canvas-bg);
		overflow: hidden;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		cursor: grab;
	}
	canvas:active { cursor: grabbing; }

	.overlay-stack {
		position: absolute;
		top: 10px;
		left: 10px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		max-height: calc(100% - 20px);
	}
	.overlay-panel {
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		padding: 8px 10px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 96px;
	}
	.op-title {
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 1.5px;
		color: var(--accent);
		font-weight: 600;
	}
	.op-body {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 4px;
		margin: 0 -4px;
		background: transparent;
		border: 0;
		color: var(--text-muted);
		cursor: pointer;
		text-align: left;
		font-family: inherit;
		font-size: inherit;
		text-transform: none;
		letter-spacing: 0;
		transition: background var(--transition), color var(--transition);
	}
	.legend-item:hover { background: var(--accent-dim); color: var(--text); }
	.legend-item.hidden { color: var(--text-dim); }
	.legend-item.hidden .swatch { opacity: 0.25; }
	.legend-item.hidden .legend-name { text-decoration: line-through; }
	.swatch { width: 10px; height: 10px; flex-shrink: 0; transition: opacity var(--transition); }

	.viewer-toolbar {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 10;
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 2px;
		max-width: calc(100% - 20px);
	}
	.tb-sep {
		display: inline-block;
		width: 1px;
		height: 20px;
		margin: 4px 4px;
		background: var(--border);
	}
	.tb.tb-label {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.tb.tb-label.active {
		color: var(--accent);
		background: var(--accent-dim);
		border-color: var(--accent);
	}
	.tb.tb-label:disabled {
		color: var(--text-dim);
		cursor: default;
		opacity: 0.4;
		border-color: var(--border);
		background: var(--bg-surface);
	}
	.tb {
		position: relative;
		width: 28px;
		height: 28px;
		border: 1px solid var(--border);
		background: var(--bg-surface);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tb:hover { background: var(--bg-panel); border-color: var(--accent); color: var(--text); }
	.tb:disabled { opacity: 0.6; cursor: progress; }
	.tb .tip {
		display: none;
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		white-space: nowrap;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		font-weight: 400;
		color: var(--text-muted);
		background: var(--bg-surface);
		border: 1px solid var(--border);
		padding: 3px 8px;
		pointer-events: none;
		z-index: 20;
	}
	.tb .tip kbd { margin-left: 6px; color: var(--accent); font-weight: 600; }
	.tb:hover .tip { display: flex; align-items: center; gap: 4px; }

	.hud {
		position: absolute;
		bottom: 8px;
		left: 8px;
		display: flex;
		gap: 12px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		pointer-events: none;
	}
	.hud .stats { color: var(--text-muted); }

	/* Crinkle clip slider row: full-width so it wraps to its own line in the
	   flex-wrap toolbar. The range input fills the available width. */
	.clip-row {
		width: 100%;
		display: flex;
		align-items: center;
		padding: 2px 0 0;
	}
	.clip-slider {
		width: 100%;
		height: 4px;
		accent-color: var(--accent);
		cursor: pointer;
		appearance: auto;
	}
</style>
