<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import {
		addLineMesh,
		addMesh,
		clearMeshes,
		disposeGL,
		fitCamera,
		initGL,
		render3D,
		setBBox,
		setTagVisible,
		type GLState
	} from '$lib/render/canvas3d';
	import { canvas as canvasTheme, palette, plotColors } from '$lib/theme';
	import {
		animate_camera,
		camera,
		cancel_camera_animation,
		effective_camera,
		register_renderer,
		render_all
	} from '$lib/viewbus';
	import type { MeshJson, ViewSettings } from '$lib/mesh_types';

	let {
		data,
		settings,
		bbox
	}: {
		data: MeshJson;
		settings: ViewSettings;
		bbox: { min: [number, number, number]; max: [number, number, number] };
	} = $props();

	// Display tags for visibility toggles.
	const TAG_SURFACE = 1;
	const TAG_TETFILL = 2;
	const TAG_WIRE_SURFACE = 3;
	const TAG_WIRE_TETS = 4;

	function hexToRgb(hex: string): [number, number, number] {
		return [
			parseInt(hex.slice(1, 3), 16) / 255,
			parseInt(hex.slice(3, 5), 16) / 255,
			parseInt(hex.slice(5, 7), 16) / 255
		];
	}
	const regionCycle = plotColors.cycle.map(hexToRgb);
	const pecColor = hexToRgb(palette.accentSecondary);
	// Wireframe colors are rapidfem canvas tokens: crosshair is what
	// scene_builder uses for its wireframe overlay; grid for the dimmer
	// interior edges.
	const wireSurface = hexToRgb(canvasTheme.crosshair);
	const wireInterior = hexToRgb(canvasTheme.grid);
	const regionColor = (r: number) => regionCycle[(r + regionCycle.length - 1) % regionCycle.length];

	let container: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	// Reactive: the settings $effect runs before initGL and must re-run once
	// the GL state exists (a plain let would leave it with no tracked
	// dependencies after the early return, never re-running).
	let gl_state: GLState | null = $state(null);
	let is_dragging = false;
	let is_right_drag = false;
	let last_mouse = { x: 0, y: 0 };
	let unregister: (() => void) | null = null;

	// ── Canvas sizing (verbatim pattern from MeshViewer) ────────────────
	function get_size(): { w: number; h: number } {
		if (!container) return { w: 0, h: 0 };
		const r = container.getBoundingClientRect();
		return { w: Math.round(r.width), h: Math.round(r.height) };
	}
	function sync_canvas(): { w: number; h: number } {
		const { w, h } = get_size();
		if (w <= 0 || h <= 0 || !canvas) return { w, h };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(w * dpr),
			bh = Math.round(h * dpr);
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
		render3D(gl_state, camera, w, h);
	}

	// ── Geometry upload ─────────────────────────────────────────────────
	function triNormal(a: number, b: number, c: number): [number, number, number] {
		const pts = data.points;
		const u = [pts[b][0] - pts[a][0], pts[b][1] - pts[a][1], pts[b][2] - pts[a][2]];
		const v = [pts[c][0] - pts[a][0], pts[c][1] - pts[a][1], pts[c][2] - pts[a][2]];
		const n: [number, number, number] = [
			u[1] * v[2] - u[2] * v[1],
			u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0]
		];
		const l = Math.hypot(n[0], n[1], n[2]) || 1;
		return [n[0] / l, n[1] / l, n[2] / l];
	}

	function pushFace(pos: number[], nrm: number[], a: number, b: number, c: number) {
		const pts = data.points;
		const n = triNormal(a, b, c);
		for (const v of [a, b, c]) {
			pos.push(pts[v][0], pts[v][1], pts[v][2]);
			nrm.push(n[0], n[1], n[2]);
		}
	}

	// ── Crinkle clip precomputation ─────────────────────────────────────
	// Whole tets are kept or dropped by their centroid against the clip
	// plane (ParaView's crinkle clip): closed tetrahedra at the cut instead
	// of fragment-sliced ones. The fragment clip of canvas3d stays unused.
	const tet_centroid: [number, number, number][] = data.tets.map((t) => {
		const p = data.points;
		return [
			(p[t[0]][0] + p[t[1]][0] + p[t[2]][0] + p[t[3]][0]) / 4,
			(p[t[0]][1] + p[t[1]][1] + p[t[2]][1] + p[t[3]][1]) / 4,
			(p[t[0]][2] + p[t[1]][2] + p[t[2]][2] + p[t[3]][2]) / 4
		];
	});
	const face_centroid: [number, number, number][] = data.faces.map((f) => {
		const p = data.points;
		return [
			(p[f.tri[0]][0] + p[f.tri[1]][0] + p[f.tri[2]][0]) / 3,
			(p[f.tri[0]][1] + p[f.tri[1]][1] + p[f.tri[2]][1]) / 3,
			(p[f.tri[0]][2] + p[f.tri[1]][2] + p[f.tri[2]][2]) / 3
		];
	});

	// ── Prefix-sorted geometry for O(log n) crinkle clipping ────────────
	// The clip plane is axis-aligned, so the kept tets/faces are exactly a
	// PREFIX of the geometry sorted by centroid along that axis. Buffers are
	// uploaded once per axis in sorted order; moving the slider only binary-
	// searches the cut value and adjusts each draw count. No re-uploads.
	interface PrefixMesh {
		idx: number; // index into state.meshes / state.lineMeshes
		line: boolean;
		vals: Float64Array; // sorted unit centroid values along built_axis
		vpu: number; // vertices per unit (12 per tet, 3 per face, 2 per edge)
	}
	let prefix_meshes: PrefixMesh[] = [];
	let built_axis = -1;

	function upper_bound(vals: Float64Array, d: number): number {
		let lo = 0,
			hi = vals.length;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (vals[mid] <= d) lo = mid + 1;
			else hi = mid;
		}
		return lo;
	}

	function build_for_axis(state: GLState, axis: 0 | 1 | 2) {
		clearMeshes(state);
		prefix_meshes = [];
		const pts = data.points;

		// Surface faces grouped by color, each group sorted by face centroid.
		const groups = new Map<string, { color: [number, number, number]; fis: number[] }>();
		for (let fi = 0; fi < data.faces.length; fi++) {
			const f = data.faces[fi];
			const color = f.tag !== 0 ? pecColor : regionColor(Math.max(f.regions[0], f.regions[1]));
			const key = color.join(',');
			if (!groups.has(key)) groups.set(key, { color, fis: [] });
			groups.get(key)!.fis.push(fi);
		}
		for (const g of groups.values()) {
			g.fis.sort((a, b) => face_centroid[a][axis] - face_centroid[b][axis]);
			const pos: number[] = [];
			const nrm: number[] = [];
			for (const fi of g.fis) {
				const [a, b, c] = data.faces[fi].tri;
				pushFace(pos, nrm, a, b, c);
			}
			prefix_meshes.push({
				idx: state.meshes.length,
				line: false,
				vals: Float64Array.from(g.fis, (fi) => face_centroid[fi][axis]),
				vpu: 3
			});
			addMesh(state, new Float32Array(pos), new Float32Array(nrm), g.color, TAG_SURFACE, [-1, -1]);
		}

		// Tet fill per region, sorted by tet centroid.
		const byRegion = new Map<number, number[]>();
		for (let ti = 0; ti < data.tets.length; ti++) {
			const r = data.tet_regions[ti];
			if (!byRegion.has(r)) byRegion.set(r, []);
			byRegion.get(r)!.push(ti);
		}
		for (const [r, tis] of byRegion) {
			tis.sort((a, b) => tet_centroid[a][axis] - tet_centroid[b][axis]);
			const pos: number[] = [];
			const nrm: number[] = [];
			for (const ti of tis) {
				const t = data.tets[ti];
				pushFace(pos, nrm, t[1], t[3], t[2]);
				pushFace(pos, nrm, t[0], t[2], t[3]);
				pushFace(pos, nrm, t[0], t[3], t[1]);
				pushFace(pos, nrm, t[0], t[1], t[2]);
			}
			prefix_meshes.push({
				idx: state.meshes.length,
				line: false,
				vals: Float64Array.from(tis, (ti) => tet_centroid[ti][axis]),
				vpu: 12
			});
			addMesh(state, new Float32Array(pos), new Float32Array(nrm), regionColor(r), TAG_TETFILL);
		}

		// Surface wireframe: an edge follows the first (smallest) adjacent
		// face, so it appears exactly when some adjacent face is kept.
		const surfEdgeVal = new Map<string, { a: number; b: number; val: number }>();
		for (let fi = 0; fi < data.faces.length; fi++) {
			const f = data.faces[fi];
			const v = face_centroid[fi][axis];
			for (let e = 0; e < 3; e++) {
				const a = f.tri[e],
					b = f.tri[(e + 1) % 3];
				const key = a < b ? `${a},${b}` : `${b},${a}`;
				const cur = surfEdgeVal.get(key);
				if (!cur) surfEdgeVal.set(key, { a, b, val: v });
				else cur.val = Math.min(cur.val, v);
			}
		}
		const push_edge_mesh = (
			entries: { a: number; b: number; val: number }[],
			color: [number, number, number],
			tag: number
		) => {
			entries.sort((x, y) => x.val - y.val);
			const pos: number[] = [];
			for (const e of entries) {
				pos.push(pts[e.a][0], pts[e.a][1], pts[e.a][2], pts[e.b][0], pts[e.b][1], pts[e.b][2]);
			}
			prefix_meshes.push({
				idx: state.lineMeshes.length,
				line: true,
				vals: Float64Array.from(entries, (e) => e.val),
				vpu: 2
			});
			addLineMesh(state, new Float32Array(pos), color, tag);
		};
		push_edge_mesh([...surfEdgeVal.values()], wireSurface, TAG_WIRE_SURFACE);

		// Interior edges: follow the smallest adjacent tet.
		const intEdgeVal = new Map<string, { a: number; b: number; val: number }>();
		for (let ti = 0; ti < data.tets.length; ti++) {
			const t = data.tets[ti];
			const v = tet_centroid[ti][axis];
			for (let i = 0; i < 4; i++) {
				for (let j = i + 1; j < 4; j++) {
					const a = t[i],
						b = t[j];
					const key = a < b ? `${a},${b}` : `${b},${a}`;
					if (surfEdgeVal.has(key)) continue;
					const cur = intEdgeVal.get(key);
					if (!cur) intEdgeVal.set(key, { a, b, val: v });
					else cur.val = Math.min(cur.val, v);
				}
			}
		}
		push_edge_mesh([...intEdgeVal.values()], wireInterior, TAG_WIRE_TETS);
	}

	function apply_clip(state: GLState, clip_enable: boolean, clip_axis: 0 | 1 | 2, clip_t: number) {
		if (clip_axis !== built_axis) {
			build_for_axis(state, clip_axis);
			built_axis = clip_axis;
		}
		const d = clip_enable
			? bbox.min[clip_axis] + clip_t * (bbox.max[clip_axis] - bbox.min[clip_axis])
			: Infinity;
		for (const pm of prefix_meshes) {
			const n = clip_enable ? upper_bound(pm.vals, d) : pm.vals.length;
			const target = pm.line ? state.lineMeshes[pm.idx] : state.meshes[pm.idx];
			target.count = n * pm.vpu;
		}
	}

	// ── Settings → GL state ─────────────────────────────────────────────
	$effect(() => {
		// Read every dependency up front so they are tracked even while the
		// GL state is not ready yet.
		const surface = settings.surface;
		const surface_wire = settings.surface_wire;
		const tet_fill = settings.tet_fill;
		const tet_wire = settings.tet_wire;
		const clip_enable = settings.clip_enable;
		const clip_axis = settings.clip_axis;
		const clip_t = settings.clip_t;
		if (!gl_state) return;
		apply_clip(gl_state, clip_enable, clip_axis, clip_t);
		setTagVisible(gl_state, TAG_SURFACE, surface);
		setTagVisible(gl_state, TAG_WIRE_SURFACE, surface_wire);
		setTagVisible(gl_state, TAG_TETFILL, tet_fill);
		setTagVisible(gl_state, TAG_WIRE_TETS, tet_wire);
		render_all();
	});

	// ── Pointer / wheel handlers (verbatim from MeshViewer, on the shared
	//    camera so all panels stay in sync) ─────────────────────────────
	function on_wheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
		camera.distance *= factor;
		render_all();
	}
	function on_pointer_down(e: PointerEvent) {
		cancel_camera_animation();
		is_dragging = true;
		is_right_drag = e.button === 2;
		last_mouse = { x: e.clientX, y: e.clientY };
		canvas?.setPointerCapture(e.pointerId);
	}
	function on_pointer_move(e: PointerEvent) {
		if (!is_dragging) return;
		const dx = e.clientX - last_mouse.x;
		const dy = e.clientY - last_mouse.y;
		last_mouse = { x: e.clientX, y: e.clientY };
		if (is_right_drag) {
			const panScale = camera.distance * 0.0007;
			const ct = Math.cos(camera.theta),
				st = Math.sin(camera.theta);
			camera.target = [
				camera.target[0] + (dx * ct - dy * st * Math.sin(camera.phi)) * panScale,
				camera.target[1] - (dx * st + dy * ct * Math.sin(camera.phi)) * panScale,
				camera.target[2] + dy * Math.cos(camera.phi) * panScale
			];
		} else {
			camera.theta += dx * 0.005;
			camera.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.005));
		}
		render_all();
	}
	function on_pointer_up() {
		is_dragging = false;
		is_right_drag = false;
	}
	function on_context_menu(e: Event) {
		e.preventDefault();
	}
	function on_dbl_click() {
		fit_view();
	}

	// ── Toolbar actions (animated, same durations as MeshViewer) ────────
	function zoom_in() {
		const base = effective_camera();
		animate_camera(
			{ ...base, target: [...base.target] as [number, number, number], distance: base.distance / 1.3 },
			200
		);
	}
	function zoom_out() {
		const base = effective_camera();
		animate_camera(
			{ ...base, target: [...base.target] as [number, number, number], distance: base.distance * 1.3 },
			200
		);
	}
	function fit_view() {
		animate_camera(fitCamera(bbox.min, bbox.max), 350);
	}
	function rotate_90() {
		const base = effective_camera();
		animate_camera(
			{ ...base, target: [...base.target] as [number, number, number], theta: base.theta + Math.PI / 2 },
			400
		);
	}
	function save_png() {
		if (!canvas) return;
		render_frame();
		canvas.toBlob((blob) => {
			if (!blob) return;
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `${data.mesher}-${data.name}.png`;
			a.click();
			URL.revokeObjectURL(url);
		});
	}

	onMount(() => {
		if (!canvas) return;
		gl_state = initGL(canvas);
		if (!gl_state) return;
		const min: [number, number, number] = [...bbox.min];
		const max: [number, number, number] = [...bbox.max];
		setBBox(gl_state, min, max);
		// Geometry upload happens in the settings effect (rebuild_clipped),
		// which re-runs now that gl_state exists.
		unregister = register_renderer(render_frame);
		const ro = new ResizeObserver(() => render_all());
		ro.observe(container!);
		render_all();
		return () => ro.disconnect();
	});

	onDestroy(() => {
		unregister?.();
		if (gl_state) disposeGL(gl_state);
		gl_state = null;
	});

	const s = $derived(data.stats);
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

	<div class="viewer-toolbar">
		<button class="tb" onclick={zoom_in}><span class="tip">Zoom in<kbd>+</kbd></span>+</button>
		<button class="tb" onclick={zoom_out}><span class="tip">Zoom out<kbd>-</kbd></span>-</button>
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
			<span class="tip">Save PNG</span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M2 10v3h12v-3" /><path d="M8 2v8" /><path d="M5 7l3 3 3-3" />
			</svg>
		</button>
	</div>

	<div class="hud">
		<span class="mesher">{data.mesher}</span>
		<span class="stats">
			{s.n_tets} tets · {s.n_points} pts · min∠ {s.min_dihedral_deg.toFixed(1)}° ·
			r/e {s.max_radius_edge.toFixed(2)} · {s.millis} ms
		</span>
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
	canvas:active {
		cursor: grabbing;
	}

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
	.tb:hover {
		background: var(--bg-panel);
		border-color: var(--accent);
		color: var(--text);
	}
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
	.tb .tip kbd {
		margin-left: 6px;
		color: var(--accent);
		font-weight: 600;
	}
	.tb:hover .tip {
		display: flex;
		align-items: center;
		gap: 4px;
	}

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
	.hud .mesher {
		color: var(--accent);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 1.5px;
	}
	.hud .stats {
		color: var(--text-muted);
	}
</style>
