<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import {
		addLineMesh,
		addMesh,
		disposeGL,
		fitCamera,
		initGL,
		render3D,
		setBBox,
		setClipPlane,
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
	const wireSurface = hexToRgb(canvasTheme.bg);
	const wireTets = hexToRgb(palette.accentPurple);
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

	function build_meshes(state: GLState) {
		const pts = data.points;

		// Surface faces grouped by color (tagged sheets win over regions).
		const groups = new Map<string, { color: [number, number, number]; tris: [number, number, number][] }>();
		for (const f of data.faces) {
			const color = f.tag !== 0 ? pecColor : regionColor(Math.max(f.regions[0], f.regions[1]));
			const key = color.join(',');
			if (!groups.has(key)) groups.set(key, { color, tris: [] });
			groups.get(key)!.tris.push(f.tri);
		}
		for (const g of groups.values()) {
			const pos: number[] = [];
			const nrm: number[] = [];
			for (const [a, b, c] of g.tris) pushFace(pos, nrm, a, b, c);
			addMesh(state, new Float32Array(pos), new Float32Array(nrm), g.color, TAG_SURFACE, [-1, -1]);
		}

		// Tet fill (all faces, per region): interior becomes visible under clip.
		const byRegion = new Map<number, number[]>();
		for (let ti = 0; ti < data.tets.length; ti++) {
			const r = data.tet_regions[ti];
			if (!byRegion.has(r)) byRegion.set(r, []);
			byRegion.get(r)!.push(ti);
		}
		for (const [r, tis] of byRegion) {
			const pos: number[] = [];
			const nrm: number[] = [];
			for (const ti of tis) {
				const t = data.tets[ti];
				pushFace(pos, nrm, t[1], t[3], t[2]);
				pushFace(pos, nrm, t[0], t[2], t[3]);
				pushFace(pos, nrm, t[0], t[3], t[1]);
				pushFace(pos, nrm, t[0], t[1], t[2]);
			}
			addMesh(state, new Float32Array(pos), new Float32Array(nrm), regionColor(r), TAG_TETFILL);
		}

		// Wireframes: surface edges, then all tet edges (deduplicated).
		const lines = (edges: Iterable<[number, number]>) => {
			const out: number[] = [];
			for (const [a, b] of edges) {
				out.push(pts[a][0], pts[a][1], pts[a][2], pts[b][0], pts[b][1], pts[b][2]);
			}
			return new Float32Array(out);
		};
		const surfEdges = new Map<string, [number, number]>();
		for (const f of data.faces) {
			for (let e = 0; e < 3; e++) {
				const a = f.tri[e],
					b = f.tri[(e + 1) % 3];
				surfEdges.set(a < b ? `${a},${b}` : `${b},${a}`, [a, b]);
			}
		}
		addLineMesh(state, lines(surfEdges.values()), wireSurface, TAG_WIRE_SURFACE);

		const tetEdges = new Map<string, [number, number]>();
		for (const t of data.tets) {
			for (let i = 0; i < 4; i++) {
				for (let j = i + 1; j < 4; j++) {
					const a = t[i],
						b = t[j];
					tetEdges.set(a < b ? `${a},${b}` : `${b},${a}`, [a, b]);
				}
			}
		}
		addLineMesh(state, lines(tetEdges.values()), wireTets, TAG_WIRE_TETS);
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
		const lo = bbox.min[clip_axis];
		const hi = bbox.max[clip_axis];
		if (!gl_state) return;
		setTagVisible(gl_state, TAG_SURFACE, surface);
		setTagVisible(gl_state, TAG_WIRE_SURFACE, surface_wire);
		setTagVisible(gl_state, TAG_TETFILL, tet_fill);
		setTagVisible(gl_state, TAG_WIRE_TETS, tet_wire);
		const normal: [number, number, number] = [0, 0, 0];
		normal[clip_axis] = 1;
		setClipPlane(gl_state, normal, lo + clip_t * (hi - lo), clip_enable);
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
		build_meshes(gl_state);
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
