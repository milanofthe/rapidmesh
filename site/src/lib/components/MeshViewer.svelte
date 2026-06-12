<script lang="ts">
	/**
	 * Fullscreen showcase viewer for mesh.rapidpassives.org.
	 *
	 * Reuses the shared WebGL renderer ($lib/render/canvas3d) and shared camera
	 * bus ($lib/viewbus) from the dev viewer (rapidmesh/viewer, source of truth).
	 * The GL setup, sizing, pointer-control and geometry-upload logic are derived
	 * from rapidmesh/viewer/src/lib/components/MeshPanel.svelte; here it is trimmed
	 * to a single chrome-free canvas that fills its parent, renders the surface as
	 * a dim lit fill with a bright wireframe overlay, and slowly orbits while idle.
	 */
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
		type GLState
	} from '$lib/render/canvas3d';
	import { plotColors } from '$lib/theme';
	import { camera } from '$lib/viewbus';
	import { ORBIT_SPEED, SURFACE_FILL_DIM, SURFACE_WIRE_COLOR } from '$lib/constants';
	import type { MeshJson } from '$lib/mesh_types';

	let {
		data,
		orbit = true,
		oninteract
	}: {
		data: MeshJson | null;
		orbit?: boolean;
		oninteract?: () => void;
	} = $props();

	const TAG_FILL = 1;
	const TAG_WIRE = 2;

	function hexToRgb(hex: string): [number, number, number] {
		return [
			parseInt(hex.slice(1, 3), 16) / 255,
			parseInt(hex.slice(3, 5), 16) / 255,
			parseInt(hex.slice(5, 7), 16) / 255
		];
	}
	const regionCycle = plotColors.cycle.map(hexToRgb);
	const wireColor = hexToRgb(SURFACE_WIRE_COLOR);
	const regionColor = (r: number): [number, number, number] => {
		const c = regionCycle[((r % regionCycle.length) + regionCycle.length) % regionCycle.length];
		return [c[0] * SURFACE_FILL_DIM, c[1] * SURFACE_FILL_DIM, c[2] * SURFACE_FILL_DIM];
	};

	let container: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let gl_state: GLState | null = $state(null);
	let dragging = false;
	let right_drag = false;
	let last_mouse = { x: 0, y: 0 };
	let raf = 0;
	let last_t = 0;

	// ── Canvas sizing (verbatim pattern from MeshPanel) ──────────────────
	function sync_canvas(): { w: number; h: number } {
		if (!container || !canvas) return { w: 0, h: 0 };
		const r = container.getBoundingClientRect();
		const w = Math.round(r.width);
		const h = Math.round(r.height);
		if (w <= 0 || h <= 0) return { w: 0, h: 0 };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(w * dpr);
		const bh = Math.round(h * dpr);
		if (canvas.width !== bw || canvas.height !== bh) {
			canvas.width = bw;
			canvas.height = bh;
			canvas.style.width = w + 'px';
			canvas.style.height = h + 'px';
		}
		return { w: bw, h: bh };
	}
	function render_frame() {
		if (!gl_state) return;
		const { w, h } = sync_canvas();
		if (w <= 0 || h <= 0) return;
		render3D(gl_state, camera, w, h);
	}

	// ── Geometry upload: dim lit surface fill + bright wireframe ─────────
	function triNormal(d: MeshJson, a: number, b: number, c: number): [number, number, number] {
		const p = d.points;
		const u = [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
		const v = [p[c][0] - p[a][0], p[c][1] - p[a][1], p[c][2] - p[a][2]];
		const n: [number, number, number] = [
			u[1] * v[2] - u[2] * v[1],
			u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0]
		];
		const l = Math.hypot(n[0], n[1], n[2]) || 1;
		return [n[0] / l, n[1] / l, n[2] / l];
	}

	function build_geometry(state: GLState, d: MeshJson) {
		clearMeshes(state);
		const pts = d.points;

		// Surface triangle fill, grouped per adjacent region for subtle color.
		const byRegion = new Map<number, { pos: number[]; nrm: number[] }>();
		for (const f of d.faces) {
			const r = f.regions[0] >= 0 ? f.regions[0] : f.regions[1] >= 0 ? f.regions[1] : 0;
			let g = byRegion.get(r);
			if (!g) {
				g = { pos: [], nrm: [] };
				byRegion.set(r, g);
			}
			const n = triNormal(d, f.tri[0], f.tri[1], f.tri[2]);
			for (const v of f.tri) {
				g.pos.push(pts[v][0], pts[v][1], pts[v][2]);
				g.nrm.push(n[0], n[1], n[2]);
			}
		}
		for (const [r, g] of byRegion) {
			addMesh(state, new Float32Array(g.pos), new Float32Array(g.nrm), regionColor(r), TAG_FILL);
		}

		// Surface wireframe: every unique boundary-face edge once.
		const stride = pts.length;
		const seen = new Set<number>();
		const wpos: number[] = [];
		for (const f of d.faces) {
			for (let e = 0; e < 3; e++) {
				const a = f.tri[e];
				const b = f.tri[(e + 1) % 3];
				const key = a < b ? a * stride + b : b * stride + a;
				if (seen.has(key)) continue;
				seen.add(key);
				wpos.push(pts[a][0], pts[a][1], pts[a][2], pts[b][0], pts[b][1], pts[b][2]);
			}
		}
		addLineMesh(state, new Float32Array(wpos), wireColor, TAG_WIRE);
	}

	// Rebuild + refit whenever the model changes (after GL is ready).
	$effect(() => {
		const d = data;
		if (!gl_state || !d) return;
		const min: [number, number, number] = [Infinity, Infinity, Infinity];
		const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
		for (const p of d.points) {
			for (let k = 0; k < 3; k++) {
				if (p[k] < min[k]) min[k] = p[k];
				if (p[k] > max[k]) max[k] = p[k];
			}
		}
		setBBox(gl_state, min, max);
		build_geometry(gl_state, d);
		// Reset to the framed iso view; the idle orbit advances theta from here.
		Object.assign(camera, fitCamera(min, max));
		render_frame();
	});

	// ── Idle orbit loop ─────────────────────────────────────────────────
	function loop(t: number) {
		if (gl_state) {
			if (orbit && !dragging && last_t) camera.theta += (ORBIT_SPEED * (t - last_t)) / 1000;
			last_t = t;
			render_frame();
		}
		raf = requestAnimationFrame(loop);
	}

	// ── Pointer / wheel controls (drag orbit, right-drag pan, wheel zoom) ─
	function on_wheel(e: WheelEvent) {
		e.preventDefault();
		camera.distance *= e.deltaY > 0 ? 1.1 : 1 / 1.1;
		oninteract?.();
	}
	function on_pointer_down(e: PointerEvent) {
		dragging = true;
		right_drag = e.button === 2;
		last_mouse = { x: e.clientX, y: e.clientY };
		canvas?.setPointerCapture(e.pointerId);
		oninteract?.();
	}
	function on_pointer_move(e: PointerEvent) {
		if (!dragging) return;
		const dx = e.clientX - last_mouse.x;
		const dy = e.clientY - last_mouse.y;
		last_mouse = { x: e.clientX, y: e.clientY };
		if (right_drag) {
			const panScale = camera.distance * 0.0007;
			const ct = Math.cos(camera.theta);
			const st = Math.sin(camera.theta);
			camera.target = [
				camera.target[0] + (dx * ct - dy * st * Math.sin(camera.phi)) * panScale,
				camera.target[1] - (dx * st + dy * ct * Math.sin(camera.phi)) * panScale,
				camera.target[2] + dy * Math.cos(camera.phi) * panScale
			];
		} else {
			camera.theta += dx * 0.005;
			camera.phi = Math.max(
				-Math.PI / 2 + 0.01,
				Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.005)
			);
		}
	}
	function on_pointer_up() {
		dragging = false;
		right_drag = false;
	}
	function on_context_menu(e: Event) {
		e.preventDefault();
	}

	onMount(() => {
		if (!canvas) return;
		gl_state = initGL(canvas);
		raf = requestAnimationFrame(loop);
	});

	onDestroy(() => {
		if (raf) cancelAnimationFrame(raf);
		if (gl_state) disposeGL(gl_state);
		gl_state = null;
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
	></canvas>
</div>

<style>
	.viewer {
		position: absolute;
		inset: 0;
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
</style>
