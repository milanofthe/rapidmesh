<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { WORDMARK } from '$lib/wordmark';

	// The landing page IS a live rapidmesh run: the project's own 2D Ruppert
	// mesher (compiled to WASM) triangulates the whole viewport around the
	// "RapidMESH" wordmark -- whose glyph outline is baked in (src/lib/wordmark.ts)
	// as polygon holes plus a crisp SVG path -- and animates itself building,
	// coarse to fine.

	const BG = '#131316'; // theme background
	const MESH = 'rgba(154, 150, 160, 0.22)'; // our mesh colour: a quiet structural line
	const ACCENT = '#d9513c'; // lava accent for the wordmark

	let canvas: HTMLCanvasElement;

	onMount(() => {
		let disposed = false;
		let timer = 0;
		let resizeTimer = 0;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		let wasm: any;
		let ctx: CanvasRenderingContext2D;
		const glyph = new Path2D(WORDMARK.path);

		// Place the wordmark in the W x H viewport: a similarity transform (scale s,
		// translate tx,ty) shared by the holes and the SVG fill so they coincide.
		function placement(W: number, H: number) {
			const bb = WORDMARK.bbox;
			const s = (0.5 * W) / (bb.x2 - bb.x1);
			const tx = W / 2 - (s * (bb.x1 + bb.x2)) / 2;
			const ty = H / 2 - (s * (bb.y1 + bb.y2)) / 2;
			return { s, tx, ty };
		}

		// The baked glyph loops, transformed into viewport pixels: the holes.
		function holes(s: number, tx: number, ty: number) {
			const flat: number[] = [];
			const lens: number[] = [];
			for (const lp of WORDMARK.loops) {
				lens.push(lp.length / 2);
				for (let i = 0; i < lp.length; i += 2) {
					flat.push(s * lp[i] + tx, s * lp[i + 1] + ty);
				}
			}
			return { outline: new Float64Array(flat), loopLens: new Uint32Array(lens) };
		}

		function drawFrame(
			pts: Float32Array,
			idx: Uint32Array,
			lw: number,
			s: number,
			tx: number,
			ty: number
		) {
			ctx.setTransform(1, 0, 0, 1, 0, 0);
			ctx.fillStyle = BG;
			ctx.fillRect(0, 0, canvas.width, canvas.height);
			ctx.strokeStyle = MESH;
			ctx.lineWidth = lw;
			ctx.beginPath();
			for (let t = 0; t < idx.length; t += 3) {
				const a = idx[t];
				const b = idx[t + 1];
				const c = idx[t + 2];
				ctx.moveTo(pts[2 * a], pts[2 * a + 1]);
				ctx.lineTo(pts[2 * b], pts[2 * b + 1]);
				ctx.lineTo(pts[2 * c], pts[2 * c + 1]);
				ctx.closePath();
			}
			ctx.stroke();
			// The crisp wordmark, same transform as the holes.
			ctx.setTransform(s, 0, 0, s, tx, ty);
			ctx.fillStyle = ACCENT;
			ctx.fill(glyph);
			ctx.setTransform(1, 0, 0, 1, 0, 0);
		}

		function run(animate: boolean) {
			clearTimeout(timer);
			const dpr = Math.min(window.devicePixelRatio || 1, 2);
			const W = Math.round(window.innerWidth * dpr);
			const H = Math.round(window.innerHeight * dpr);
			canvas.width = W;
			canvas.height = H;
			canvas.style.width = window.innerWidth + 'px';
			canvas.style.height = window.innerHeight + 'px';

			const { s, tx, ty } = placement(W, H);
			const { outline, loopLens } = holes(s, tx, ty);
			// Edge length grades from fine at the wordmark to coarse out in the
			// viewport, over ~half the short side -- adaptive to the viewport size.
			const u = Math.min(W, H);
			const hNear = u / 75; // fine at the wordmark
			const hFar = u / 11; // coarse out in the viewport
			const grade = 0.13; // gentle linear growth -> a smooth gradient
			const lw = Math.max(1.0 * dpr, 1.0);

			if (animate) {
				const steps = wasm.triangulate_steps(W, H, outline, loopLens, hNear, hFar, grade, 20);
				const n = steps.n_steps as number;
				let i = 0;
				const tick = () => {
					if (disposed) return;
					drawFrame(steps.points(i), steps.indices(i), lw, s, tx, ty);
					i++;
					if (i < n) timer = window.setTimeout(tick, 32);
				};
				tick();
			} else {
				const m = wasm.triangulate(W, H, outline, loopLens, hNear, hFar, grade, 20);
				drawFrame(m.points, m.indices, lw, s, tx, ty);
			}
		}

		async function boot() {
			// A runtime URL (not a static literal) so the bundler leaves the WASM
			// alone; it ships in /static.
			const wasmUrl = `${base}/wasm/rapidmesh_wasm.js`;
			const mod = await import(/* @vite-ignore */ wasmUrl);
			await mod.default();
			wasm = mod;
			ctx = canvas.getContext('2d')!;
			if (disposed) return;
			run(true);
			window.addEventListener('resize', onResize);
		}

		function onResize() {
			clearTimeout(resizeTimer);
			resizeTimer = window.setTimeout(() => !disposed && run(false), 160);
		}

		boot();
		return () => {
			disposed = true;
			clearTimeout(timer);
			clearTimeout(resizeTimer);
			window.removeEventListener('resize', onResize);
		};
	});
</script>

<svelte:head>
	<title>rapidmesh</title>
</svelte:head>

<canvas bind:this={canvas} aria-label="RapidMESH"></canvas>

<style>
	:global(html),
	:global(body) {
		margin: 0;
		height: 100%;
		background: #131316;
		overflow: hidden;
	}
	canvas {
		display: block;
		position: fixed;
		inset: 0;
	}
</style>
