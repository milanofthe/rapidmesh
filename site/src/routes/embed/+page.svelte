<script lang="ts">
	// REPORT EMBED ROUTE — mounts the verbatim showcase MeshViewer with all
	// state driven by the URL query string, so report/viewer.py can drive it
	// from Python (interactive window or headless screenshot). The ONLY viewer
	// change it relies on is the `transparentBackground` prop; everything else
	// (scene, materials, lighting, region colouring, camera) is the unchanged
	// component.
	import { onMount, tick } from 'svelte';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import { adaptMesh } from '$lib/mesh_adapter';
	import { fitCamera, type Camera } from '$lib/render/canvas3d';
	import type { MeshJson } from '$lib/mesh_types';
	import type { MeshData } from '$lib/msh';

	let mesh = $state.raw<MeshData | null>(null);
	let camera = $state.raw<Camera>({ theta: Math.PI / 4, phi: Math.PI / 4, distance: 1, target: [0, 0, 0] });
	let errored = $state('');

	// Query-driven display state (defaults reproduce the showcase look).
	let transparent = $state(false);
	let controls = $state(true);
	let layer_surface = $state(true);
	let layer_wire = $state(true);
	let layer_tets = $state(true);
	let clip_enable = $state(false);
	let clip_axis = $state<0 | 1 | 2>(1);
	let clip_t = $state(0.6);
	let width = $state(0);
	let height = $state(0);

	function raf2(): Promise<void> {
		return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
	}
	function markReady() {
		(window as unknown as { __viewerReady?: boolean }).__viewerReady = true;
		document.body.setAttribute('data-viewer-ready', errored ? 'error' : '1');
		if (errored) document.body.setAttribute('data-viewer-error', errored);
	}

	onMount(() => {
		const p = new URLSearchParams(window.location.search);
		const num = (k: string, d: number) => {
			const v = p.get(k);
			return v == null || v === '' ? d : parseFloat(v);
		};
		const bool = (k: string, d: boolean) => {
			const v = p.get(k);
			return v == null ? d : v === '1' || v === 'true';
		};

		transparent = bool('transparent', false);
		controls = bool('controls', true);
		layer_surface = bool('tets', true);
		layer_wire = bool('wire', true);
		layer_tets = bool('edges', true);
		clip_enable = bool('clip', false);
		clip_axis = Math.max(0, Math.min(2, Math.round(num('clipaxis', 1)))) as 0 | 1 | 2;
		clip_t = num('clipt', 0.6);
		width = num('width', 0);
		height = num('height', 0);
		const azim = num('azim', 30);
		const elev = num('elev', 20);
		const distRaw = p.get('dist');

		if (transparent) {
			document.documentElement.style.background = 'transparent';
			document.body.style.background = 'transparent';
		}

		void (async () => {
			const url = p.get('mesh');
			if (!url) {
				errored = 'missing mesh param';
				markReady();
				return;
			}
			try {
				const resp = await fetch(url);
				if (!resp.ok) throw new Error('fetch ' + resp.status);
				mesh = adaptMesh((await resp.json()) as MeshJson);
			} catch (e) {
				errored = String(e);
				markReady();
				return;
			}
			// Let the viewer mount and run its mesh-change auto-fit $effect, then
			// override the viewing angle (and distance, if asked) while keeping
			// the fitted framing target.
			await tick();
			await raf2();
			const fit = fitCamera(mesh.bbox.min, mesh.bbox.max);
			camera = {
				theta: (azim * Math.PI) / 180,
				phi: (elev * Math.PI) / 180,
				distance: distRaw != null && distRaw !== '' ? parseFloat(distRaw) : fit.distance,
				target: fit.target
			};
			await tick();
			await raf2();
			await raf2();
			markReady();
		})();
	});
</script>

<div
	class="embed"
	style:width={width > 0 ? width + 'px' : '100vw'}
	style:height={height > 0 ? height + 'px' : '100vh'}
	style:background={transparent ? 'transparent' : null}
>
	<MeshViewer
		{mesh}
		{controls}
		orbit={false}
		bind:camera
		{layer_surface}
		{layer_wire}
		{layer_tets}
		{clip_enable}
		{clip_axis}
		{clip_t}
		transparentBackground={transparent}
	/>
</div>

<style>
	:global(html),
	:global(body) {
		margin: 0;
		padding: 0;
	}
	.embed {
		position: relative;
		overflow: hidden;
	}
</style>
