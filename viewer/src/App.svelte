<script lang="ts">
	import { onMount } from 'svelte';
	import MeshPanel from '$lib/components/MeshPanel.svelte';
	import type { MeshJson, ViewSettings } from '$lib/mesh_types';
	import Select from '$lib/components/Select.svelte';
	import { initTooltips } from '$lib/tooltip';
	import { camera, render_all } from '$lib/viewbus';
	import { fitCamera } from '$lib/render/canvas3d';

	const MESHERS = ['rapidmesh', 'gmsh', 'tetgen'];

	let manifest: string[] = $state([]);
	let geometry: string = $state('');
	let meshes: MeshJson[] = $state([]);
	let bbox = $state({
		min: [0, 0, 0] as [number, number, number],
		max: [1, 1, 1] as [number, number, number]
	});

	let settings: ViewSettings = $state({
		surface: true,
		surface_wire: true,
		tet_fill: false,
		tet_wire: false,
		clip_enable: false,
		clip_axis: 2,
		clip_t: 0.5
	});

	const layer_toggles: { key: 'surface' | 'surface_wire' | 'tet_fill' | 'tet_wire'; label: string; tip: string }[] = [
		{ key: 'surface', label: 'srf', tip: 'Surface faces' },
		{ key: 'surface_wire', label: 'wire', tip: 'Surface wireframe' },
		{ key: 'tet_fill', label: 'tet', tip: 'Tet fill (use with clip)' },
		{ key: 'tet_wire', label: 'edges', tip: 'All tet edges' }
	];

	async function load_geometry(name: string) {
		const loaded: MeshJson[] = [];
		for (const mesher of MESHERS) {
			try {
				const resp = await fetch(`meshes/${mesher}_${name}.json`);
				if (!resp.ok) continue;
				loaded.push((await resp.json()) as MeshJson);
			} catch {
				// Mesher output not available for this geometry: skip.
			}
		}
		if (loaded.length > 0) {
			const min: [number, number, number] = [Infinity, Infinity, Infinity];
			const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
			for (const p of loaded[0].points) {
				for (let k = 0; k < 3; k++) {
					min[k] = Math.min(min[k], p[k]);
					max[k] = Math.max(max[k], p[k]);
				}
			}
			bbox = { min, max };
			Object.assign(camera, fitCamera(min, max));
		}
		meshes = loaded;
		render_all();
	}

	$effect(() => {
		if (geometry) void load_geometry(geometry);
	});

	onMount(() => {
		initTooltips();
		void (async () => {
			manifest = (await (await fetch('meshes/manifest.json')).json()) as string[];
			if (manifest.length > 0) geometry = manifest[0];
		})();
	});
</script>

<div class="shell">
	<header>
		<span class="brand">rapidmesh</span>
		<span class="brand-sub">mesh comparison</span>
		<div class="geometry-select">
			<Select bind:value={geometry} options={manifest.map((m) => ({ value: m, label: m }))} />
		</div>
		<span class="tb-sep" aria-hidden="true"></span>
		{#each layer_toggles as t}
			<button
				class="tb tb-label has-tip"
				class:active={settings[t.key]}
				onclick={() => (settings = { ...settings, [t.key]: !settings[t.key] })}
			>
				<span class="tip">{t.tip}</span>
				{t.label}
			</button>
		{/each}
		<span class="tb-sep" aria-hidden="true"></span>
		<button
			class="tb tb-label has-tip"
			class:active={settings.clip_enable}
			onclick={() => (settings = { ...settings, clip_enable: !settings.clip_enable })}
		>
			<span class="tip">Clip plane (inspect interior)</span>
			clip
		</button>
		{#each [0, 1, 2] as axis}
			<button
				class="tb tb-label"
				class:active={settings.clip_axis === axis}
				disabled={!settings.clip_enable}
				onclick={() => (settings = { ...settings, clip_axis: axis as 0 | 1 | 2 })}
			>
				{['x', 'y', 'z'][axis]}
			</button>
		{/each}
		<input
			class="clip-slider"
			type="range"
			min="0"
			max="1"
			step="0.005"
			disabled={!settings.clip_enable}
			value={settings.clip_t}
			oninput={(e) => (settings = { ...settings, clip_t: parseFloat(e.currentTarget.value) })}
		/>
		<span class="hint">drag orbit · right-drag pan · wheel zoom · double-click fit</span>
	</header>

	<div class="panels">
		{#each meshes as mesh (mesh.mesher + mesh.name)}
			<div class="panel-card">
				<MeshPanel data={mesh} {settings} {bbox} />
			</div>
		{:else}
			<div class="empty">
				No mesh data. Run <code>cargo run --release --bin export_meshes</code> first.
			</div>
		{/each}
	</div>
</div>

<style>
	.shell {
		display: flex;
		flex-direction: column;
		height: 100vh;
	}
	header {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 8px 12px;
		background: var(--bg-mid);
		border-bottom: 1px solid var(--border);
		flex-wrap: wrap;
	}
	.brand {
		font-size: var(--fs-md);
		font-weight: 700;
		color: var(--accent);
		letter-spacing: 0.3px;
	}
	.brand-sub {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		text-transform: uppercase;
		letter-spacing: 1.5px;
		margin-right: 8px;
	}
	.geometry-select {
		width: 160px;
	}
	.tb-sep {
		display: inline-block;
		width: 1px;
		height: 20px;
		margin: 4px 4px;
		background: var(--border);
	}
	.tb {
		position: relative;
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
	.tb.tb-label {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		padding: 0 10px;
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
	/* Range input restyled square-on-square to match the rapidfem input
	   look (no native rounded track/thumb). */
	.clip-slider {
		-webkit-appearance: none;
		appearance: none;
		width: 140px;
		height: 28px;
		background: transparent;
		padding: 0;
		margin: 0;
	}
	.clip-slider::-webkit-slider-runnable-track {
		height: 4px;
		background: var(--input-bg);
		border: 1px solid var(--input-border);
	}
	.clip-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 10px;
		height: 16px;
		margin-top: -7px;
		background: var(--accent);
		border: none;
		border-radius: 0;
		cursor: ew-resize;
		transition: background var(--transition);
	}
	.clip-slider:hover::-webkit-slider-thumb {
		background: var(--accent-hover);
	}
	.clip-slider:disabled::-webkit-slider-thumb {
		background: var(--border);
		cursor: default;
	}
	.clip-slider::-moz-range-track {
		height: 4px;
		background: var(--input-bg);
		border: 1px solid var(--input-border);
	}
	.clip-slider::-moz-range-thumb {
		width: 10px;
		height: 16px;
		background: var(--accent);
		border: none;
		border-radius: 0;
		cursor: ew-resize;
	}
	.clip-slider:disabled::-moz-range-thumb {
		background: var(--border);
	}
	.hint {
		margin-left: auto;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
	}
	.panels {
		flex: 1;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
		gap: 10px;
		padding: 10px;
		min-height: 0;
	}
	.panel-card {
		border: 1px solid var(--border-subtle);
		min-height: 0;
		display: flex;
	}
	.panel-card :global(.viewer) {
		flex: 1;
	}
	.empty {
		color: var(--text-dim);
		font-family: var(--font-mono);
		padding: 24px;
	}
	.empty code {
		color: var(--text-muted);
	}
</style>
