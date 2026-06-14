<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import { adaptMesh } from '$lib/mesh_adapter';
	import { createCamera, fitCamera, type Camera } from '$lib/render/canvas3d';
	import type { MeshJson } from '$lib/mesh_types';
	import type { MeshData } from '$lib/msh';

	// rapidmesh vs gmsh vs tetgen: three of the EXACT same MeshViewer (the
	// verbatim rapidfem renderer), meshing the same geometry at the same target
	// size. One shared camera and one shared set of layer/clip toggles drive all
	// three. tetgen has no CAD kernel and meshes gmsh's surface.

	interface MesherStats {
		n_tets: number;
		n_points: number;
		n_regions?: number;
		min_dihedral_deg: number;
		max_radius_edge: number;
		max_edge: number;
		millis: number;
		on_surface_of?: string;
	}
	interface MesherEntry { file: string; stats: MesherStats; }
	interface GeomEntry {
		id: string;
		name: string;
		category: string;
		target_h: number;
		meshers: Record<string, MesherEntry>;
	}

	const MESHERS = ['rapidmesh', 'gmsh', 'tetgen'] as const;

	let geoms: GeomEntry[] = $state([]);
	let activeId = $state('');
	let meshes: Record<string, MeshData | null> = $state.raw({});

	// SHARED across all three panes via bind: — see MeshViewer COMPARE SEAM.
	let camera = $state.raw<Camera>(createCamera());
	let layer_surface = $state(true);
	let layer_wire = $state(true);
	let layer_tets = $state(true);
	let clip_enable = $state(true);
	let clip_axis = $state<0 | 1 | 2>(1);
	let clip_t = $state(0.6);

	const CLIP_AXES: { lbl: string; ax: 0 | 1 | 2 }[] = [
		{ lbl: 'X', ax: 0 }, { lbl: 'Y', ax: 1 }, { lbl: 'Z', ax: 2 },
	];

	let categories = $derived.by(() => {
		const seen: string[] = [];
		for (const g of geoms) if (!seen.includes(g.category)) seen.push(g.category);
		return seen.map((c) => ({ name: c, items: geoms.filter((g) => g.category === c) }));
	});
	let active = $derived(geoms.find((g) => g.id === activeId) ?? null);

	async function fetchMesh(file: string): Promise<MeshData | null> {
		try {
			const resp = await fetch(`${base}/${file}`);
			return adaptMesh((await resp.json()) as MeshJson);
		} catch {
			return null;
		}
	}

	async function select(id: string) {
		const g = geoms.find((x) => x.id === id);
		if (!g) return;
		activeId = id;
		const entries = await Promise.all(
			MESHERS.map(async (k) => {
				const e = g.meshers[k];
				return [k, e ? await fetchMesh(e.file) : null] as const;
			})
		);
		const next: Record<string, MeshData | null> = {};
		for (const [k, v] of entries) next[k] = v;
		meshes = next;
	}

	function fitAll() {
		const ref = meshes.rapidmesh ?? meshes.gmsh ?? meshes.tetgen;
		if (ref) camera = fitCamera(ref.bbox.min, ref.bbox.max);
	}

	function fmt(n: number | undefined, d = 0): string {
		if (n == null) return '—';
		return n.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
	}

	onMount(() => {
		void (async () => {
			const resp = await fetch(`${base}/meshes/compare/manifest.json`);
			const data = (await resp.json()) as { geometries: GeomEntry[] };
			geoms = data.geometries ?? [];
			if (geoms.length > 0) await select(geoms[0].id);
		})();
	});
</script>

<main class="stage">
	<header class="topbar">
		<span class="wordmark">
			<img class="logo" src="{base}/favicon.svg" alt="" />
			<span class="name">rapidmesh</span>
			<span class="tagline">pure-rust tet mesher</span>
		</span>

		<!-- shared mesh-layer toolbar (the rapidfem MeshViewer toolbar, lifted
		     out so it drives all three panes at once) -->
		<div class="toolbar">
		<button class="tb" onclick={fitAll} title="Fit view">Fit</button>
		<span class="tb-sep"></span>
		<button class="tb" class:active={layer_surface} onclick={() => (layer_surface = !layer_surface)}>Tets</button>
		<button class="tb" class:active={layer_wire} onclick={() => (layer_wire = !layer_wire)}>Wire</button>
		<button class="tb" class:active={layer_tets} onclick={() => (layer_tets = !layer_tets)}>Edge</button>
		<span class="tb-sep"></span>
		<button class="tb" class:active={clip_enable} onclick={() => (clip_enable = !clip_enable)}>Clip</button>
		{#each CLIP_AXES as a (a.ax)}
			<button class="tb" class:active={clip_enable && clip_axis === a.ax}
				disabled={!clip_enable} onclick={() => (clip_axis = a.ax)}>{a.lbl}</button>
		{/each}
		<input class="clip-slider" type="range" min="0" max="1" step="0.001"
			bind:value={clip_t} disabled={!clip_enable} />
	</div>

	</header>

	<section class="panes">
		{#each MESHERS as key (key)}
			{@const entry = active?.meshers[key]}
			<article class="pane">
				<div class="pane-head">
					<span class="pane-name">{key}</span>
					{#if entry?.stats.on_surface_of}
						<span class="note">meshes {entry.stats.on_surface_of}'s surface</span>
					{/if}
				</div>
				<div class="pane-canvas">
					<MeshViewer
						mesh={meshes[key] ?? null}
						controls={false}
						orbit={false}
						bind:camera
						{layer_surface}
						{layer_wire}
						{layer_tets}
						{clip_enable}
						{clip_axis}
						{clip_t}
					/>
				</div>
				<table class="stats">
					<tbody>
						<tr><th>tets</th><td>{fmt(entry?.stats.n_tets)}</td></tr>
						<tr><th>points</th><td>{fmt(entry?.stats.n_points)}</td></tr>
						{#if active?.category === 'Multi-Region'}
							<tr><th>regions</th><td>{fmt(entry?.stats.n_regions)}</td></tr>
						{/if}
						<tr><th>min&#8736;</th><td>{fmt(entry?.stats.min_dihedral_deg, 1)}&#176;</td></tr>
						<tr><th>radius/edge</th><td>{fmt(entry?.stats.max_radius_edge, 1)}</td></tr>
						<tr><th>time</th><td>{fmt(entry?.stats.millis)} ms</td></tr>
					</tbody>
				</table>
			</article>
		{/each}
	</section>

	<nav class="tabs">
		{#each categories as cat (cat.name)}
			<span class="cat">{cat.name}</span>
			{#each cat.items as g (g.id)}
				<button class="tab" class:active={g.id === activeId} onclick={() => select(g.id)}>{g.name}</button>
			{/each}
		{/each}
	</nav>
</main>

<style>
	.stage {
		position: fixed;
		inset: 0;
		display: flex;
		flex-direction: column;
		background: var(--bg);
		overflow: hidden;
	}

	/* top bar in normal flow: brand left, shared toolbar right, both wrap */
	.topbar {
		flex: none;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-md) var(--space-lg);
		padding: var(--space-md) var(--space-lg);
		border-bottom: 1px solid var(--border);
	}
	.wordmark { display: flex; align-items: baseline; gap: var(--space-md); }
	.logo { height: 20px; width: auto; align-self: center; }
	.name {
		font-family: var(--font-mono);
		font-size: var(--fs-lg);
		font-weight: 700;
		color: var(--accent);
		letter-spacing: 2px;
	}
	.tagline {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 1.5px;
		color: var(--text-dim);
	}

	/* shared toolbar — verbatim rapidfem MeshViewer .tb button look */
	.toolbar {
		margin-left: auto;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: flex-end;
		gap: 2px;
	}
	.tb {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 28px;
		height: 28px;
		padding: 0 9px;
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		cursor: pointer;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tb:hover:not(:disabled) { background: var(--bg-panel); color: var(--text); }
	.tb.active { color: var(--accent); border-color: var(--accent); background: var(--accent-dim); }
	.tb:disabled { opacity: 0.4; cursor: default; }
	.tb-sep { width: 1px; height: 20px; margin: 0 4px; background: var(--border); }

	/* verbatim rapidfem MeshViewer clip slider (flat track + squared thumb) */
	.clip-slider {
		-webkit-appearance: none;
		appearance: none;
		width: 110px;
		height: 18px;
		margin-left: 4px;
		background: transparent;
		cursor: pointer;
		padding: 0;
	}
	.clip-slider:focus { outline: none; }
	.clip-slider::-webkit-slider-runnable-track { height: 2px; background: var(--border); border: 0; }
	.clip-slider::-moz-range-track { height: 2px; background: var(--border); border: 0; }
	.clip-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 10px;
		height: 14px;
		margin-top: -6px;
		background: var(--accent);
		border: 0;
		border-radius: 0;
		cursor: grab;
	}
	.clip-slider::-moz-range-thumb {
		width: 10px;
		height: 14px;
		background: var(--accent);
		border: 0;
		border-radius: 0;
		cursor: grab;
	}
	.clip-slider:hover::-webkit-slider-thumb { background: var(--accent-hover); }
	.clip-slider:hover::-moz-range-thumb { background: var(--accent-hover); }
	.clip-slider:active::-webkit-slider-thumb { cursor: grabbing; }
	.clip-slider:active::-moz-range-thumb { cursor: grabbing; }
	.clip-slider:disabled { opacity: 0.4; cursor: default; }

	.panes {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 1px;
		flex: 1 1 auto;
		min-height: 0;
		background: var(--border);
		margin-bottom: var(--space-lg); /* breathing room above the geometry tabs */
	}
	.pane {
		display: flex;
		flex-direction: column;
		min-height: 0;
		background: var(--bg);
	}
	.pane-head {
		display: flex;
		align-items: baseline;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-md);
		border-top: 1px solid var(--border);
		border-bottom: 1px solid var(--border);
		flex: none;
	}
	.pane-name {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 700;
		color: var(--accent);
		letter-spacing: 1px;
		text-transform: uppercase;
	}
	.note {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
	}
	.pane-canvas {
		flex: 1 1 auto;
		min-height: 0;
		position: relative;
		overflow: hidden;
	}

	/* stats as a real table, rapidfem panel aesthetic */
	.stats {
		flex: none;
		width: 100%;
		border-collapse: separate;   /* collapse hides the table's own edges */
		border-spacing: 0;
		/* separators above AND below the stats block */
		border-top: 1px solid var(--border);
		border-bottom: 1px solid var(--border);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.stats th, .stats td {
		padding: 4px var(--space-md);
		text-align: left;
		border-bottom: 1px solid var(--border-subtle);
	}
	.stats th {
		color: var(--text-dim);
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		width: 45%;
	}
	.stats td { color: var(--text); text-align: right; }
	.stats tr:last-child th, .stats tr:last-child td { border-bottom: 0; }

	.tabs {
		position: relative;
		z-index: 20;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1px;
		padding: var(--space-md) var(--space-md) var(--space-lg);
		justify-content: center;
	}
	.cat {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
		margin: 0 var(--space-sm) 0 var(--space-md);
	}
	.cat:first-child { margin-left: 0; }
	.tab {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		padding: 6px 14px;
		cursor: pointer;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tab:hover { background: var(--bg-panel); border-color: var(--accent); color: var(--text); }
	.tab.active { color: var(--accent); background: var(--accent-dim); border-color: var(--accent); }

	/* Narrow / mobile: three panes can't sit side by side usefully, so the
	   stage scrolls and each pane stacks at a usable height. Touch-orbit works
	   inside each canvas (touch-action: none); page scroll happens on the
	   topbar / gaps / tabs. */
	@media (max-width: 820px) {
		.stage {
			position: static;
			min-height: 100vh;
			overflow-y: auto;
		}
		.panes {
			grid-template-columns: 1fr;
			margin-bottom: 0;
		}
		.pane { min-height: 78vh; }
		.toolbar { width: 100%; margin-left: 0; justify-content: flex-start; }
	}
</style>
