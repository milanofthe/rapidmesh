<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { base } from '$app/paths';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import { adaptMesh } from '$lib/mesh_adapter';
	import { CYCLE_MS, FADE_MS, RESUME_MS } from '$lib/constants';
	import type { MeshJson } from '$lib/mesh_types';

	interface ModelStats {
		n_tets: number;
		n_points: number;
		min_dihedral_deg: number;
	}
	interface ModelEntry {
		id: string;
		name: string;
		file: string;
		stats?: ModelStats;
	}

	let models: ModelEntry[] = $state([]);
	let activeIndex = $state(0);
	// $state.raw: mesh payloads are large (10^5 faces); deep proxying would
	// stall the main thread for seconds on every model swap.
	let currentData: MeshJson | null = $state.raw(null);
	let covered = $state(true);
	let paused = $state(false);

	let busy = false;
	let cycleTimer: ReturnType<typeof setInterval> | null = null;
	let resumeTimer: ReturnType<typeof setTimeout> | null = null;

	const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

	async function fetchMesh(file: string): Promise<MeshJson> {
		const resp = await fetch(`${base}/${file}`);
		return (await resp.json()) as MeshJson;
	}

	// Fade to black, swap the model under cover, fade back in.
	async function transitionTo(i: number) {
		if (busy || models.length === 0) return;
		busy = true;
		covered = true;
		await delay(FADE_MS);
		activeIndex = i;
		try {
			currentData = await fetchMesh(models[i].file);
		} catch {
			currentData = null;
		}
		// Brief hold so the first frame builds while still covered.
		await delay(80);
		covered = false;
		busy = false;
	}

	function pauseCycle() {
		paused = true;
		if (resumeTimer) clearTimeout(resumeTimer);
		resumeTimer = setTimeout(() => (paused = false), RESUME_MS);
	}

	function onTab(i: number) {
		pauseCycle();
		if (i !== activeIndex) void transitionTo(i);
	}

	const activeStats = $derived(models[activeIndex]?.stats);

	onMount(() => {
		void (async () => {
			const resp = await fetch(`${base}/meshes/manifest.json`);
			const data = (await resp.json()) as { models: ModelEntry[] };
			models = data.models ?? [];
			if (models.length > 0) await transitionTo(0);
		})();

		cycleTimer = setInterval(() => {
			if (!paused && !busy && models.length > 1) {
				void transitionTo((activeIndex + 1) % models.length);
			}
		}, CYCLE_MS);
	});

	onDestroy(() => {
		if (cycleTimer) clearInterval(cycleTimer);
		if (resumeTimer) clearTimeout(resumeTimer);
	});
</script>

<main class="stage" style="--fade: {FADE_MS}ms">

	<!-- Any interaction pauses BOTH the auto-cycle and the idle orbit; the
	     paused flag resumes them together after the inactivity window. -->
	<MeshViewer mesh={currentData ? adaptMesh(currentData) : null} oninteract={pauseCycle} orbit={!paused} />

	<div class="overlay" class:visible={covered}></div>

	<header class="brand">
		<span class="wordmark">
			<img class="logo" src="{base}/favicon.svg" alt="" />
			<span class="name">rapidmesh</span>
		</span>
		<span class="tagline">pure-rust tet mesher</span>
		{#if activeStats}
			<span class="stats">
				{activeStats.n_tets.toLocaleString()} tets · {activeStats.n_points.toLocaleString()} points ·
				{activeStats.min_dihedral_deg.toFixed(1)}° min dihedral
			</span>
		{/if}
	</header>

	<nav class="tabs">
		{#each models as m, i (m.id)}
			<button class="tab" class:active={i === activeIndex} onclick={() => onTab(i)}>
				{m.name}
			</button>
		{/each}
	</nav>
</main>

<style>
	.stage {
		position: fixed;
		inset: 0;
		overflow: hidden;
		background: var(--bg);
	}

	/* Fade-through-black transition cover. */
	.overlay {
		position: absolute;
		inset: 0;
		background: var(--bg);
		opacity: 0;
		pointer-events: none;
		transition: opacity var(--fade);
		z-index: 5;
	}
	.overlay.visible {
		opacity: 1;
	}

	/* Top LEFT; the viewer's legend is pushed down below this block (see the
	   SHOWCASE CHANGE in MeshViewer) so the two never clip. */
	.brand {
		position: absolute;
		top: var(--space-xl);
		left: var(--space-xl);
		z-index: 10;
		display: flex;
		flex-direction: column;
		gap: var(--space-xs);
		pointer-events: none;
	}
	/* Mirror rapidfem's brand lockup: logo mark + monospace, letter-spaced
	   accent title (rapidfem renders its hero title in JetBrains Mono / 700 /
	   accent with wide tracking). The name stays "rapidmesh". */
	.wordmark {
		display: flex;
		align-items: center;
		gap: var(--space-md);
	}
	.logo {
		height: 22px;
		width: auto;
		display: block;
	}
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
	.stats {
		margin-top: var(--space-md);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}

	.tabs {
		position: absolute;
		bottom: 0;
		left: 0;
		right: 0;
		z-index: 10;
		display: flex;
		flex-wrap: wrap;
		gap: 1px;
		padding: var(--space-md);
		justify-content: center;
		background: linear-gradient(to top, var(--bg), transparent);
	}
	.tab {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		padding: 8px 16px;
		cursor: pointer;
		transition:
			background var(--transition),
			border-color var(--transition),
			color var(--transition);
	}
	.tab:hover {
		background: var(--bg-panel);
		border-color: var(--accent);
		color: var(--text);
	}
	.tab.active {
		color: var(--accent);
		background: var(--accent-dim);
		border-color: var(--accent);
	}
</style>
