<script lang="ts" generics="T extends string | number">
	import { onMount, onDestroy } from 'svelte';

	let {
		value = $bindable<T>(),
		options,
		open_up = false,
	}: {
		value: T;
		options: Array<{ value: T; label: string; description?: string }>;
		open_up?: boolean;
	} = $props();

	let open = $state(false);
	let host: HTMLDivElement | undefined = $state();

	const selected = $derived(options.find((o) => o.value === value));

	function on_outside(e: MouseEvent) {
		if (!open || !host) return;
		if (!host.contains(e.target as Node)) open = false;
	}

	onMount(() => document.addEventListener('mousedown', on_outside, true));
	onDestroy(() => document.removeEventListener('mousedown', on_outside, true));
</script>

<div class="dropdown" bind:this={host}>
	<button class="btn" onclick={() => (open = !open)} type="button">
		<span class="label">{selected?.label ?? String(value)}</span>
		<svg width="8" height="5" viewBox="0 0 8 5" fill="currentColor"><path d="M0 0L4 5L8 0Z"/></svg>
	</button>
	{#if open}
		<div class="menu" class:up={open_up}>
			{#each options as o}
				<button
					class="option"
					class:active={value === o.value}
					type="button"
					onclick={() => { value = o.value; open = false; }}
				>
					<span>{o.label}</span>
					{#if o.description}<span class="desc">{o.description}</span>{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.dropdown {
		position: relative;
		display: inline-block;
	}
	.btn {
		min-width: 80px;
		height: 22px;
		padding: 0 var(--space-md);
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		background: var(--input-bg);
		border: 1px solid var(--input-border);
		color: var(--text-muted);
		cursor: pointer;
		text-align: left;
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: var(--space-md);
		text-transform: none;
		letter-spacing: 0;
		font-weight: 500;
		transition: border-color var(--transition);
	}
	.btn:hover { border-color: var(--accent); }
	.btn .label { color: var(--text); }
	.menu {
		position: absolute;
		top: calc(100% + 2px);
		left: 0;
		min-width: 100%;
		z-index: 30;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		max-height: 240px;
		overflow: auto;
	}
	.menu.up {
		top: auto;
		bottom: calc(100% + 2px);
	}
	.option {
		padding: 5px var(--space-md);
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-muted);
		background: none;
		border: 0;
		text-align: left;
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: 400;
		display: flex;
		flex-direction: column;
		gap: 1px;
		transition: background var(--transition);
		white-space: nowrap;
	}
	.option:hover { background: var(--accent-dim); }
	.option.active { color: var(--accent); font-weight: 600; }
	.desc { font-size: 9px; color: var(--text-dim); }
</style>
