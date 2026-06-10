/** Shared camera and render coalescing across comparison panels.
 *
 * All panels orbit the same camera; any interaction schedules one render of
 * every registered panel on the next animation frame (the same coalescing
 * rationale as MeshViewer's schedule_render).
 */

import type { Camera } from '$lib/render/canvas3d';

export const camera: Camera = {
	theta: Math.PI / 4,
	phi: Math.PI / 4,
	distance: 10,
	target: [0, 0, 0]
};

const renderers = new Set<() => void>();
let scheduled = false;

export function register_renderer(fn: () => void): () => void {
	renderers.add(fn);
	return () => renderers.delete(fn);
}

export function render_all(): void {
	if (scheduled) return;
	scheduled = true;
	requestAnimationFrame(() => {
		scheduled = false;
		for (const fn of renderers) fn();
	});
}
