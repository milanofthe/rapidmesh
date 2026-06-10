/** Shared camera, camera animation and render coalescing across panels.
 *
 * All panels orbit the same camera; any interaction schedules one render of
 * every registered panel on the next animation frame (the same coalescing
 * rationale as MeshViewer's schedule_render). The ease-out cubic camera
 * animation is MeshViewer's animate_camera, operating on the shared camera.
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

// ── Camera animation (ease-out cubic, from MeshViewer) ────────────────
let anim_id = 0;
let anim_target: Camera | null = null;

/** The camera an interaction should base itself on: the in-flight animation
 *  target if one exists (so chained zooms compound), else the live camera. */
export function effective_camera(): Camera {
	return anim_target ?? camera;
}

export function animate_camera(target: Camera, durationMs = 300) {
	anim_target = target;
	const start = { ...camera, target: [...camera.target] as [number, number, number] };
	const t0 = performance.now();
	const id = ++anim_id;
	function tick() {
		if (id !== anim_id) return;
		const t = Math.min(1, (performance.now() - t0) / durationMs);
		const e = 1 - Math.pow(1 - t, 3);
		camera.theta = start.theta + (target.theta - start.theta) * e;
		camera.phi = start.phi + (target.phi - start.phi) * e;
		camera.distance = start.distance + (target.distance - start.distance) * e;
		camera.target = [
			start.target[0] + (target.target[0] - start.target[0]) * e,
			start.target[1] + (target.target[1] - start.target[1]) * e,
			start.target[2] + (target.target[2] - start.target[2]) * e
		];
		render_all();
		if (t < 1) requestAnimationFrame(tick);
		else anim_target = null;
	}
	requestAnimationFrame(tick);
}

/** Cancel an in-flight animation (call when the user grabs the camera). */
export function cancel_camera_animation() {
	anim_id++;
	anim_target = null;
}
