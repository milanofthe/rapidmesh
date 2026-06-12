/* SHOWCASE STUB for rapidfem's $lib/api.
 *
 * The MeshViewer component is a verbatim copy from rapidfem (source of
 * truth: rapidfem/python/python_src/rapidfem/ui/frontend-src/src/lib/
 * components/MeshViewer.svelte). Its field / time-domain visualization
 * paths call into rapidfem's FEM runtime through these functions; the
 * static showcase never enables those paths (show_field stays false,
 * td_trajectory stays null), so the stubs are never reached at runtime.
 * They exist only to satisfy the imports without touching the component.
 */

export type TdTrajectoryPayload = {
	times: number[] | Float64Array;
	[key: string]: unknown;
};

export async function viz_load_mesh(..._args: unknown[]): Promise<void> {}

export async function viz_sample(..._args: unknown[]): Promise<null> {
	return null;
}

export async function viz_sample_static(..._args: unknown[]): Promise<null> {
	return null;
}

export async function viz_eval_static(..._args: unknown[]): Promise<null> {
	return null;
}
