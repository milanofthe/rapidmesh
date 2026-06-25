// Bundle entry: re-export the shared (browser+node) render pipeline for Node.
export { adaptMesh } from '$lib/mesh_adapter';
export { buildScene } from '$lib/render/scene_build';
export * as gpu3d from '$lib/render/canvas3d_webgpu';
