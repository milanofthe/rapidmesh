/**
 * WebGPU port of `canvas3d.ts` with the SAME public API (createCamera,
 * cameraEye, fitCamera, addMesh, addLineMesh, setClipPlane, setBBox, render3D),
 * so the browser MeshViewer and the headless Node rasterizer drive ONE renderer.
 * Per-mesh uniform colour (1:1 with the WebGL2 path, not per-vertex). The
 * point-sprite field pass is rapidpassives-only and intentionally omitted here.
 *
 * Target-agnostic: `render3D` is handed the colour/depth texture views for the
 * frame, so the browser (canvas context texture) and Node (offscreen MSAA +
 * resolve) share the same code.
 */
import { MESH_WGSL, LINE_WGSL } from './webgpu_shaders';

export interface Camera { theta: number; phi: number; distance: number; target: [number, number, number]; }
export function createCamera(): Camera { return { theta: Math.PI / 4, phi: Math.PI / 4, distance: 3, target: [0, 0, 0] }; }

export function cameraEye(cam: Camera): [number, number, number] {
  const cp = Math.cos(cam.phi);
  return [
    cam.target[0] + cam.distance * cp * Math.sin(cam.theta),
    cam.target[1] + cam.distance * cp * Math.cos(cam.theta),
    cam.target[2] + cam.distance * Math.sin(cam.phi),
  ];
}

export function fitCamera(min: [number, number, number], max: [number, number, number]): Camera {
  const cx = (min[0] + max[0]) / 2, cy = (min[1] + max[1]) / 2, cz = (min[2] + max[2]) / 2;
  const dx = max[0] - min[0], dy = max[1] - min[1], dz = max[2] - min[2];
  const diag = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9);
  const distance = (diag * 0.5) / Math.tan(Math.PI / 12);
  return { theta: Math.PI / 4, phi: Math.PI / 4, distance, target: [cx, cy, cz] };
}

// ── matrices (identical to canvas3d.ts) ──
type Mat4 = Float32Array;
function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2), nf = 1 / (near - far), m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = (far + near) * nf; m[11] = -1; m[14] = 2 * far * near * nf; return m;
}
function mat4LookAt(eye: number[], center: number[], up: number[]): Mat4 {
  const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz); const z0 = zx / len, z1 = zy / len, z2 = zz / len;
  const xx = up[1] * z2 - up[2] * z1, xy = up[2] * z0 - up[0] * z2, xz = up[0] * z1 - up[1] * z0;
  len = Math.hypot(xx, xy, xz); const x0 = xx / len, x1 = xy / len, x2 = xz / len;
  const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  const m = new Float32Array(16);
  m[0] = x0; m[1] = y0; m[2] = z0; m[4] = x1; m[5] = y1; m[6] = z1; m[8] = x2; m[9] = y2; m[10] = z2;
  m[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]); m[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]); m[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]); m[15] = 1; return m;
}
function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    o[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
  return o;
}

interface MeshObj { posBuf: GPUBuffer; nrmBuf: GPUBuffer; scaBuf: GPUBuffer; count: number; tag: number; visible: boolean; objBG: GPUBindGroup; }
interface LineObj { segBuf: GPUBuffer; nseg: number; tag: number; visible: boolean; objBG: GPUBindGroup; }

export interface GPUState {
  device: GPUDevice;
  format: GPUTextureFormat;
  sampleCount: number;
  meshPipe: GPURenderPipeline;
  linePipe: GPURenderPipeline;
  meshFrameUB: GPUBuffer; meshFrameBG: GPUBindGroup;
  lineFrameUB: GPUBuffer; lineFrameBG: GPUBindGroup;
  meshes: MeshObj[];
  lineMeshes: LineObj[];
  clip_plane: [number, number, number, number];
  clip_enable: boolean;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  /** Wireframe half-width in framebuffer pixels (the original uses thin 1px
   *  gl.LINES; ~0.6 here matches that). Settable per render. */
  lineHalfPx: number;
}

export function initGPU(device: GPUDevice, format: GPUTextureFormat = 'rgba8unorm', sampleCount = 4): GPUState {
  const meshMod = device.createShaderModule({ code: MESH_WGSL });
  const lineMod = device.createShaderModule({ code: LINE_WGSL });
  const ds: GPUDepthStencilState = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
  const meshPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: meshMod, entryPoint: 'vs', buffers: [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }] }] },
    fragment: { module: meshMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: ds, multisample: { count: sampleCount },
  });
  const linePipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: lineMod, entryPoint: 'vs', buffers: [
      { arrayStride: 24, stepMode: 'instance', attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }] },
    fragment: { module: lineMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' }, depthStencil: { ...ds, depthCompare: 'less-equal' }, multisample: { count: sampleCount },
  });
  const meshFrameUB = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lineFrameUB = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const meshFrameBG = device.createBindGroup({ layout: meshPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: meshFrameUB } }] });
  const lineFrameBG = device.createBindGroup({ layout: linePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: lineFrameUB } }] });
  return { device, format, sampleCount, meshPipe, linePipe, meshFrameUB, meshFrameBG, lineFrameUB, lineFrameBG, meshes: [], lineMeshes: [], clip_plane: [0, 0, 0, 0], clip_enable: false, bbox: { min: [-1, -1, -1], max: [1, 1, 1] }, lineHalfPx: 0.6 };
}

export function clearMeshes(state: GPUState): void {
  for (const m of state.meshes) { m.posBuf.destroy(); m.nrmBuf.destroy(); m.scaBuf.destroy(); }
  for (const l of state.lineMeshes) l.segBuf.destroy();
  state.meshes = []; state.lineMeshes = [];
}

function vbuf(device: GPUDevice, a: Float32Array): GPUBuffer {
  const b = device.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, a); return b;
}

export function addMesh(
  state: GPUState, positions: Float32Array, normals: Float32Array,
  color: [number, number, number], tag = 0, _depth_offset?: [number, number], scalars?: Float32Array,
): void {
  const { device } = state;
  const count = positions.length / 3;
  const sca = scalars ?? new Float32Array(count);          // constant 0 when absent
  const objUB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(objUB, 0, new Float32Array([color[0], color[1], color[2], scalars ? 1 : 0]));
  const objBG = device.createBindGroup({ layout: state.meshPipe.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer: objUB } }] });
  state.meshes.push({ posBuf: vbuf(device, positions), nrmBuf: vbuf(device, normals), scaBuf: vbuf(device, sca), count, tag, visible: true, objBG });
}

/** Line segments: positions are 3-per-vertex, every two vertices = one segment. */
export function addLineMesh(state: GPUState, positions: Float32Array, color: [number, number, number], tag = 0): void {
  const { device } = state;
  const objUB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(objUB, 0, new Float32Array([color[0], color[1], color[2], 1]));
  const objBG = device.createBindGroup({ layout: state.linePipe.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer: objUB } }] });
  state.lineMeshes.push({ segBuf: vbuf(device, positions), nseg: positions.length / 6, tag, visible: true, objBG });
}

export function setTagVisible(state: GPUState, tag: number, visible: boolean): void {
  for (const m of state.meshes) if (m.tag === tag) m.visible = visible;
  for (const m of state.lineMeshes) if (m.tag === tag) m.visible = visible;
}
export function setClipPlane(state: GPUState, normal: [number, number, number], d: number, enable: boolean): void {
  state.clip_plane = [normal[0], normal[1], normal[2], d]; state.clip_enable = enable;
}
export function setBBox(state: GPUState, min: [number, number, number], max: [number, number, number]): void {
  state.bbox.min = min; state.bbox.max = max;
}

export interface RenderTarget { colorView: GPUTextureView; resolveView?: GPUTextureView; depthView: GPUTextureView; width: number; height: number; }

export function render3D(state: GPUState, camera: Camera, target: RenderTarget, zFlip = 1): void {
  const { device } = state;
  const w = target.width, h = target.height, aspect = w / h || 1;
  const dx = state.bbox.max[0] - state.bbox.min[0], dy = state.bbox.max[1] - state.bbox.min[1], dz = state.bbox.max[2] - state.bbox.min[2];
  const sceneR = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
  const near = Math.max(camera.distance * 1e-3, sceneR * 1e-3, 1e-9), far = (camera.distance + sceneR) * 8;
  const proj = mat4Perspective(Math.PI / 6, aspect, near, far);
  const eye = cameraEye(camera);
  const view = mat4LookAt(eye, camera.target as number[], [0, 0, 1]);
  const vp = mat4Multiply(proj, view);
  const nmat = view; // upper-left 3x3 columns extracted into the std140 slots below

  const ldx = eye[0] - camera.target[0], ldy = eye[1] - camera.target[1], ldz = eye[2] - camera.target[2] + camera.distance * 0.3;
  const ll = Math.hypot(ldx, ldy, ldz);

  // mesh frame uniform (160 B): mvp normalMat lightDir ambient clipPlane clipEnable zFlip
  const mf = new Float32Array(40); mf.set(vp, 0);
  mf[16] = nmat[0]; mf[17] = nmat[1]; mf[18] = nmat[2]; mf[20] = nmat[4]; mf[21] = nmat[5]; mf[22] = nmat[6]; mf[24] = nmat[8]; mf[25] = nmat[9]; mf[26] = nmat[10];
  mf[28] = ldx / ll; mf[29] = ldy / ll; mf[30] = ldz / ll; mf[31] = 0.8;
  mf[32] = state.clip_plane[0]; mf[33] = state.clip_plane[1]; mf[34] = state.clip_plane[2]; mf[35] = state.clip_plane[3];
  mf[36] = state.clip_enable ? 1 : 0; mf[37] = zFlip;
  device.queue.writeBuffer(state.meshFrameUB, 0, mf);
  // line frame uniform (80 B): mvp viewport halfPx
  const lf = new Float32Array(20); lf.set(vp, 0); lf[16] = w; lf[17] = h; lf[18] = state.lineHalfPx;
  device.queue.writeBuffer(state.lineFrameUB, 0, lf);

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: target.colorView, resolveTarget: target.resolveView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: target.resolveView ? 'discard' : 'store' }],
    depthStencilAttachment: { view: target.depthView, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
  });
  pass.setPipeline(state.meshPipe); pass.setBindGroup(0, state.meshFrameBG);
  for (const m of state.meshes) {
    if (!m.visible) continue;
    pass.setBindGroup(1, m.objBG); pass.setVertexBuffer(0, m.posBuf); pass.setVertexBuffer(1, m.nrmBuf); pass.setVertexBuffer(2, m.scaBuf); pass.draw(m.count);
  }
  if (state.lineMeshes.length) {
    pass.setPipeline(state.linePipe); pass.setBindGroup(0, state.lineFrameBG);
    for (const l of state.lineMeshes) { if (!l.visible || l.nseg === 0) continue; pass.setBindGroup(1, l.objBG); pass.setVertexBuffer(0, l.segBuf); pass.draw(6, l.nseg); }
  }
  pass.end();
  device.queue.submit([enc.finish()]);
}
