// Headless WebGPU rasterizer -- runs the SHARED browser pipeline (mesh_adapter +
// scene_build + canvas3d_webgpu, bundled into bundle.mjs) under a headless
// @kmamal/gpu device, so the PNGs are 1:1 with the live viewer. Replaces the
// per-image playwright/chromium screenshotter (no browser, no leak).
//
// Two modes, ONE persistent GPU device + pipelines either way:
//   node rasterize.mjs <jobs.json>   -- batch: render every job, then exit.
//   node rasterize.mjs --stream      -- stream: read one job JSON per stdin line,
//                                       render it immediately, print `DONE <out>`
//                                       (so Python gets each image the moment its
//                                       mesh is ready), exit on stdin EOF.
// job: { mesh, out, clip:0.55|null, clipAxis, fills, surfWire, intWire, featEdges,
//        defects, lineHalfPx, azim, elev, width, height }

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 }
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 }
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 }
globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }

import gpu from '@kmamal/gpu'
import { PNG } from 'pngjs'
import fs from 'node:fs'
import readline from 'node:readline'
import { adaptMesh, buildScene, gpu3d } from './bundle.mjs'

const SAMPLES = 4
const DEG = Math.PI / 180

const instance = gpu.create([])
const adapter = await instance.requestAdapter()
const device = await adapter.requestDevice()
const fmt = 'rgba8unorm'
const state = gpu3d.initGPU(device, fmt, SAMPLES)
const api = { clearMeshes: gpu3d.clearMeshes, setBBox: gpu3d.setBBox, addMesh: gpu3d.addMesh, addLineMesh: gpu3d.addLineMesh }

// Render targets are (re)created only when the requested size changes.
let T = null
function ensureTargets(w, h) {
  if (T && T.w === w && T.h === h) return T
  if (T) { T.msaa.destroy(); T.resolved.destroy(); T.depth.destroy(); T.readback.destroy() }
  const msaa = device.createTexture({ size: [w, h], format: fmt, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT })
  const resolved = device.createTexture({ size: [w, h], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const depth = device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT })
  const bpr = Math.ceil(w * 4 / 256) * 256
  const readback = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  T = { w, h, msaa, resolved, depth, bpr, readback }
  return T
}

async function renderJob(job) {
  const w = job.width ?? 1500, h = job.height ?? 1230
  const t = ensureTargets(w, h)
  const j = JSON.parse(fs.readFileSync(job.mesh, 'utf8'))
  const mesh = adaptMesh(j)
  buildScene(state, mesh, api, {
    clipAxis: job.clipAxis ?? 1, clipT: job.clip ?? null,
    fills: job.fills ?? true, surfWire: job.surfWire ?? true, intWire: job.intWire ?? false,
    featEdges: job.featEdges ?? false, defects: job.defects ?? false,
  })
  state.lineHalfPx = job.lineHalfPx ?? 0.6
  const cam = gpu3d.fitCamera(mesh.bbox.min, mesh.bbox.max)
  cam.theta = (job.azim ?? 32) * DEG
  cam.phi = (job.elev ?? 32) * DEG
  gpu3d.render3D(state, cam, { colorView: t.msaa.createView(), resolveView: t.resolved.createView(), depthView: t.depth.createView(), width: w, height: h }, 1)

  const enc = device.createCommandEncoder()
  enc.copyTextureToBuffer({ texture: t.resolved }, { buffer: t.readback, bytesPerRow: t.bpr }, [w, h])
  device.queue.submit([enc.finish()])
  await t.readback.mapAsync(GPUMapMode.READ)
  const src = new Uint8Array(t.readback.getMappedRange())
  const png = new PNG({ width: w, height: h, colorType: 6 })
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const s = y * t.bpr + x * 4, d = (y * w + x) * 4; png.data[d] = src[s]; png.data[d + 1] = src[s + 1]; png.data[d + 2] = src[s + 2]; png.data[d + 3] = src[s + 3] }
  fs.mkdirSync((job.out.match(/^(.*)[\\/]/) || [, '.'])[1], { recursive: true })
  fs.writeFileSync(job.out, PNG.sync.write(png))
  t.readback.unmap()
}

const arg = process.argv[2]
if (arg && arg !== '--stream' && fs.existsSync(arg)) {
  // batch
  const jobs = JSON.parse(fs.readFileSync(arg, 'utf8'))
  const t0 = Date.now()
  let n = 0
  for (const job of jobs) {
    try { await renderJob(job); n++; process.stdout.write(`  ${job.out.split(/[\\/]/).pop()}\n`) }
    catch (e) { process.stdout.write(`  FAIL ${job.mesh}: ${e.message}\n`) }
  }
  process.stdout.write(`rasterized ${n}/${jobs.length} in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`)
} else {
  // stream: one job per stdin line, render immediately, ack with DONE/FAIL.
  const rl = readline.createInterface({ input: process.stdin })
  for await (const line of rl) {
    const s = line.trim(); if (!s) continue
    let job
    try { job = JSON.parse(s) } catch (e) { process.stdout.write(`FAIL parse ${e.message}\n`); continue }
    try { await renderJob(job); process.stdout.write(`DONE ${job.out}\n`) }
    catch (e) { process.stdout.write(`FAIL ${job.out} ${e.message}\n`) }
  }
}
device.destroy(); gpu.destroy(instance)
