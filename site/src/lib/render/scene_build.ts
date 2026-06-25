/**
 * Shared, backend-agnostic scene builder -- the EXACT geometry+colour pipeline
 * extracted verbatim from MeshViewer.svelte's `build_for_axis`, so the browser
 * viewer and the headless Node rasterizer produce identical scenes. Takes a
 * `canvas3d`-style API ({ clearMeshes, setBBox, addMesh, addLineMesh }) which is
 * the SAME for the WebGL2 (`canvas3d.ts`) and WebGPU (`canvas3d_webgpu.ts`)
 * backends, so colours/faces/wire are bit-for-bit the same.
 */
import { buildTriSoupF64, buildVolumeBoundaries } from './mesh_scene';
import type { MeshData } from '../msh';
import { palette, canvas as canvasTheme, plotColors } from '../theme';

type RGB = [number, number, number];
type Kind = 'dielectric' | 'conductor' | 'port' | 'gnd';

const TAG_WIRE_SURF = -11;
const TAG_TET_WIRE = -13;
const TAG_FEAT_EDGES = -12;
const TAG_DEFECTS = -14;

const DIELECTRIC_CYCLE = ['#4a9ec2', '#6bbf8a', '#7b5e8a', '#a78bd9', '#c4c46b'];
// Defect-marker colours by kind (match render_gallery's legend / viewer).
const DEFECT_COLORS: Record<string, RGB> = {
  sliver: [1.0, 0.749, 0.0],
  straddler: [1.0, 0.102, 0.8],
  nonmanifold_edge: [1.0, 0.102, 0.102],
};

function hex(s: string): RGB {
  return [parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255];
}
function base(name: string): string { return name.replace(/_\d+$/, ''); }
function classify(name: string): Kind | null {
  const b = base(name);
  if (b === 'region') return 'dielectric';
  if (b === 'air' || b === 'dielectric' || b === 'anisotropic') return 'dielectric';
  if (b === 'conductor') return 'conductor';
  if (b === 'port') return 'port';
  if (b === 'pec' || b === 'pmc' || b === 'surfaceimpedance' || b === 'lumpedelement') return 'conductor';
  if (b === 'abc') return null;
  if (b === 'pml') return 'dielectric';
  if (name.startsWith('_mat_')) return null;
  if (name === 'substrate' || name === 'oxide') return 'dielectric';
  if (name.endsWith('_gnd') || name === 'gnd' || name === 'ground') return 'gnd';
  if (name === 'p1' || name === 'p2' || /^p\d+$/.test(name) || name.endsWith('_port')) return 'port';
  return 'conductor';
}

const WIRE_SURF_COLOR: RGB = hex(canvasTheme.crosshair);
const WIRE_INT_COLOR: RGB = hex(canvasTheme.grid);
const FEAT_EDGE_COLOR: RGB = hex(palette.accentSecondary);

export interface SceneApi {
  clearMeshes(state: unknown): void;
  setBBox(state: unknown, min: RGB, max: RGB): void;
  addMesh(state: unknown, positions: Float32Array, normals: Float32Array, color: RGB, tag?: number): void;
  addLineMesh(state: unknown, positions: Float32Array, color: RGB, tag?: number): void;
}

export interface SceneOpts {
  clipAxis?: 0 | 1 | 2;
  clipT?: number | null;          // crinkle-clip fraction along clipAxis (null = full)
  fills?: boolean;                // tet/surface fills
  surfWire?: boolean;
  intWire?: boolean;
  featEdges?: boolean;
  defects?: boolean;
}

/** group_colors: each phys_name gets the next plotColors.cycle hue in tag order. */
function groupColors(m: MeshData): Map<string, RGB> {
  const map = new Map<string, RGB>();
  const tags = [...m.phys_names.keys()].sort((a, b) => a - b);
  let i = 0;
  for (const t of tags) {
    const name = m.phys_names.get(t)!;
    if (!map.has(name)) map.set(name, hex(plotColors.cycle[i++ % plotColors.cycle.length]));
  }
  return map;
}

/** Populate the renderer state with the EXACT scene MeshViewer builds. */
export function buildScene(state: unknown, m: MeshData, api: SceneApi, opts: SceneOpts = {}): void {
  const axis = opts.clipAxis ?? 1;
  const clipT = opts.clipT ?? null;
  const L = { fills: opts.fills ?? true, surfWire: opts.surfWire ?? true, intWire: opts.intWire ?? false, featEdges: opts.featEdges ?? true, defects: opts.defects ?? false };
  const gc = groupColors(m);
  const color_for = (kind: Kind, name: string): RGB => {
    const g = gc.get(name); if (g) return g;
    const b = base(name);
    if (b === 'region') { const mt = name.match(/_(\d+)$/); const idx = mt ? Math.max(0, parseInt(mt[1], 10)) : 0; return hex(plotColors.cycle[idx % plotColors.cycle.length]); }
    if (b === 'air') return hex('#5a5a62');
    if (b === 'conductor') return hex(palette.accentSecondary);
    if (b === 'dielectric' || b === 'anisotropic') { const mt = name.match(/_(\d+)$/); const idx = mt ? Math.max(0, parseInt(mt[1], 10) - 1) : 0; return hex(DIELECTRIC_CYCLE[idx % DIELECTRIC_CYCLE.length]); }
    if (b === 'pml') return hex('#7b5e8a');
    return hex(palette.accentSecondary);
  };

  api.clearMeshes(state);
  api.setBBox(state, m.bbox.min as RGB, m.bbox.max as RGB);
  const np = m.nodes, nf = m.tri_phys.length, nt = m.tet_phys.length;
  const face_cv = (fi: number) => (np[m.tris[fi * 3] * 3 + axis] + np[m.tris[fi * 3 + 1] * 3 + axis] + np[m.tris[fi * 3 + 2] * 3 + axis]) / 3;
  const tet_cv = (ti: number) => (np[m.tets[ti * 4] * 3 + axis] + np[m.tets[ti * 4 + 1] * 3 + axis] + np[m.tets[ti * 4 + 2] * 3 + axis] + np[m.tets[ti * 4 + 3] * 3 + axis]) / 4;
  // crinkle-clip (EXACT match to MeshViewer.apply_clip_counts): a single GLOBAL
  // centroid threshold `d` along the axis -- every group (fills AND wire) keeps
  // the units whose centroid is <= d, so they all cut at the same plane. (The
  // per-list fraction this replaced cut each group at its own quantile, leaving
  // filled tets without their edges.)
  const clipD = clipT == null ? Infinity : m.bbox.min[axis] + clipT * (m.bbox.max[axis] - m.bbox.min[axis]);
  const ub = (vals: number[]): number => {            // count of ascending vals <= clipD
    if (clipD === Infinity) return vals.length;
    let lo = 0, hi = vals.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (vals[mid] <= clipD) lo = mid + 1; else hi = mid; }
    return lo;
  };

  // ---- surface fills (named tris, dim 2) ----
  if (L.fills) {
    const by_surf = new Map<number, number[]>();
    for (let f = 0; f < nf; f++) { const tag = m.tri_phys[f]; if (!tag || (m.phys_dim.get(tag) ?? 2) !== 2) continue; (by_surf.get(tag) ?? by_surf.set(tag, []).get(tag)!).push(f); }
    for (const [tag, fis] of by_surf) {
      const name = m.phys_names.get(tag) ?? ''; const kind = classify(name); if (!kind) continue;
      fis.sort((a, b) => face_cv(a) - face_cv(b));
      const k = ub(fis.map(face_cv));
      const flat: number[] = new Array(k * 3);
      for (let i = 0; i < k; i++) { const fi = fis[i]; flat[i * 3] = m.tris[fi * 3]; flat[i * 3 + 1] = m.tris[fi * 3 + 1]; flat[i * 3 + 2] = m.tris[fi * 3 + 2]; }
      if (k) { const { positions, normals } = buildTriSoupF64(np, flat); api.addMesh(state, positions, normals, color_for(kind, name), tag); }
    }
  }

  // ---- volume hulls -> only feed surface wire ----
  const hull_wire_groups: { tag: number; tris: number[] }[] = [];
  const vol_b = buildVolumeBoundaries(m as any);
  for (const [vtag, idx] of vol_b.entries()) {
    const name = m.phys_names.get(vtag) ?? ''; if (!name) continue; if (!classify(name)) continue;
    const ntri = idx.length / 3, order = Array.from({ length: ntri }, (_, i) => i);
    order.sort((a, b) => ((np[idx[a * 3] * 3 + axis] + np[idx[a * 3 + 1] * 3 + axis] + np[idx[a * 3 + 2] * 3 + axis]) / 3) - ((np[idx[b * 3] * 3 + axis] + np[idx[b * 3 + 1] * 3 + axis] + np[idx[b * 3 + 2] * 3 + axis]) / 3));
    const s: number[] = new Array(ntri * 3); for (let i = 0; i < ntri; i++) { const t = order[i]; s[i * 3] = idx[t * 3]; s[i * 3 + 1] = idx[t * 3 + 1]; s[i * 3 + 2] = idx[t * 3 + 2]; }
    hull_wire_groups.push({ tag: vtag, tris: s });
  }

  // ---- tet fills per region (the dev-viewer volume look) ----
  if (L.fills) {
    const by_region = new Map<number, number[]>();
    for (let ti = 0; ti < nt; ti++) { const r = m.tet_phys[ti]; (by_region.get(r) ?? by_region.set(r, []).get(r)!).push(ti); }
    for (const [r, tis] of by_region) {
      const name = m.phys_names.get(r) ?? ''; const kind = classify(name); if (!kind) continue;
      tis.sort((a, b) => tet_cv(a) - tet_cv(b));
      const k = ub(tis.map(tet_cv));
      const flat: number[] = [];
      for (let i = 0; i < k; i++) { const ti = tis[i]; const t0 = m.tets[ti * 4], t1 = m.tets[ti * 4 + 1], t2 = m.tets[ti * 4 + 2], t3 = m.tets[ti * 4 + 3]; flat.push(t1, t3, t2, t0, t2, t3, t0, t3, t1, t0, t1, t2); }
      if (k) { const { positions, normals } = buildTriSoupF64(np, flat); api.addMesh(state, positions, normals, color_for(kind, name), r); }
    }
  }

  // ---- surface wireframe per group ----
  type EdgeMap = Map<bigint, { a: number; b: number; val: number }>;
  const surf_edge_groups = new Map<number, EdgeMap>();
  const gem = (g: number): EdgeMap => surf_edge_groups.get(g) ?? surf_edge_groups.set(g, new Map()).get(g)!;
  const add_edge = (em: EdgeMap, u: number, w: number, fv: number) => { const lo = u < w ? u : w, hi = u < w ? w : u; const k = (BigInt(lo) << 32n) | BigInt(hi); const c = em.get(k); if (!c) em.set(k, { a: u, b: w, val: fv }); else if (fv < c.val) c.val = fv; };
  for (let f = 0; f < nf; f++) { if (!m.tri_phys[f]) continue; const em = gem(m.tri_phys[f]); const fv = face_cv(f); const a = m.tris[f * 3], b = m.tris[f * 3 + 1], c = m.tris[f * 3 + 2]; add_edge(em, a, b, fv); add_edge(em, b, c, fv); add_edge(em, c, a, fv); }
  for (const hw of hull_wire_groups) { const em = gem(hw.tag); for (let h = 0; h + 2 < hw.tris.length; h += 3) { const a = hw.tris[h], b = hw.tris[h + 1], c = hw.tris[h + 2]; const fv = (np[a * 3 + axis] + np[b * 3 + axis] + np[c * 3 + axis]) / 3; add_edge(em, a, b, fv); add_edge(em, b, c, fv); add_edge(em, c, a, fv); } }
  const surf_edge_set = new Set<bigint>();
  if (L.surfWire) {
    for (const [, em] of surf_edge_groups) {
      for (const k of em.keys()) surf_edge_set.add(k);
      const edges = [...em.values()].sort((x, y) => x.val - y.val);
      const k = ub(edges.map((e) => e.val));
      const pos = new Float32Array(k * 6);
      for (let i = 0; i < k; i++) { const e = edges[i]; pos[i * 6] = np[e.a * 3]; pos[i * 6 + 1] = np[e.a * 3 + 1]; pos[i * 6 + 2] = np[e.a * 3 + 2]; pos[i * 6 + 3] = np[e.b * 3]; pos[i * 6 + 4] = np[e.b * 3 + 1]; pos[i * 6 + 5] = np[e.b * 3 + 2]; }
      if (k) api.addLineMesh(state, pos, WIRE_SURF_COLOR, TAG_WIRE_SURF);
    }
  } else { for (const [, em] of surf_edge_groups) for (const k of em.keys()) surf_edge_set.add(k); }

  // ---- interior tet wireframe per group ----
  if (L.intWire) {
    const int_groups = new Map<number, EdgeMap>();
    for (let ti = 0; ti < nt; ti++) { const g = m.tet_phys[ti]; const em = int_groups.get(g) ?? int_groups.set(g, new Map()).get(g)!; const tv = tet_cv(ti); const v = [m.tets[ti * 4], m.tets[ti * 4 + 1], m.tets[ti * 4 + 2], m.tets[ti * 4 + 3]]; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) { const u = v[i], w = v[j], lo = u < w ? u : w, hi = u < w ? w : u, k = (BigInt(lo) << 32n) | BigInt(hi); if (surf_edge_set.has(k)) continue; const c = em.get(k); if (!c) em.set(k, { a: u, b: w, val: tv }); else if (tv < c.val) c.val = tv; } }
    for (const [, em] of int_groups) {
      const edges = [...em.values()].sort((x, y) => x.val - y.val);
      const k = ub(edges.map((e) => e.val));
      const pos = new Float32Array(k * 6);
      for (let i = 0; i < k; i++) { const e = edges[i]; pos[i * 6] = np[e.a * 3]; pos[i * 6 + 1] = np[e.a * 3 + 1]; pos[i * 6 + 2] = np[e.a * 3 + 2]; pos[i * 6 + 3] = np[e.b * 3]; pos[i * 6 + 4] = np[e.b * 3 + 1]; pos[i * 6 + 5] = np[e.b * 3 + 2]; }
      if (k) api.addLineMesh(state, pos, WIRE_INT_COLOR, TAG_TET_WIRE);
    }
  }

  // ---- feature edges (never clipped) ----
  if (L.featEdges && m.edges && m.edges.length >= 2) {
    const ne = (m.edges.length / 2) | 0, pos = new Float32Array(ne * 6);
    for (let i = 0; i < ne; i++) { const a = m.edges[i * 2], b = m.edges[i * 2 + 1]; pos[i * 6] = np[a * 3]; pos[i * 6 + 1] = np[a * 3 + 1]; pos[i * 6 + 2] = np[a * 3 + 2]; pos[i * 6 + 3] = np[b * 3]; pos[i * 6 + 4] = np[b * 3 + 1]; pos[i * 6 + 5] = np[b * 3 + 2]; }
    api.addLineMesh(state, pos, FEAT_EDGE_COLOR, TAG_FEAT_EDGES);
  }

  // ---- defect markers (3D crosses per kind) ----
  if (L.defects && m.defects && m.defects.length > 0) {
    const d = m.bbox, diag = Math.hypot(d.max[0] - d.min[0], d.max[1] - d.min[1], d.max[2] - d.min[2]), r = 0.012 * (diag || 1);
    const byKind = new Map<string, number[]>();
    for (const f of m.defects) { const [x, y, z] = f.pos; const arr = byKind.get(f.kind) ?? byKind.set(f.kind, []).get(f.kind)!; arr.push(x - r, y, z, x + r, y, z, x, y - r, z, x, y + r, z, x, y, z - r, x, y, z + r); }
    for (const [kind, lines] of byKind) api.addLineMesh(state, Float32Array.from(lines), DEFECT_COLORS[kind] ?? [0.6, 0.6, 0.6], TAG_DEFECTS);
  }
}
