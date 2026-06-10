/**
 * rapidmesh comparison viewer: one panel per mesher (rapidmesh / gmsh /
 * tetgen) with a shared, synchronized camera. Surface faces are colored by
 * region (tagged sheets highlighted), wireframes and tet fills are
 * toggleable, and a movable clip plane exposes the interior tetrahedra.
 */

import {
	addLineMesh,
	addMesh,
	clearMeshes,
	fitCamera,
	initGL,
	render3D,
	setBBox,
	setClipPlane,
	setTagVisible,
	type Camera,
	type GLState
} from './canvas3d';
import { canvas as canvasTheme, palette, plotColors, toCSSVars } from './theme';

function hexToRgb(hex: string): [number, number, number] {
	return [
		parseInt(hex.slice(1, 3), 16) / 255,
		parseInt(hex.slice(3, 5), 16) / 255,
		parseInt(hex.slice(5, 7), 16) / 255
	];
}

/** Region colors from the rapidfem plot trace cycle. */
const regionPalette = plotColors.cycle.map(hexToRgb);
/** Tagged sheet faces (PEC, ports) use the secondary accent. */
const pecColor = hexToRgb(palette.accentSecondary);
const wireSurface = hexToRgb(canvasTheme.bg);
const wireTets = hexToRgb(palette.accentPurple);

const MESHERS = ['rapidmesh', 'gmsh', 'tetgen'];

// Display tags for visibility toggles.
const TAG_SURFACE = 1;
const TAG_TETFILL = 2;
const TAG_WIRE_SURFACE = 3;
const TAG_WIRE_TETS = 4;

interface MeshJson {
	name: string;
	mesher: string;
	points: [number, number, number][];
	tets: [number, number, number, number][];
	tet_regions: number[];
	faces: { tri: [number, number, number]; tag: number; regions: [number, number] }[];
	stats: {
		n_points: number;
		n_tets: number;
		min_dihedral_deg: number;
		max_radius_edge: number;
		max_edge: number;
		millis: number;
	};
}

interface Panel {
	canvas: HTMLCanvasElement;
	gl: GLState;
	bbox: { min: [number, number, number]; max: [number, number, number] };
}

const panels: Panel[] = [];
let camera: Camera = { theta: Math.PI / 4, phi: Math.PI / 4, distance: 10, target: [0, 0, 0] };
let sceneBBox = { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] };

function regionColor(r: number): [number, number, number] {
	return regionPalette[(r + regionPalette.length - 1) % regionPalette.length];
}

// ─── Geometry upload ────────────────────────────────────────────────

function triNormal(
	pts: [number, number, number][],
	a: number,
	b: number,
	c: number
): [number, number, number] {
	const u = [pts[b][0] - pts[a][0], pts[b][1] - pts[a][1], pts[b][2] - pts[a][2]];
	const v = [pts[c][0] - pts[a][0], pts[c][1] - pts[a][1], pts[c][2] - pts[a][2]];
	const n: [number, number, number] = [
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0]
	];
	const l = Math.hypot(n[0], n[1], n[2]) || 1;
	return [n[0] / l, n[1] / l, n[2] / l];
}

function buildPanel(mesh: MeshJson, container: HTMLElement): Panel | null {
	const card = document.createElement('div');
	card.className = 'panel';
	const title = document.createElement('div');
	title.className = 'panel-title';
	const s = mesh.stats;
	title.innerHTML =
		`<b>${mesh.mesher}</b>` +
		`<span>${s.n_tets} tets · ${s.n_points} pts · min∠ ${s.min_dihedral_deg.toFixed(1)}° · ` +
		`r/e ${s.max_radius_edge.toFixed(2)} · ${s.millis} ms</span>`;
	const canvas = document.createElement('canvas');
	card.appendChild(title);
	card.appendChild(canvas);
	container.appendChild(card);

	const gl = initGL(canvas);
	if (!gl) return null;

	const pts = mesh.points;

	// Bounding box.
	const min: [number, number, number] = [Infinity, Infinity, Infinity];
	const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
	for (const p of pts) {
		for (let k = 0; k < 3; k++) {
			min[k] = Math.min(min[k], p[k]);
			max[k] = Math.max(max[k], p[k]);
		}
	}
	setBBox(gl, min, max);

	// Surface faces grouped by color (tagged sheets win over regions).
	const groups = new Map<string, { color: [number, number, number]; tris: [number, number, number][] }>();
	for (const f of mesh.faces) {
		const color = f.tag !== 0 ? pecColor : regionColor(Math.max(f.regions[0], f.regions[1]));
		const key = color.join(',');
		if (!groups.has(key)) groups.set(key, { color, tris: [] });
		groups.get(key)!.tris.push(f.tri);
	}
	for (const g of groups.values()) {
		const pos: number[] = [];
		const nrm: number[] = [];
		for (const [a, b, c] of g.tris) {
			const n = triNormal(pts, a, b, c);
			for (const v of [a, b, c]) {
				pos.push(pts[v][0], pts[v][1], pts[v][2]);
				nrm.push(n[0], n[1], n[2]);
			}
		}
		addMesh(gl, new Float32Array(pos), new Float32Array(nrm), g.color, TAG_SURFACE, [-1, -1]);
	}

	// Tet fill (all faces, colored per region): visible interior under clip.
	const byRegion = new Map<number, number[]>();
	for (let ti = 0; ti < mesh.tets.length; ti++) {
		const t = mesh.tets[ti];
		const r = mesh.tet_regions[ti];
		if (!byRegion.has(r)) byRegion.set(r, []);
		byRegion.get(r)!.push(ti);
	}
	for (const [r, tis] of byRegion) {
		const pos: number[] = [];
		const nrm: number[] = [];
		for (const ti of tis) {
			const t = mesh.tets[ti];
			for (const [a, b, c] of [
				[t[1], t[3], t[2]],
				[t[0], t[2], t[3]],
				[t[0], t[3], t[1]],
				[t[0], t[1], t[2]]
			] as [number, number, number][]) {
				const n = triNormal(pts, a, b, c);
				for (const v of [a, b, c]) {
					pos.push(pts[v][0], pts[v][1], pts[v][2]);
					nrm.push(n[0], n[1], n[2]);
				}
			}
		}
		addMesh(gl, new Float32Array(pos), new Float32Array(nrm), regionColor(r), TAG_TETFILL);
	}

	// Wireframes: surface edges and all tet edges (deduplicated).
	const surfEdges = new Set<string>();
	const surfLines: number[] = [];
	for (const f of mesh.faces) {
		for (let e = 0; e < 3; e++) {
			const a = f.tri[e];
			const b = f.tri[(e + 1) % 3];
			const key = a < b ? `${a},${b}` : `${b},${a}`;
			if (surfEdges.has(key)) continue;
			surfEdges.add(key);
			surfLines.push(pts[a][0], pts[a][1], pts[a][2], pts[b][0], pts[b][1], pts[b][2]);
		}
	}
	addLineMesh(gl, new Float32Array(surfLines), wireSurface, TAG_WIRE_SURFACE);

	const tetEdges = new Set<string>();
	const tetLines: number[] = [];
	for (const t of mesh.tets) {
		for (let i = 0; i < 4; i++) {
			for (let j = i + 1; j < 4; j++) {
				const a = t[i];
				const b = t[j];
				const key = a < b ? `${a},${b}` : `${b},${a}`;
				if (tetEdges.has(key)) continue;
				tetEdges.add(key);
				tetLines.push(pts[a][0], pts[a][1], pts[a][2], pts[b][0], pts[b][1], pts[b][2]);
			}
		}
	}
	addLineMesh(gl, new Float32Array(tetLines), wireTets, TAG_WIRE_TETS);

	const panel: Panel = { canvas, gl, bbox: { min, max } };
	attachCameraControls(canvas);
	return panel;
}

// ─── Camera controls (shared across panels) ─────────────────────────

function attachCameraControls(canvas: HTMLCanvasElement) {
	let dragging = false;
	let panning = false;
	let lastX = 0;
	let lastY = 0;
	canvas.addEventListener('mousedown', (ev) => {
		dragging = !ev.shiftKey;
		panning = ev.shiftKey;
		lastX = ev.clientX;
		lastY = ev.clientY;
	});
	window.addEventListener('mouseup', () => {
		dragging = false;
		panning = false;
	});
	window.addEventListener('mousemove', (ev) => {
		const dx = ev.clientX - lastX;
		const dy = ev.clientY - lastY;
		if (dragging) {
			camera.theta += dx * 0.008;
			camera.phi = Math.min(Math.PI / 2 - 0.01, Math.max(-Math.PI / 2 + 0.01, camera.phi + dy * 0.008));
			lastX = ev.clientX;
			lastY = ev.clientY;
		} else if (panning) {
			const scale = camera.distance * 0.0015;
			const ct = Math.cos(camera.theta);
			const st = Math.sin(camera.theta);
			camera.target[0] -= (dx * ct) * scale;
			camera.target[1] += (dx * st) * scale;
			camera.target[2] += dy * scale;
			lastX = ev.clientX;
			lastY = ev.clientY;
		}
	});
	canvas.addEventListener(
		'wheel',
		(ev) => {
			ev.preventDefault();
			camera.distance *= Math.exp(ev.deltaY * 0.001);
		},
		{ passive: false }
	);
}

// ─── UI shell ───────────────────────────────────────────────────────

// CSS custom properties come from the shared theme, exactly as in rapidfem.
const styleEl = document.createElement('style');
styleEl.textContent = `:root { ${toCSSVars()} }`;
document.head.appendChild(styleEl);

const app = document.getElementById('app')!;
app.innerHTML = `
	<header>
		<span class="brand">rapidmesh viewer</span>
		<label>Geometry <select id="geometry"></select></label>
		<label><input type="checkbox" id="show-surface" checked> Surface</label>
		<label><input type="checkbox" id="show-surface-wire" checked> Wireframe</label>
		<label><input type="checkbox" id="show-tets"> Tet fill</label>
		<label><input type="checkbox" id="show-tet-wire"> Tet edges</label>
		<label><input type="checkbox" id="clip-enable"> Clip</label>
		<select id="clip-axis"><option value="0">x</option><option value="1">y</option><option value="2" selected>z</option></select>
		<input type="range" id="clip-pos" min="0" max="1" step="0.005" value="0.5">
		<span class="hint">drag rotate · shift-drag pan · wheel zoom</span>
	</header>
	<div id="panels"></div>
`;

const panelsDiv = document.getElementById('panels')!;
const geometrySel = document.getElementById('geometry') as HTMLSelectElement;

function applyToggles() {
	const surf = (document.getElementById('show-surface') as HTMLInputElement).checked;
	const surfWire = (document.getElementById('show-surface-wire') as HTMLInputElement).checked;
	const tets = (document.getElementById('show-tets') as HTMLInputElement).checked;
	const tetWire = (document.getElementById('show-tet-wire') as HTMLInputElement).checked;
	for (const p of panels) {
		setTagVisible(p.gl, TAG_SURFACE, surf);
		setTagVisible(p.gl, TAG_WIRE_SURFACE, surfWire);
		setTagVisible(p.gl, TAG_TETFILL, tets);
		setTagVisible(p.gl, TAG_WIRE_TETS, tetWire);
	}
}

function applyClip() {
	const enable = (document.getElementById('clip-enable') as HTMLInputElement).checked;
	const axis = parseInt((document.getElementById('clip-axis') as HTMLSelectElement).value);
	const tpos = parseFloat((document.getElementById('clip-pos') as HTMLInputElement).value);
	const normal: [number, number, number] = [0, 0, 0];
	normal[axis] = 1;
	const d = sceneBBox.min[axis] + tpos * (sceneBBox.max[axis] - sceneBBox.min[axis]);
	for (const p of panels) {
		setClipPlane(p.gl, normal, d, enable);
	}
}

for (const id of ['show-surface', 'show-surface-wire', 'show-tets', 'show-tet-wire']) {
	document.getElementById(id)!.addEventListener('change', applyToggles);
}
for (const id of ['clip-enable', 'clip-axis', 'clip-pos']) {
	document.getElementById(id)!.addEventListener('input', applyClip);
}

async function loadGeometry(name: string) {
	for (const p of panels) clearMeshes(p.gl);
	panels.length = 0;
	panelsDiv.innerHTML = '';
	let first = true;
	for (const mesher of MESHERS) {
		try {
			const resp = await fetch(`meshes/${mesher}_${name}.json`);
			if (!resp.ok) continue;
			const mesh = (await resp.json()) as MeshJson;
			const panel = buildPanel(mesh, panelsDiv);
			if (!panel) continue;
			panels.push(panel);
			if (first) {
				sceneBBox = panel.bbox;
				camera = fitCamera(panel.bbox.min, panel.bbox.max);
				first = false;
			}
		} catch {
			// Mesher output not available for this geometry: skip.
		}
	}
	applyToggles();
	applyClip();
}

async function boot() {
	const manifest = (await (await fetch('meshes/manifest.json')).json()) as string[];
	for (const name of manifest) {
		const opt = document.createElement('option');
		opt.value = name;
		opt.textContent = name;
		geometrySel.appendChild(opt);
	}
	geometrySel.addEventListener('change', () => loadGeometry(geometrySel.value));
	if (manifest.length > 0) await loadGeometry(manifest[0]);

	function frame() {
		for (const p of panels) {
			const w = p.canvas.clientWidth;
			const h = p.canvas.clientHeight;
			if (p.canvas.width !== w || p.canvas.height !== h) {
				p.canvas.width = w;
				p.canvas.height = h;
			}
			render3D(p.gl, camera, w, h);
		}
		requestAnimationFrame(frame);
	}
	requestAnimationFrame(frame);
}

boot();
