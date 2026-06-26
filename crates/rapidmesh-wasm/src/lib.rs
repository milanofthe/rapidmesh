//! WASM entry point for the landing page: triangulate the browser viewport
//! around the "RapidMESH" wordmark, live, with the project's own 2D mesher
//! (`rapidmesh_tet::surf2d::mesh_polygon`). The viewport rectangle is the outer
//! contour, the glyph outlines are holes, and a graded sizing field makes the
//! mesh fine at the wordmark and coarse out in the viewport -- the same path the
//! surface stage meshes coils with.
//!
//! [`triangulate`] returns the finished mesh; [`triangulate_steps`] returns the
//! mesher's coarse-to-fine per-pass intermediate states, so the page can animate
//! the refinement live.

use rapidmesh_tet::quadfield::QuadtreeField;
use rapidmesh_tet::surf2d::{mesh_polygon, mesh_polygon_with};
use wasm_bindgen::prelude::*;

type P2 = [f64; 2];

/// CVT (Lloyd) relaxation passes for the interior seed. Each is a full Delaunay
/// rebuild -- the dominant cost. The wordmark is a display mesh, not an FEM run,
/// so a few passes (Ruppert smooths after) suffice; the surface stage uses more.
const CVT_ITERS: usize = 4;

/// Ruppert refinement-pass cap. Each pass re-triangulates, and from a good CVT
/// seed the boundary conformity + angle bound are met in the first few; the rest
/// is marginal density. A low cap keeps the live mesher snappy.
const REFINE_PASSES: usize = 12;

/// A flat 2D triangulation handed back to JS: interleaved `points` (`x,y,...`)
/// and triangle `indices` (`i,j,k,...`).
#[wasm_bindgen]
pub struct Mesh2D {
    pts: Vec<f32>,
    tris: Vec<u32>,
}

#[wasm_bindgen]
impl Mesh2D {
    #[wasm_bindgen(getter)]
    pub fn points(&self) -> Vec<f32> {
        self.pts.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<u32> {
        self.tris.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn n_points(&self) -> usize {
        self.pts.len() / 2
    }
    #[wasm_bindgen(getter)]
    pub fn n_triangles(&self) -> usize {
        self.tris.len() / 3
    }
}

/// The mesher's coarse-to-fine sequence: one triangulation per refinement pass
/// (de-duplicated by vertex count), ending on the finished mesh.
#[wasm_bindgen]
pub struct MeshSteps {
    pts: Vec<Vec<f32>>,
    tris: Vec<Vec<u32>>,
}

#[wasm_bindgen]
impl MeshSteps {
    #[wasm_bindgen(getter)]
    pub fn n_steps(&self) -> usize {
        self.pts.len()
    }
    pub fn points(&self, i: usize) -> Vec<f32> {
        self.pts[i].clone()
    }
    pub fn indices(&self, i: usize) -> Vec<u32> {
        self.tris[i].clone()
    }
}

/// Squared distance from `p` to segment `a`-`b`.
fn pt_seg_d2(p: P2, a: P2, b: P2) -> f64 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let c1 = vx * wx + vy * wy;
    if c1 <= 0.0 {
        return wx * wx + wy * wy;
    }
    let c2 = vx * vx + vy * vy;
    if c2 <= c1 {
        let (dx, dy) = (p[0] - b[0], p[1] - b[1]);
        return dx * dx + dy * dy;
    }
    let t = c1 / c2;
    let (dx, dy) = (p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
    dx * dx + dy * dy
}

/// The graded target edge length at `p`: a gradient-limited field that grows
/// LINEARLY from `h_near` at the wordmark `glyphs` by slope `grade` per unit
/// distance, capped at `h_far`. Linear growth from the feature is the natural,
/// smooth mesh grading (no abrupt density ring).
fn graded(p: P2, glyphs: &[Vec<P2>], h_near: f64, h_far: f64, grade: f64) -> f64 {
    let mut best = f64::INFINITY;
    for lp in glyphs {
        let n = lp.len();
        for i in 0..n {
            let d2 = pt_seg_d2(p, lp[i], lp[(i + 1) % n]);
            if d2 < best {
                best = d2;
            }
        }
    }
    (h_near + grade * best.sqrt()).min(h_far)
}

/// Resample a closed contour so no edge is longer than `step`: subdivide each
/// segment, keeping the original vertices. Uniform boundary spacing that matches
/// the interior target kills the slivers that long constraint edges seed.
fn resample(lp: &[P2], step: f64) -> Vec<P2> {
    let n = lp.len();
    let mut out: Vec<P2> = Vec::new();
    for i in 0..n {
        let a = lp[i];
        let b = lp[(i + 1) % n];
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let sub = (len / step).ceil().max(1.0) as usize;
        for k in 0..sub {
            let t = k as f64 / sub as f64;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    out
}

/// The contour loops: the viewport rectangle (subdivided at `h_far`) as the outer
/// boundary, then each glyph outline as a hole -- the glyphs resampled at
/// `h_near` so the boundary spacing matches the fine interior there.
fn build_loops(
    width: f64,
    height: f64,
    outline: &[f64],
    loop_lens: &[u32],
    h_near: f64,
    h_far: f64,
) -> Vec<Vec<P2>> {
    let corners = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]];
    let mut rect: Vec<P2> = Vec::new();
    for k in 0..4 {
        let a = corners[k];
        let b = corners[(k + 1) % 4];
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let n = (len / h_far).round().max(1.0) as usize;
        for i in 0..n {
            let t = i as f64 / n as f64;
            rect.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    let mut loops = vec![rect];
    let mut o = 0usize;
    for &len in loop_lens {
        let len = len as usize;
        let mut lp = Vec::with_capacity(len);
        for k in 0..len {
            lp.push([outline[2 * (o + k)], outline[2 * (o + k) + 1]]);
        }
        o += len;
        loops.push(resample(&lp, h_near));
    }
    loops
}

fn pack_points(pts: &[P2]) -> Vec<f32> {
    pts.iter().flat_map(|p| [p[0] as f32, p[1] as f32]).collect()
}
fn pack_tris(tris: &[[usize; 3]]) -> Vec<u32> {
    tris.iter()
        .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect()
}

/// Triangulate the `width` x `height` viewport around the wordmark, returning the
/// finished mesh. `outline`/`loop_lens` are the glyph contours (the holes); the
/// edge length grades from `h_near` at the wordmark to `h_far` once `falloff`
/// away; `min_angle` is the Ruppert quality bound in degrees.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn triangulate(
    width: f64,
    height: f64,
    outline: &[f64],
    loop_lens: &[u32],
    h_near: f64,
    h_far: f64,
    grade: f64,
    min_angle: f64,
) -> Mesh2D {
    let loops = build_loops(width, height, outline, loop_lens, h_near, h_far);
    // Bake the (expensive) distance-graded field onto a quadtree once, then the
    // mesher's millions of size queries are O(depth) lookups.
    let field = QuadtreeField::from_fn(
        [0.0, 0.0], [width, height], h_near, 12,
        |p| graded(p, &loops[1..], h_near, h_far, grade),
    );
    let (pts, tris) =
        mesh_polygon(&loops, |p| field.eval(p), h_near, min_angle, CVT_ITERS, REFINE_PASSES);
    Mesh2D { pts: pack_points(&pts), tris: pack_tris(&tris) }
}

/// Like [`triangulate`], but also returns every coarse-to-fine intermediate
/// state (one per refinement pass, de-duplicated by vertex count), ending on the
/// finished mesh -- for animating the mesher building the page live.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn triangulate_steps(
    width: f64,
    height: f64,
    outline: &[f64],
    loop_lens: &[u32],
    h_near: f64,
    h_far: f64,
    grade: f64,
    min_angle: f64,
) -> MeshSteps {
    let loops = build_loops(width, height, outline, loop_lens, h_near, h_far);
    let field = QuadtreeField::from_fn(
        [0.0, 0.0], [width, height], h_near, 12,
        |p| graded(p, &loops[1..], h_near, h_far, grade),
    );
    let mut step_pts: Vec<Vec<f32>> = Vec::new();
    let mut step_tris: Vec<Vec<u32>> = Vec::new();
    let mut last_n = usize::MAX;
    let (pts, tris) = mesh_polygon_with(
        &loops,
        |p| field.eval(p),
        h_near,
        min_angle,
        CVT_ITERS,
        REFINE_PASSES,
        |all, tris| {
            if all.len() != last_n {
                last_n = all.len();
                step_pts.push(pack_points(all));
                step_tris.push(pack_tris(tris));
            }
        },
    );
    step_pts.push(pack_points(&pts));
    step_tris.push(pack_tris(&tris));
    MeshSteps { pts: step_pts, tris: step_tris }
}
