//! The count-driven volume entry point ([`mesh_budgeted`]) and the
//! surface-only export ([`surface_mesh`]).
//!
//! The volume engine itself is the restricted-Delaunay refinement core
//! ([`crate::refine`], reached via [`crate::conform::mesh_plc_with`]); this
//! module wraps it with the element-budget retune loop and hosts the
//! chart-based surface-only path (stages 1+2: graded feature-edge points,
//! per-patch 2D meshing in the plane / the analytic chart).

use crate::conform::{build_patches, MeshParams, Patch, SurfaceFace, SurfaceMesh, TetMesh};
use rapidmesh_geom::vec3::{V3, sub, scale, dot, cross, dist};
use crate::geomutil::in_loops;
use crate::domain::DomainTree;
use crate::surf2d::cvt_fill;
use crate::surfchart::build_chart;
use rapidmesh_csg::Tri;
use rapidmesh_exact::Point3;
use rapidmesh_geom::{RegionTag, SurfaceKind, TaggedPlc};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

/// Deterministic hashing: the mesher iterates these containers (boundary edges,
/// face owners, tilings), and the result must be reproducible run-to-run, so a
/// downstream pass (e.g. `optimize`) sees a fixed order. std's RandomState would
/// make the surface relaxation and the optimize sequence vary per run.
type DHasher = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DHasher>;
type DSet<T> = HashSet<T, DHasher>;

// All tuning constants are centralised in crate::constants.
use crate::constants::{
    SURFACE_OVERSAMPLE, SURF_LLOYD_ITERS,
};

/// Parameters `t in (0,1)` for graded points along edge `va->vb`: places points
/// at equal fractions of the graded integral `∫ ds / (OVERSAMPLE * h)`. For a
/// constant target this reduces to even `k/n` spacing (so uniform geometry keeps
/// its old, conformity-safe pattern); where `h` varies, points cluster where the
/// local size is fine. Symmetric in the endpoints regardless of grading.
fn graded_edge_fracs(va: V3, vb: V3, domain: &DomainTree) -> Vec<f64> {
    let len = dist(va, vb);
    if len <= 0.0 {
        return Vec::new();
    }
    let dir: V3 = std::array::from_fn(|k| (vb[k] - va[k]) / len);
    // Sample the inverse local spacing finely enough to resolve the grading.
    let samples = ((len / (SURFACE_OVERSAMPLE * domain.finest())).ceil() as usize * 4).clamp(16, 4096);
    let dl = len / samples as f64;
    let mut cum = vec![0.0f64; samples + 1];
    for i in 0..samples {
        let s = (i as f64 + 0.5) * dl;
        let p: V3 = std::array::from_fn(|k| va[k] + dir[k] * s);
        let spacing = (SURFACE_OVERSAMPLE * domain.h_at_surf(p)).max(len * 1e-3);
        cum[i + 1] = cum[i] + dl / spacing;
    }
    let total = cum[samples];
    let n = (total.round() as usize).max(1);
    let mut fracs = Vec::with_capacity(n.saturating_sub(1));
    let mut i = 0usize;
    for k in 1..n {
        let target = k as f64 / n as f64 * total;
        while i < samples && cum[i + 1] < target {
            i += 1;
        }
        let seg = (cum[i + 1] - cum[i]).max(1e-30);
        let arc = (i as f64 + (target - cum[i]) / seg) * dl;
        fracs.push((arc / len).clamp(0.0, 1.0));
    }
    fracs
}

fn tri_of(plc: &TaggedPlc, t: [u32; 3]) -> Tri {
    Tri::new(
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    )
}

/// Plane (a point, unit normal) of a patch, from its first facet.
fn patch_plane(plc: &TaggedPlc, patch: &Patch) -> (V3, V3) {
    let t = plc.triangles[patch.member_indices[0]];
    let (a, b, c) = (
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    );
    let mut n = cross(sub(b, a), sub(c, a));
    let l = dot(n, n).sqrt();
    if l > 0.0 {
        n = scale(n, 1.0 / l);
    }
    (a, n)
}

fn drop_axis(n: V3) -> usize {
    let a = n.map(f64::abs);
    if a[0] >= a[1] && a[0] >= a[2] {
        0
    } else if a[1] >= a[2] {
        1
    } else {
        2
    }
}

fn kept_axes(drop: usize) -> (usize, usize) {
    match drop {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

fn project2(p: V3, drop: usize) -> [f64; 2] {
    let (k1, k2) = kept_axes(drop);
    [p[k1], p[k2]]
}

/// Lifts a 2D point (the two kept axes) back onto the patch plane.
fn lift3(uv: [f64; 2], drop: usize, p0: V3, n: V3) -> V3 {
    let (k1, k2) = kept_axes(drop);
    let mut q = [0.0; 3];
    q[k1] = uv[0];
    q[k2] = uv[1];
    q[drop] = p0[drop] - (n[k1] * (uv[0] - p0[k1]) + n[k2] * (uv[1] - p0[k2])) / n[drop];
    q
}

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Boundary edges of a patch: corner pairs appearing once among its facets.
fn patch_boundary_edges(plc: &TaggedPlc, patch: &Patch) -> Vec<(usize, usize)> {
    let mut count: DMap<(usize, usize), usize> = DMap::default();
    for &fi in &patch.member_indices {
        let t = plc.triangles[fi];
        let c = [t[0] as usize, t[1] as usize, t[2] as usize];
        for e in 0..3 {
            *count.entry(sorted2(c[e], c[(e + 1) % 3])).or_insert(0) += 1;
        }
    }
    count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect()
}

/// True if the point `p` (a valid `Point3`, assumed on the patch plane) lies on
/// a member facet of the patch (exact `Tri::contains_coplanar`).
fn point_in_patch(plc: &TaggedPlc, patch: &Patch, p: &Point3) -> bool {
    patch.member_indices.iter().any(|&fi| {
        let tri = tri_of(plc, plc.triangles[fi]);
        let (ax, or) = tri.projection_axis();
        tri.contains_coplanar(p, ax, or)
    })
}

/// Mesh `plc` to an optional element budget, with the optional quality pass.
///
/// `optimize_passes`: `Some(n)` runs [`crate::optimize::optimize`] (whose size
/// targets mirror the params, so the quality pass respects the mesher's sizing)
/// for `n` passes after each remesh; `None` skips it.
///
/// `target_elements`: `Some(target)` retunes the GLOBAL size scale over a few
/// remeshes so the FINAL tet count (after optimize, which can shrink it ~25%)
/// lands within 6% of `target` -- the tet count scales as `scale^-3`, so each
/// step multiplies the scale by `(n/target)^(1/3)`. The relative refinement
/// (curvature + size points) keeps its shape throughout. `None` meshes once.
///
/// Returns the mesh and the (possibly budget-scaled) params it was built with.
/// This is the count-driven volume entry point; the surface analogue is the
/// `surf_target_count` cap inside [`surface_mesh`].
pub fn mesh_budgeted(
    plc: &TaggedPlc,
    params: &MeshParams,
    target_elements: Option<usize>,
    optimize_passes: Option<usize>,
) -> (TetMesh, MeshParams) {
    let mesh_once = |p: &MeshParams| -> TetMesh {
        // The volume backend is the restricted-Delaunay REFINEMENT core
        // (analytic carriers, protecting balls, manifold sweeps); the budget
        // loop and the optimize pass wrap it unchanged.
        let mut m = crate::conform::mesh_plc_with(plc, p);
        if let Some(passes) = optimize_passes {
            let opt = crate::optimize::OptimizeParams {
                passes,
                maxh: p.maxh,
                region_maxh: p.region_maxh.clone(),
                face_maxh: p.face_maxh.clone(),
                surface_maxh: p.surface_maxh.clone(),
            };
            crate::optimize::optimize(&mut m, &opt);
        }
        m
    };
    match target_elements {
        Some(target) if target > 0 => {
            let mut s = 1.0_f64;
            let mut out: Option<(TetMesh, MeshParams)> = None;
            for _ in 0..6 {
                let p = params.scaled(s);
                let m = mesh_once(&p);
                let n = m.tets.len().max(1);
                let rel = (n as f64 - target as f64).abs() / target as f64;
                out = Some((m, p));
                if rel < 0.06 {
                    break;
                }
                s *= (n as f64 / target as f64).powf(1.0 / 3.0);
            }
            out.expect("budget loop runs at least once")
        }
        _ => {
            let m = mesh_once(params);
            (m, params.clone())
        }
    }
}

/// Boundary edges of a curved smooth group: corner pairs appearing once across
/// all its member facets (interior facet seams appear twice).
fn group_boundary_edges(plc: &TaggedPlc, members: &[usize]) -> Vec<(usize, usize)> {
    let mut count: DMap<(usize, usize), usize> = DMap::default();
    for &fi in members {
        let t = plc.triangles[fi];
        let c = [t[0] as usize, t[1] as usize, t[2] as usize];
        for e in 0..3 {
            *count.entry(sorted2(c[e], c[(e + 1) % 3])).or_insert(0) += 1;
        }
    }
    let mut out: Vec<(usize, usize)> = count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect();
    out.sort_unstable();
    out
}

/// Surface-only meshing: the early-exit export path. Runs the hierarchy's
/// stage 1 (corners + graded feature-edge points) and stage 2 (2D Lloyd per
/// tile), triangulates each tile and lifts it to 3D, giving the conforming
/// boundary mesh WITHOUT the volume tetrahedralization. Shared edge points
/// (cached) keep the tile triangulations conforming across seams.
///
/// A tile is either a planar patch (relaxed in its own plane, the classic path)
/// or a curved SMOOTH GROUP: all facets of one analytic surface + region pair +
/// face tag, meshed in a distance-faithful chart ([`Chart`]) with interior
/// points placed by a curvature/volume-error sizing bias and lifted EXACTLY onto
/// the analytic surface. A closed group (no boundary loop) or one whose chart is
/// not a bijection (round-trip check fails) falls back to emitting its input
/// facets unchanged.
/// Builds the domain sizing octree with the per-entity overrides applied: per-face
/// `surf_maxh` -> per-facet volume target (`facet_surf`), and per-edge `edge_maxh`
/// -> point sources sampled along the brep edge chain (so the field stays fine
/// along a refined edge). Shared by the volume path (`mesh_refine`) and the
/// surface-only export (`surface_mesh`), so BOTH honor the same sizing knobs
/// (per-entity AND global caps, which `DomainTree::build` composes).
pub(crate) fn build_sizing_domain(
    plc: &TaggedPlc,
    params: &MeshParams,
    brep: &rapidmesh_brep::Brep,
) -> DomainTree {
    // Per-face `surf_maxh` -> per-facet volume target.
    let mut facet_surf = vec![f64::INFINITY; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        if let Some(&(_, h)) = params.surf_maxh.iter().find(|&&(i, _)| i as usize == fid) {
            for &ti in &f.facets {
                facet_surf[ti as usize] = facet_surf[ti as usize].min(h);
            }
        }
    }
    if params.edge_maxh.is_empty() {
        return DomainTree::build(plc, params, &facet_surf);
    }
    // Per-edge `edge_maxh` -> point sources along the brep edge chain. Only clones
    // the params when an edge override is actually present.
    let mut pa = params.clone();
    for (eid, e) in brep.edges.iter().enumerate() {
        let Some(&(_, h)) = params.edge_maxh.iter().find(|&&(i, _)| i as usize == eid) else {
            continue;
        };
        for w in e.chain.windows(2) {
            let n = ((dist(w[0], w[1]) / h).ceil() as usize).max(1);
            for k in 0..n {
                let t = k as f64 / n as f64;
                pa.size_points
                    .push((std::array::from_fn(|c| w[0][c] + t * (w[1][c] - w[0][c])), h));
            }
        }
        if let Some(&last) = e.chain.last() {
            pa.size_points.push((last, h));
        }
    }
    DomainTree::build(plc, &pa, &facet_surf)
}

pub fn surface_mesh(plc: &TaggedPlc, params: &MeshParams) -> SurfaceMesh {
    // Same per-entity-aware domain the volume path builds, so the surface export
    // honors `surf_maxh`/`edge_maxh` overrides (not just the global caps) -- the
    // patch target below reads `domain.h_at`.
    let brep = rapidmesh_brep::build::from_plc(plc);
    let domain = build_sizing_domain(plc, params, &brep);
    let patches = build_patches(plc);

    let mut diag = 0.0_f64;
    {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in &plc.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        diag = (0..3).map(|k| hi[k] - lo[k]).fold(diag, f64::max);
    }

    let is_curved = |sid: u32| !matches!(plc.surfaces[sid as usize], SurfaceKind::Plane);

    // Planar patches keep the per-patch plane path; curved facets regroup into
    // smooth groups keyed by (surface, region-lo, region-hi, face tag).
    type GKey = (u32, u32, u32, u32);
    let mut groups: DMap<GKey, Vec<usize>> = DMap::default();
    for i in 0..plc.triangles.len() {
        let sid = plc.surface_refs[i].0;
        if is_curved(sid) {
            let r = plc.region_tags[i];
            let key = (sid, r[0].0.min(r[1].0), r[0].0.max(r[1].0), plc.face_tags[i].0);
            groups.entry(key).or_default().push(i);
        }
    }
    let mut group_list: Vec<(GKey, Vec<usize>)> = groups.into_iter().collect();
    group_list.sort_by_key(|(_, m)| m.iter().copied().min());

    // Boundary edges per planar patch and per curved group (true feature edges).
    let planar: Vec<usize> = patches
        .iter()
        .enumerate()
        .filter(|(_, p)| !is_curved(p.surface))
        .map(|(pi, _)| pi)
        .collect();
    let pbe: Vec<Vec<(usize, usize)>> = planar.iter().map(|&pi| patch_boundary_edges(plc, &patches[pi])).collect();
    let gbe: Vec<Vec<(usize, usize)>> =
        group_list.iter().map(|(_, m)| group_boundary_edges(plc, m)).collect();

    // Global surface points: PLC corners, then graded points on every boundary
    // edge (shared across tiles via the cache), then per-tile interior points.
    let mut points: Vec<V3> = plc.vertices.clone();
    let mut edge_pts: DMap<(usize, usize), Vec<usize>> = DMap::default();
    for e in pbe.iter().flatten().chain(gbe.iter().flatten()) {
        if edge_pts.contains_key(e) {
            continue;
        }
        let (va, vb) = (plc.vertices[e.0], plc.vertices[e.1]);
        let idx: Vec<usize> = graded_edge_fracs(va, vb, &domain)
            .into_iter()
            .map(|f| {
                points.push(std::array::from_fn(|k| va[k] + f * (vb[k] - va[k])));
                points.len() - 1
            })
            .collect();
        edge_pts.insert(*e, idx);
    }

    let mut faces: Vec<SurfaceFace> = Vec::new();

    // Triangle BUDGET (a cap): split the global budget across the planar patches by
    // area (one uniform element size), the LAST patch absorbing the rounding
    // remainder. Each patch refines to min(its field, its share). 0 => uncapped.
    let patch_budget: Vec<usize> = if params.surf_target_count > 0 {
        let areas: Vec<f64> = planar
            .iter()
            .enumerate()
            .map(|(li, &pi)| {
                let (_, n) = patch_plane(plc, &patches[pi]);
                let drop = drop_axis(n);
                let mut s = 0.0;
                for &(a, b) in &pbe[li] {
                    let mut chain = vec![a];
                    chain.extend(edge_pts[&sorted2(a, b)].iter().copied());
                    chain.push(b);
                    for w in chain.windows(2) {
                        let p = project2(points[w[0]], drop);
                        let q = project2(points[w[1]], drop);
                        s += p[0] * q[1] - q[0] * p[1];
                    }
                }
                0.5 * s.abs()
            })
            .collect();
        let total: f64 = areas.iter().sum::<f64>().max(1e-30);
        let mut t = vec![0usize; planar.len()];
        let mut assigned = 0usize;
        for li in 0..planar.len() {
            t[li] = if li + 1 == planar.len() {
                params.surf_target_count.saturating_sub(assigned)
            } else {
                let n = ((params.surf_target_count as f64) * areas[li] / total).round() as usize;
                assigned += n;
                n
            };
        }
        t
    } else {
        vec![0; planar.len()]
    };

    // ---- planar patches: relax + triangulate in the patch plane -------------
    for (li, &pi) in planar.iter().enumerate() {
        let patch = &patches[pi];
        let (p0, n) = patch_plane(plc, patch);
        let drop = drop_axis(n);
        let mut loc2: Vec<[f64; 2]> = Vec::new();
        let mut gidx: Vec<usize> = Vec::new();
        let mut g2l: DMap<usize, usize> = DMap::default();
        let mut seen: DSet<usize> = DSet::default();
        let push_pt = |g: usize, uv: [f64; 2], loc2: &mut Vec<[f64; 2]>, gidx: &mut Vec<usize>, g2l: &mut DMap<usize, usize>| {
            let l = loc2.len();
            loc2.push(uv);
            gidx.push(g);
            g2l.insert(g, l);
        };
        for &(a, b) in &pbe[li] {
            for cv in [a, b] {
                if seen.insert(cv) {
                    push_pt(cv, project2(points[cv], drop), &mut loc2, &mut gidx, &mut g2l);
                }
            }
            for &gi in &edge_pts[&sorted2(a, b)] {
                if seen.insert(gi) {
                    push_pt(gi, project2(points[gi], drop), &mut loc2, &mut gidx, &mut g2l);
                }
            }
        }
        if loc2.len() < 3 {
            continue;
        }
        // The frozen boundary chains (corner -> graded edge points -> corner) are
        // the constraint segments: forcing them as mesh edges makes a non-convex
        // or holed plate triangulate to the face, not its convex hull (the 2D
        // analogue of the boundary-constrained volume).
        let mut bsegs: Vec<(usize, usize)> = Vec::new();
        for &(a, b) in &pbe[li] {
            let mut chain = vec![a];
            chain.extend(edge_pts[&sorted2(a, b)].iter().copied());
            chain.push(b);
            for w in chain.windows(2) {
                bsegs.push((g2l[&w[0]], g2l[&w[1]]));
            }
        }
        let nb = loc2.len();
        let inside2 =
            |uv: [f64; 2]| point_in_patch(plc, patch, &Point3::Explicit(lift3(uv, drop, p0, n)));
        let step = SURFACE_OVERSAMPLE * domain.finest();
        let target = |uv: [f64; 2]| {
            SURFACE_OVERSAMPLE * domain.h_at(lift3(uv, drop, p0, n)).min(params.surf_cap()).max(params.min_h_surf)
        };
        // Pure Ruppert from the boundary when refining (it grades to the field
        // itself); otherwise the Lloyd scatter. Mixing Lloyd points with refinement
        // seeds slivers, so refine starts from an empty interior.
        // ONE 2D path for every planar patch -- the volume stage AND the
        // surface product go through the shared core. surf_min_angle=0 (volume)
        // -> size-only refinement on a protected, edge-cleared boundary; >0
        // (surface product) -> full Ruppert. The boundary stays first in `all2`.
        let (all2, tris) = crate::surf2d::mesh_constrained(
            loc2[..nb].to_vec(), bsegs, target, inside2, step,
            params.surf_min_angle, patch_budget[li], SURF_LLOYD_ITERS, 60, |_, _| {},
        );
        for &uv in &all2[nb..] {
            points.push(lift3(uv, drop, p0, n));
            gidx.push(points.len() - 1);
        }
        for t in tris {
            faces.push(SurfaceFace {
                tri: [gidx[t[0]], gidx[t[1]], gidx[t[2]]],
                face_tag: patch.face_tag,
                regions: patch.regions,
                patch: pi as u32,
                surface: patch.surface,
            });
        }
    }

    // ---- curved smooth groups: chart-based curved Lloyd, with fallback ------
    for (gi, (key, members)) in group_list.iter().enumerate() {
        let (sid, r_lo, r_hi, tag) = *key;
        let kind = plc.surfaces[sid as usize].clone();
        let regions = [RegionTag(r_lo), RegionTag(r_hi)];
        let face_tag = rapidmesh_geom::FaceTag(tag);
        let patch_id = (patches.len() + gi) as u32;
        let emit_input = |faces: &mut Vec<SurfaceFace>| {
            for &fi in members {
                let t = plc.triangles[fi];
                faces.push(SurfaceFace {
                    tri: [t[0] as usize, t[1] as usize, t[2] as usize],
                    face_tag,
                    regions,
                    patch: patch_id,
                    surface: sid,
                });
            }
        };

        let bedges = &gbe[gi];
        if bedges.is_empty() {
            // Closed group (a full sphere): no chart covers it bijectively.
            emit_input(&mut faces);
            continue;
        }
        // Chart frame from the group's (on-surface) vertices.
        let mut gverts: Vec<usize> = members
            .iter()
            .flat_map(|&fi| plc.triangles[fi].iter().map(|&v| v as usize).collect::<Vec<_>>())
            .collect();
        gverts.sort_unstable();
        gverts.dedup();
        let chart = match build_chart(&kind, &gverts.iter().map(|&v| points[v]).collect::<Vec<_>>()) {
            Some(c) => c,
            None => {
                emit_input(&mut faces);
                continue;
            }
        };
        // Validate the chart is a bijection over the group. A boundary vertex on
        // the chord-approximated intersection curve sits slightly off the
        // surface, so compare the chart round-trip against the surface PROJECTION
        // (both land on the surface): they agree iff the chart did not fold (a
        // singular point, e.g. the sphere antipode in the group, fails).
        let tol = 1e-6 * diag.max(1.0);
        let bijective = gverts.iter().all(|&v| {
            let p = points[v];
            dist(chart.project(p), chart.to_xyz(chart.to_uv(p))) < tol
        });
        if !bijective {
            emit_input(&mut faces);
            continue;
        }

        // Boundary loop points in chart coordinates, plus corner-to-corner
        // segments for the inside test.
        let mut loc2: Vec<[f64; 2]> = Vec::new();
        let mut gidx: Vec<usize> = Vec::new();
        let mut seen: DSet<usize> = DSet::default();
        let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for &(a, b) in bedges {
            segs.push((chart.to_uv(points[a]), chart.to_uv(points[b])));
            for cv in [a, b] {
                if seen.insert(cv) {
                    loc2.push(chart.to_uv(points[cv]));
                    gidx.push(cv);
                }
            }
            for &gj in &edge_pts[&sorted2(a, b)] {
                if seen.insert(gj) {
                    loc2.push(chart.to_uv(points[gj]));
                    gidx.push(gj);
                }
            }
        }
        if loc2.len() < 3 {
            emit_input(&mut faces);
            continue;
        }
        let nb = loc2.len();
        let (mut lo2, mut hi2) = (loc2[0], loc2[0]);
        for &p in &loc2[..nb] {
            for k in 0..2 {
                lo2[k] = lo2[k].min(p[k]);
                hi2[k] = hi2[k].max(p[k]);
            }
        }
        // Curvature/volume-error bias: the finest curvature radius over the group
        // sets the grid step (so the scatter is fine enough to honor it); the
        // per-point target is the finer of the domain field and the curvature cap.
        let chord = (8.0 * params.tol_surf).sqrt();
        let hc_min = gverts
            .iter()
            .map(|&v| chart.curvature_radius(chart.to_uv(points[v])))
            .fold(f64::INFINITY, f64::min)
            * chord;
        let step = SURFACE_OVERSAMPLE * domain.finest().min(hc_min);
        let inside2 = |uv: [f64; 2]| in_loops(uv, &segs);
        let target = |uv: [f64; 2]| {
            let xyz = chart.to_xyz(uv);
            let hc = chart.curvature_radius(uv) * chord;
            SURFACE_OVERSAMPLE * domain.h_at(xyz).min(hc).min(params.surf_cap()).max(params.min_h_surf)
        };
        for uv in cvt_fill(&loc2[..nb], lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2, params.density_weighted) {
            points.push(chart.to_xyz(uv));
            loc2.push(uv);
            gidx.push(points.len() - 1);
        }
        // `delaunay2` triangulates the convex hull; the curved group's boundary
        // (a chord-approximated intersection curve) is not exactly convex in the
        // chart, so keep only the triangles whose centroid is inside the region.
        for t in crate::surf2d::delaunay2(&loc2) {
            let c = [
                (loc2[t[0]][0] + loc2[t[1]][0] + loc2[t[2]][0]) / 3.0,
                (loc2[t[0]][1] + loc2[t[1]][1] + loc2[t[2]][1]) / 3.0,
            ];
            if !in_loops(c, &segs) {
                continue;
            }
            faces.push(SurfaceFace {
                tri: [gidx[t[0]], gidx[t[1]], gidx[t[2]]],
                face_tag,
                regions,
                patch: patch_id,
                surface: sid,
            });
        }
    }

    SurfaceMesh {
        points,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{extrude_spline_profile, icosphere, solid_box, NurbsCurve, Scene};

    #[test]
    fn surface_mesh_box_is_closed_manifold() {
        // The surface-only export of a closed box is a closed manifold surface:
        // every edge is shared by exactly two triangles, and it covers all six
        // faces (well over a dozen triangles at this size).
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let sm = surface_mesh(&plc, &MeshParams { maxh: 0.8, ..Default::default() });
        assert!(sm.faces.len() > 12, "box surface should be tessellated, got {}", sm.faces.len());
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for f in &sm.faces {
            for e in 0..3 {
                let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2), "closed manifold: every edge in exactly 2 faces");
    }

    #[test]
    fn curved_surface_points_lie_on_sphere() {
        // Two overlapping spheres: the curved boundary groups are meshed in the
        // analytic chart and lifted onto the sphere, so every vertex of a Sphere
        // face sits EXACTLY on its sphere (radius), and the result is a closed
        // 2-manifold (every edge shared by two faces).
        let mut scene = Scene::new();
        scene.add_solid(icosphere([0.0, 0.0, 0.0], 1.0, 2));
        scene.add_solid(icosphere([1.2, 0.0, 0.0], 1.0, 2));
        let plc = scene.assemble();
        let n_plc = plc.vertices.len();
        let sm = surface_mesh(&plc, &MeshParams { maxh: 0.5, ..Default::default() });

        // Curved Lloyd added interior points; those the chart placed lie EXACTLY
        // on the analytic sphere. Boundary points sit on the chord-approximated
        // intersection curve (shared with the other sphere), off the sphere by at
        // most the facet sagitta, so the max deviation stays small. Verify both.
        let mut curved_faces = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if let SurfaceKind::Sphere { center, radius } = sm.surfaces[f.surface as usize] {
                curved_faces += 1;
                for &v in &f.tri {
                    let p = sm.points[v];
                    let d = ((p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2))
                    .sqrt();
                    let dev = (d - radius).abs();
                    max_dev = max_dev.max(dev);
                    if v >= n_plc && dev < 1e-9 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved_faces > 0, "expected curved faces");
        assert!(exact_on > 0, "curved Lloyd should place interior points exactly on the sphere");
        assert!(max_dev < 0.05, "no vertex grossly off the sphere, max_dev {max_dev}");

        // Per-region closure: the boundary of each region is a closed 2-manifold
        // (every edge shared by exactly two of that region's faces). Edges on the
        // triple curve where three regions meet are manifold within each region
        // but carry three faces overall, which a global 2-manifold test rejects.
        let mut regions: Vec<u32> = sm.faces.iter().flat_map(|f| [f.regions[0].0, f.regions[1].0]).collect();
        regions.sort_unstable();
        regions.dedup();
        for r in regions.into_iter().filter(|&r| r != 0) {
            let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
            for f in sm.faces.iter().filter(|f| f.regions[0].0 == r || f.regions[1].0 == r) {
                for e in 0..3 {
                    let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            let bad = edges.values().filter(|&&c| c != 2).count();
            assert_eq!(bad, 0, "region {r} boundary not closed: {bad} edges");
        }
    }

    #[test]
    fn extruded_spline_surface_is_on_the_analytic_surface() {
        // A semicircle profile extruded into a half-cylinder (D-prism). The
        // curved wall is one Extruded surface; its chart is the developable
        // (arc length x height) isometric chart. Interior points the curved
        // Lloyd places land EXACTLY on the cylinder (radial distance == r).
        let r = 1.0;
        let w = 0.5_f64.sqrt();
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![[r, 0.0], [r, r], [0.0, r], [-r, r], [-r, 0.0]],
            vec![1.0, w, 1.0, w, 1.0],
        );
        let solid = extrude_spline_profile(
            profile,
            24,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        );
        let mut scene = Scene::new();
        scene.add_solid(solid);
        let plc = scene.assemble();
        let n_plc = plc.vertices.len();
        let sm = surface_mesh(&plc, &MeshParams { maxh: 0.4, ..Default::default() });

        let mut curved = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if matches!(sm.surfaces[f.surface as usize], SurfaceKind::Extruded { .. }) {
                curved += 1;
                for &vtx in &f.tri {
                    let p = sm.points[vtx];
                    let rad = (p[0] * p[0] + p[1] * p[1]).sqrt();
                    let dev = (rad - r).abs();
                    max_dev = max_dev.max(dev);
                    if vtx >= n_plc && dev < 1e-7 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved > 0, "expected extruded curved faces");
        assert!(exact_on > 0, "curved Lloyd should place interior points on the cylinder");
        assert!(max_dev < 0.02, "no curved vertex grossly off radius, max_dev {max_dev}");

        // Per-region closure (single solid: region 1 boundary closed).
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for f in &sm.faces {
            for e in 0..3 {
                let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let bad = edges.values().filter(|&&c| c != 2).count();
        assert_eq!(bad, 0, "closed manifold, {bad} non-manifold edges");
    }

}
