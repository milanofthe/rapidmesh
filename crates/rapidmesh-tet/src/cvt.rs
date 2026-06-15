//! CVT (centroidal Voronoi) tetrahedral meshing of a tagged PLC.
//!
//! Replaces the constrained-Delaunay + Steiner boundary recovery of the old
//! pipeline. The exact CSG arrangement (the `TaggedPlc`) is untouched; this
//! stage fills it with a variational (Lloyd-relaxed) tetrahedralization.
//!
//! Unified exact kernel: every mesh point is a [`Site`] = a `Point3` pinned to a
//! carrier (vertex / edge line / face plane / volume). Surface points stay
//! EXACTLY on their carrier (`Lnc`/`Pac`), so the boundary is conforming and
//! region tagging is exact (`orient3d == 0`); volume points are explicit f64
//! (the fast predicate path).
//!
//! Strict density-driven hierarchy, each level fixed before the next: corners ->
//! feature edges (1D, uniform = the 1D CVT optimum) -> faces (2D Lloyd in
//! `surf2d`) -> volume (3D Lloyd). Conformity comes from seeding the surface
//! FINER than the volume (oversampling: the restricted Delaunay then recovers
//! the boundary as tet faces); the volume is seeded as a graded BCC lattice from
//! the central [`DomainTree`], which also assigns each tet's region by its
//! centroid (an exact per-region ray-cast). Out-of-domain volume moves are
//! rejected (kept).

use crate::conform::{build_patches, quality_stats, MeshParams, Patch, SurfaceFace, SurfaceMesh, TetMesh};
use crate::delaunay::DelaunayBuilder;
use crate::domain::DomainTree;
use crate::seed::SizingField;
use crate::site::{Carrier, Site};
use crate::spatial::Octree;
use crate::surf2d::cvt_fill;
use crate::surfchart::build_chart;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::Point3;
use rapidmesh_geom::{RegionTag, SurfaceKind, TaggedPlc};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

type V3 = [f64; 3];

/// Deterministic hashing: the mesher iterates these containers (boundary edges,
/// face owners, tilings), and the result must be reproducible run-to-run, so a
/// downstream pass (e.g. `optimize`) sees a fixed order. std's RandomState would
/// make the surface relaxation and the optimize sequence vary per run.
type DHasher = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DHasher>;
type DSet<T> = HashSet<T, DHasher>;

/// The four vertex-index triples spanning a tet's faces.
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

/// Volume (3D) Lloyd relaxation passes (the cap; the loop stops earlier when it
/// converges, see `LLOYD_CONVERGE_FRAC`).
const LLOYD_ITERS: usize = 8;
/// Lloyd convergence: stop once the largest site move in a pass falls below this
/// fraction of the finest spacing. Each saved pass is a saved Delaunay rebuild
/// (the dominant cost), and late passes that barely move sites add little.
const LLOYD_CONVERGE_FRAC: f64 = 0.02;
/// Surface (2D) Lloyd passes (a planar grid scatter needs little relaxation).
const SURF_LLOYD_ITERS: usize = 4;
/// Bounding-box subdivisions for the default spacing when no `maxh` is given.
const DEFAULT_SUBDIV: f64 = 8.0;
/// Per-triangle bounding-box pad for the inside test, fraction of the diagonal.
const BOX_PAD_FRAC: f64 = 1e-6;
/// Minimum separation of a seeded/moved site from any other, fraction of spacing.
const SEPARATION_FRAC: f64 = 0.45;
/// The surface (edges + faces) is seeded finer than the volume by this factor so
/// the restricted Delaunay recovers the boundary without explicit recovery.
/// 0.5 is the conformity threshold here; coarser (0.7) reintroduces straddlers.
pub(crate) const SURFACE_OVERSAMPLE: f64 = 0.5;
/// Chord/volume-error sizing bias for curved surfaces: a facet edge of length
/// `h` on a surface of principal radius `R` deviates from the true surface by a
/// sagitta `eps ~ h^2/(8R)`. Bounding the relative sagitta `eps/R <= this` caps
/// the faceting (and thus the enclosed-volume error) independent of `maxh`:
/// `h_curv = R * sqrt(8 * frac)`. 0.02 gives ~16 facets around a full circle.
pub(crate) const SURF_CHORD_FRAC: f64 = 0.02;
/// A face is coplanar with a patch if all its vertices are within this fraction
/// of the scene diagonal of the patch plane (f64 surface points sit on tilted
/// planes only to ~1e-15; the precise assignment is the exact containment test).
const COPLANAR_EPS_FRAC: f64 = 1e-9;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
fn tet_det(p: [V3; 4]) -> f64 {
    dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])))
}
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
        let spacing = (SURFACE_OVERSAMPLE * domain.h_at(p)).max(len * 1e-3);
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

fn centroid4(p: [V3; 4]) -> V3 {
    std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]))
}

fn centroid3(p: [V3; 3]) -> V3 {
    std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0)
}

fn tri_of(plc: &TaggedPlc, t: [u32; 3]) -> Tri {
    Tri::new(
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    )
}

/// The OUTER boundary of the meshable domain: PLC facets with background
/// (region 0) on one side. Internal interfaces (both sides nonzero) are
/// excluded, so a parity ray-cast against this soup answers "inside the meshable
/// union" correctly for nested regions and voids.
fn domain_soup(plc: &TaggedPlc) -> Vec<Tri> {
    let outer: Vec<Tri> = plc
        .triangles
        .iter()
        .zip(&plc.region_tags)
        .filter(|(_, rt)| rt[0].0 == 0 || rt[1].0 == 0)
        .map(|(t, _)| tri_of(plc, *t))
        .collect();
    if outer.is_empty() {
        plc.triangles.iter().map(|t| tri_of(plc, *t)).collect()
    } else {
        outer
    }
}

fn regions_of(plc: &TaggedPlc) -> Vec<u32> {
    let mut rs: Vec<u32> = Vec::new();
    for pair in &plc.region_tags {
        for r in pair {
            if r.0 != 0 && !rs.contains(&r.0) {
                rs.push(r.0);
            }
        }
    }
    rs.sort_unstable();
    rs
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

/// The patch a tet face tiles: all three vertices within `eps` of the patch
/// plane (tolerant, since f64 surface points sit on a tilted plane only to
/// ~1e-15) AND the exact barycenter on a member facet (`contains_coplanar`,
/// the precise assignment). `None` if on no patch (a volume vertex matches none).
fn patch_of_face(
    plc: &TaggedPlc,
    patches: &[Patch],
    sites: &[Site],
    f: [usize; 3],
    eps: f64,
) -> Option<usize> {
    let c: V3 = std::array::from_fn(|k| {
        (sites[f[0]].pos()[k] + sites[f[1]].pos()[k] + sites[f[2]].pos()[k]) / 3.0
    });
    let bary = Point3::Explicit(c);
    for (pi, p) in patches.iter().enumerate() {
        let (p0, n) = patch_plane(plc, p);
        let coplanar = f.iter().all(|&v| dot(sub(sites[v].pos(), p0), n).abs() < eps);
        if coplanar && point_in_patch(plc, p, &bary) {
            return Some(pi);
        }
    }
    None
}

fn positions(sites: &[Site]) -> Vec<V3> {
    sites.iter().map(|s| s.pos()).collect()
}

/// A bounded, chart-meshable curved surface group for the VOLUME path: all
/// facets of one analytic curved surface + region pair + face tag whose chart
/// is a bijection (an open/bounded patch). These get on-surface, curvature-
/// graded seeding and restricted-Delaunay face recovery; closed or wrapping
/// curved groups are absent here and stay on the per-facet planar path.
struct ChartGroup {
    surface: u32,
    kind: SurfaceKind,
    regions: [RegionTag; 2],
    face_tag: rapidmesh_geom::FaceTag,
    members: Vec<usize>,
    boundary: Vec<(usize, usize)>,
    chart: Box<dyn crate::surfchart::SurfaceChart>,
    /// Smallest principal radius of curvature over the group (drives the finest
    /// seeding step); computed over ALL vertices, so the high-curvature interior
    /// (an airfoil nose) is captured, not just the boundary loop.
    min_radius: f64,
}

/// The bounded curved smooth-groups of the PLC suitable for chart-based volume
/// seeding (the same grouping `surface_mesh` uses, minus the closed/wrapping
/// ones, which fail the bijectivity round-trip and fall back).
fn chart_groups(plc: &TaggedPlc, diag: f64) -> Vec<ChartGroup> {
    type GKey = (u32, u32, u32, u32);
    let is_curved = |sid: u32| !matches!(plc.surfaces[sid as usize], SurfaceKind::Plane);
    let mut groups: DMap<GKey, Vec<usize>> = DMap::default();
    for i in 0..plc.triangles.len() {
        let sid = plc.surface_refs[i].0;
        if is_curved(sid) {
            let r = plc.region_tags[i];
            let key = (sid, r[0].0.min(r[1].0), r[0].0.max(r[1].0), plc.face_tags[i].0);
            groups.entry(key).or_default().push(i);
        }
    }
    let mut list: Vec<(GKey, Vec<usize>)> = groups.into_iter().collect();
    list.sort_by_key(|(_, m)| m.iter().copied().min());
    let tol = 1e-6 * diag.max(1.0);
    let mut out = Vec::new();
    for ((sid, r_lo, r_hi, tag), members) in list {
        let boundary = group_boundary_edges(plc, &members);
        if boundary.is_empty() {
            continue; // closed group: no chart covers it bijectively
        }
        let kind = plc.surfaces[sid as usize].clone();
        let mut gverts: Vec<usize> = members
            .iter()
            .flat_map(|&fi| plc.triangles[fi].iter().map(|&v| v as usize).collect::<Vec<_>>())
            .collect();
        gverts.sort_unstable();
        gverts.dedup();
        let pts: Vec<V3> = gverts.iter().map(|&v| plc.vertices[v]).collect();
        let chart = match build_chart(&kind, &pts) {
            Some(c) => c,
            None => continue,
        };
        let bijective = gverts.iter().all(|&v| {
            let p = plc.vertices[v];
            dist(chart.project(p), chart.to_xyz(chart.to_uv(p))) < tol
        });
        if !bijective {
            continue; // folding chart: stay on the faceted path
        }
        // Per-vertex round-trip passes even for a chart that WRAPS (a full
        // cylinder/torus barrel): each vertex maps consistently, but the chart
        // tears at the seam. Detect it: a boundary edge that straddles the seam
        // maps to a huge chart segment (~2*pi*R) while its 3D length is one
        // facet. If any boundary edge's chart length far exceeds its 3D length,
        // the chart is not a global bijection over the group -> fall back.
        let seam = boundary.iter().any(|&(a, b)| {
            let (ca, cb) = (chart.to_uv(plc.vertices[a]), chart.to_uv(plc.vertices[b]));
            let chart_len = ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2)).sqrt();
            let real_len = dist(plc.vertices[a], plc.vertices[b]);
            chart_len > 4.0 * real_len + tol
        });
        if seam {
            continue; // wrapping chart: stay on the faceted path
        }
        let min_radius = gverts
            .iter()
            .map(|&v| chart.curvature_radius(chart.to_uv(plc.vertices[v])))
            .fold(f64::INFINITY, f64::min);
        out.push(ChartGroup {
            surface: sid,
            kind,
            regions: [RegionTag(r_lo), RegionTag(r_hi)],
            face_tag: rapidmesh_geom::FaceTag(tag),
            members,
            boundary,
            chart,
            min_radius,
        });
    }
    out
}

/// Adds graded edge points for boundary edge `e` (once, cached) as `on_edge`
/// sites; shared by the planar patches and the curved groups.
fn ensure_edge(
    plc: &TaggedPlc,
    domain: &DomainTree,
    sites: &mut Vec<Site>,
    edge_pts: &mut DMap<(usize, usize), Vec<usize>>,
    e: (usize, usize),
) {
    if edge_pts.contains_key(&e) {
        return;
    }
    let (va, vb) = (plc.vertices[e.0], plc.vertices[e.1]);
    let idx: Vec<usize> = graded_edge_fracs(va, vb, domain)
        .into_iter()
        .map(|f| {
            sites.push(Site::on_edge(va, vb, f));
            sites.len() - 1
        })
        .collect();
    edge_pts.insert(e, idx);
}

/// Meshes a tagged PLC into a region-tagged tet mesh by hierarchical CVT.
pub fn mesh(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    let t_start = std::time::Instant::now();

    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max);

    let regions = regions_of(plc);
    let field = SizingField::new(params);
    let cap = field.finest_cap(&regions);
    let spacing = if cap.is_finite() && cap > 0.0 { cap } else { diag / DEFAULT_SUBDIV };

    let soup = domain_soup(plc);
    let boxes = TriBoxes::build(&soup, BOX_PAD_FRAC * diag.max(1.0));
    let inside = |p: V3| point_inside_solid(&Point3::Explicit(p), p, &soup, &boxes, (lo, hi));

    // The central domain octree: refined to the local sizing field h(x). It
    // drives graded volume seeding, the Lloyd crowding neighbor search (its
    // per-leaf site buckets, re-filled each pass), and tet region classification,
    // so the interior coarsens away from the fine, fixed boundary.
    let t_domain = std::time::Instant::now();
    let mut domain = DomainTree::build(plc, params);
    rmlog::stage("mesh.domain", t_domain.elapsed().as_secs_f64());

    let patches = build_patches(plc);
    // Bounded curved groups meshed on their analytic surface (the curved Lloyd,
    // now in the volume path); their facets are excluded from the per-facet
    // planar path. Closed/wrapping curved surfaces produce no chart group and
    // stay on the faceted path, so every planar/faceted fixture is unchanged.
    let cgroups = chart_groups(plc, diag);
    let chart_facets: DSet<usize> = cgroups.iter().flat_map(|g| g.members.iter().copied()).collect();
    let is_chart_patch = |p: &Patch| p.member_indices.iter().any(|fi| chart_facets.contains(fi));
    // Site separation (surface clearance, volume crowding) is LOCAL: a fraction
    // of `domain.h_at(p)`, computed at each seed/move below. A global value from
    // the coarse bulk size would reject seeds near fine curved features and
    // orphan their surface nodes; grading lives in the octree's density.

    // ---- stage 1: corners + feature edges (1D), fixed --------------------
    // Edge points are placed by GRADED arc length: the local spacing along the
    // edge is `SURFACE_OVERSAMPLE * h(p)` from the domain octree, so an edge
    // bordering a finely sized face gets denser points there and coarsens away.
    // Shared edges are seeded once (cached), so both adjacent patches agree.
    let t_surf = std::time::Instant::now();
    let nb = plc.vertices.len();
    let mut sites: Vec<Site> = plc.vertices.iter().map(|&v| Site::vertex(v)).collect();
    let pbe: Vec<Vec<(usize, usize)>> = patches.iter().map(|p| patch_boundary_edges(plc, p)).collect();
    let mut edge_pts: DMap<(usize, usize), Vec<usize>> = DMap::default();
    // Non-chart patches keep their per-patch boundary edges; chart groups use
    // their GROUP boundary (no points on internal curved facet seams).
    for (pi, patch) in patches.iter().enumerate() {
        if is_chart_patch(patch) {
            continue;
        }
        for &e in &pbe[pi] {
            ensure_edge(plc, &domain, &mut sites, &mut edge_pts, e);
        }
    }
    for g in &cgroups {
        for &e in &g.boundary {
            ensure_edge(plc, &domain, &mut sites, &mut edge_pts, e);
        }
    }

    // ---- stage 2: per-patch 2D Lloyd (faces), edges fixed ----------------
    // Patches are independent, so they fill in parallel. `cvt_fill` keeps each
    // patch's interior points clear of its boundary (corners + edge points),
    // which is the only separation needed: adjacent patches meet only along
    // shared edges, whose points both sides already keep clear of.
    use rayon::prelude::*;
    let face_sites: Vec<Site> = patches
        .par_iter()
        .enumerate()
        .filter(|(_, patch)| !is_chart_patch(patch))
        .flat_map(|(pi, patch)| {
            let (p0, n) = patch_plane(plc, patch);
            let drop = drop_axis(n);
            let mut bnd: Vec<[f64; 2]> = Vec::new();
            let mut seen: DSet<usize> = DSet::default();
            for &(a, b) in &pbe[pi] {
                for cv in [a, b] {
                    if seen.insert(cv) {
                        bnd.push(project2(plc.vertices[cv], drop));
                    }
                }
                for &i in &edge_pts[&sorted2(a, b)] {
                    bnd.push(project2(sites[i].pos(), drop));
                }
            }
            if bnd.len() < 3 {
                return Vec::new();
            }
            let (mut lo2, mut hi2) = (bnd[0], bnd[0]);
            for &p in &bnd {
                for k in 0..2 {
                    lo2[k] = lo2[k].min(p[k]);
                    hi2[k] = hi2[k].max(p[k]);
                }
            }
            let inside2 =
                |uv: [f64; 2]| point_in_patch(plc, patch, &Point3::Explicit(lift3(uv, drop, p0, n)));
            // Graded fill: grid step is the global finest surface spacing, the
            // local target is `SURFACE_OVERSAMPLE * h(lift(uv))` from the octree.
            let step = SURFACE_OVERSAMPLE * domain.finest();
            let target = |uv: [f64; 2]| SURFACE_OVERSAMPLE * domain.h_at(lift3(uv, drop, p0, n));
            cvt_fill(&bnd, lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2)
                .into_iter()
                .map(|uv| Site::on_plane(p0, n, lift3(uv, drop, p0, n)))
                .collect::<Vec<_>>()
        })
        .collect();
    sites.extend(face_sites);

    // Curved smooth-groups: relax interior points in the analytic chart and seed
    // them as `on_surface` sites (curvature-graded, EXACTLY on the surface), the
    // curved Lloyd now feeding the volume Delaunay. Boundary is corners + shared
    // edge points (same sites both sides). Oversampled like the planar faces so
    // the restricted Delaunay recovers the curved boundary as tet faces.
    for g in &cgroups {
        let mut loc2: Vec<[f64; 2]> = Vec::new();
        let mut seen: DSet<usize> = DSet::default();
        let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for &(a, b) in &g.boundary {
            segs.push((g.chart.to_uv(plc.vertices[a]), g.chart.to_uv(plc.vertices[b])));
            for cv in [a, b] {
                if seen.insert(cv) {
                    loc2.push(g.chart.to_uv(plc.vertices[cv]));
                }
            }
            for &gj in &edge_pts[&sorted2(a, b)] {
                if seen.insert(gj) {
                    loc2.push(g.chart.to_uv(sites[gj].pos()));
                }
            }
        }
        if loc2.len() < 3 {
            continue;
        }
        let nb2 = loc2.len();
        let (mut lo2, mut hi2) = (loc2[0], loc2[0]);
        for &p in &loc2[..nb2] {
            for k in 0..2 {
                lo2[k] = lo2[k].min(p[k]);
                hi2[k] = hi2[k].max(p[k]);
            }
        }
        // Curvature/volume-error bias drives the surface density: fine where the
        // surface is tightly curved (an airfoil nose), coarse where it is flat,
        // so the boundary is captured with FEW triangles, the nodes EXACTLY on
        // the analytic surface. `min_radius` (over the whole group, not just the
        // boundary loop) sets the finest grid step so the high-curvature interior
        // is resolved. The chart is isometric, so the target is a true length.
        let chord = (8.0 * SURF_CHORD_FRAC).sqrt();
        let step = SURFACE_OVERSAMPLE * domain.finest().min(g.min_radius * chord);
        let inside2 = |uv: [f64; 2]| in_loops(uv, &segs);
        let target = |uv: [f64; 2]| {
            let hc = g.chart.curvature_radius(uv) * chord;
            SURFACE_OVERSAMPLE * domain.h_at(g.chart.to_xyz(uv)).min(hc)
        };
        for uv in cvt_fill(&loc2[..nb2], lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2) {
            sites.push(Site::on_surface(g.kind.clone(), g.chart.to_xyz(uv)));
        }
    }

    let n_surf = sites.len();
    rmlog::stage("mesh.surface", t_surf.elapsed().as_secs_f64());

    // ---- stage 3: interior volume points, graded by the domain octree -----
    // One seed per interior leaf center: dense near the fine boundary, coarse in
    // the bulk (the octree is refined to h(x)). Keep each seed clear of the
    // fixed surface by the LOCAL spacing, and respect `max_points`.
    let t_seed = std::time::Instant::now();
    {
        let pos = positions(&sites);
        let surf_tree = Octree::build(&pos);
        let budget = params.max_points.saturating_sub(sites.len());
        let mut added = 0usize;
        for p in domain.seed_points() {
            if added >= budget {
                break;
            }
            if !inside(p) {
                continue;
            }
            // LOCAL clearance: the surface is graded (fine at a curved nose,
            // coarse elsewhere), so a global `sep` from the coarse bulk size
            // would reject every seed near a fine feature and orphan its surface
            // nodes. Scale the clearance by the local sizing field.
            let lsep = SEPARATION_FRAC * domain.h_at(p);
            let near = surf_tree.nearest(p).map(|j| dist(p, pos[j as usize]) < lsep).unwrap_or(false);
            if !near {
                sites.push(Site::free(p));
                added += 1;
            }
        }
    }
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // ---- 3D Lloyd on the volume sites (surface fixed) --------------------
    // The surface is fixed now. Build its Delaunay ONCE; each pass clones it and
    // inserts only the moving volume points (incremental: no surface re-insert).
    // Its octree (for mirror-in) is built once too.
    let surf_pos: Vec<V3> = sites[..n_surf].iter().map(|s| s.pos()).collect();
    let surf_tree = Octree::build(&surf_pos);
    let order_surf = crate::spatial::morton_order(&surf_pos);
    let mut surf_db = DelaunayBuilder::enclosing(lo, hi);
    for &si in &order_surf {
        surf_db.insert(surf_pos[si]);
    }
    // Full Delaunay = surface clone + Morton-ordered volume inserts; returns tets
    // remapped from insertion indices back to site indices.
    let build = |vol_pos: &[V3]| -> Vec<[usize; 4]> {
        let order_vol = crate::spatial::morton_order(vol_pos);
        let mut db = surf_db.clone();
        for &vi in &order_vol {
            db.insert(vol_pos[vi]);
        }
        db.tets()
            .into_iter()
            .map(|t| {
                std::array::from_fn(|j| {
                    let p = t[j];
                    if p < n_surf {
                        order_surf[p]
                    } else {
                        n_surf + order_vol[p - n_surf]
                    }
                })
            })
            .collect()
    };

    let t_lloyd = std::time::Instant::now();
    let mut lloyd_passes = 0usize;
    for _ in 0..LLOYD_ITERS {
        if !sites.iter().any(|s| s.is_volume()) {
            break;
        }
        lloyd_passes += 1;
        let pos = positions(&sites);
        let vol_pos: Vec<V3> = sites[n_surf..].iter().map(|s| s.pos()).collect();
        let tets = build(&vol_pos);
        let mut num = vec![[0.0f64; 3]; sites.len()];
        let mut den = vec![0.0f64; sites.len()];
        for t in &tets {
            let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
            let w = tet_det(p).abs();
            let c = centroid4(p);
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * c[k];
                }
                den[i] += w;
            }
        }
        // Crowding neighbor search runs on the central domain octree: refill its
        // per-leaf site buckets with this pass's positions, then query a radius.
        domain.rebucket(&pos);
        let mut max_move = 0.0f64;
        for i in 0..sites.len() {
            if !sites[i].is_volume() || den[i] == 0.0 {
                continue;
            }
            let mut tgt: V3 = std::array::from_fn(|k| num[i][k] / den[i]);
            // Out of the domain: mirror back in across the nearest boundary face
            // (the planar carrier of the closest surface site). If that does not
            // land inside, keep the point where it was.
            if !inside(tgt) {
                let mirrored = surf_tree.nearest(tgt).and_then(|j| {
                    if let Carrier::Plane { p0, n } = sites[j as usize].carrier {
                        let r = reflect(tgt, p0, n);
                        inside(r).then_some(r)
                    } else {
                        None
                    }
                });
                match mirrored {
                    Some(r) => tgt = r,
                    None => continue,
                }
            }
            let crowded = domain
                .neighbors(tgt, SEPARATION_FRAC * domain.h_at(tgt))
                .into_iter()
                .any(|j| j as usize != i);
            if !crowded {
                sites[i].move_to(tgt);
                max_move = max_move.max(dist(pos[i], sites[i].pos()));
            }
        }
        // Converged: the largest move this pass is a tiny fraction of the finest
        // spacing, so further passes (and their Delaunay rebuilds) buy little.
        if max_move < LLOYD_CONVERGE_FRAC * spacing {
            break;
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());
    rmlog::stat("mesh.lloyd_passes", lloyd_passes as f64);

    // ---- final triangulation, tilings, region classification --------------
    let vol_pos: Vec<V3> = sites[n_surf..].iter().map(|s| s.pos()).collect();
    let t_build = std::time::Instant::now();
    let all_tets = build(&vol_pos);
    rmlog::stage("mesh.build_final", t_build.elapsed().as_secs_f64());
    let t_classify = std::time::Instant::now();
    let pts = positions(&sites);
    let mut face_owners: DMap<[usize; 3], Vec<u32>> = DMap::default();
    for (ti, t) in all_tets.iter().enumerate() {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            face_owners.entry(f).or_default().push(ti as u32);
        }
    }
    // A face tiles a patch only if all three vertices are surface sites.
    let t_tile = std::time::Instant::now();
    let coplanar_eps = COPLANAR_EPS_FRAC * diag.max(1.0);
    let mut tilings: Vec<Vec<[usize; 3]>> = vec![Vec::new(); patches.len()];
    for key in face_owners.keys() {
        if key.iter().any(|&v| v >= n_surf) {
            continue;
        }
        if let Some(pi) = patch_of_face(plc, &patches, &sites, *key, coplanar_eps) {
            tilings[pi].push(*key);
        }
    }
    rmlog::stage("mesh.tilings", t_tile.elapsed().as_secs_f64());
    let t_region = std::time::Instant::now();
    // Classify each tet by its centroid's region in the domain octree: a cached
    // lookup deep inside a region, an exact per-region ray-cast on the boundary
    // leaves. This is robust where the surface tilings are incomplete (a tet
    // face spanning two facets of a curved or concave boundary is untagged), so
    // it does not leak across the gap the way a face-crossing flood-fill does,
    // and it resolves the right region directly for nested/multi-material
    // domains without walking the dual graph.
    let region: Vec<u32> = all_tets
        .par_iter()
        .map(|t| {
            let c = centroid4([pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]]);
            domain.region_at(c)
        })
        .collect();
    let mut kept: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for (t, &r) in all_tets.iter().zip(&region) {
        if r != 0 {
            kept.push(*t);
            tet_regions.push(RegionTag(r));
        }
    }
    rmlog::stage("mesh.region", t_region.elapsed().as_secs_f64());
    rmlog::stage("mesh.classify", t_classify.elapsed().as_secs_f64());

    let mut faces: Vec<SurfaceFace> = Vec::new();
    for (pi, patch) in patches.iter().enumerate() {
        for &f in &tilings[pi] {
            faces.push(SurfaceFace {
                tri: f,
                face_tag: patch.face_tag,
                regions: patch.regions,
                patch: pi as u32,
                surface: patch.surface,
            });
        }
    }

    // Curved-group boundary faces: restricted-Delaunay extraction. A tet face
    // with all-surface vertices that the planar tilings did not claim, that
    // separates two distinct regions (or a region from outside), and whose
    // centroid lies on a chart group's analytic surface, is that group's
    // boundary face. These ARE tet faces (conformity holds) where coplanar
    // tiling cannot reach a smoothly curved surface.
    if !cgroups.is_empty() {
        let claimed: DSet<[usize; 3]> = tilings.iter().flatten().copied().collect();
        for (key, owners) in &face_owners {
            if key.iter().any(|&v| v >= n_surf) || claimed.contains(key) {
                continue;
            }
            let mut rs: Vec<u32> = owners.iter().map(|&ti| region[ti as usize]).collect();
            if owners.len() == 1 {
                rs.push(0);
            }
            rs.sort_unstable();
            rs.dedup();
            if rs.len() < 2 {
                continue; // interior face (same region both sides)
            }
            let tri = [pts[key[0]], pts[key[1]], pts[key[2]]];
            let c = centroid3(tri);
            let max_edge = (0..3).map(|k| dist(tri[k], tri[(k + 1) % 3])).fold(0.0, f64::max);
            for g in &cgroups {
                let mut grs = vec![g.regions[0].0, g.regions[1].0];
                grs.sort_unstable();
                grs.dedup();
                if grs == rs && dist(c, g.chart.project(c)) < 0.3 * max_edge {
                    faces.push(SurfaceFace {
                        tri: *key,
                        face_tag: g.face_tag,
                        regions: g.regions,
                        patch: u32::MAX,
                        surface: g.surface,
                    });
                    break;
                }
            }
        }
    }

    // Per-point local target size (the graded sizing field), so the optimizer
    // coarsens to the LOCAL size, not one region-uniform floor that would erase
    // curvature-fine detail.
    let point_size: Vec<f64> = pts.iter().map(|&p| domain.h_at(p)).collect();
    let mesh = TetMesh {
        points: pts,
        tets: kept,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points: nb,
        point_size,
    };

    let q = quality_stats(&mesh);
    rmlog::stat("mesh.points", mesh.points.len() as f64);
    rmlog::stat("mesh.tets", mesh.tets.len() as f64);
    rmlog::stat("mesh.surface_points", n_surf as f64);
    rmlog::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    rmlog::stage("mesh.total", t_start.elapsed().as_secs_f64());
    mesh
}

/// Reflection of `p` across the plane (`p0`, unit `n`).
fn reflect(p: V3, p0: V3, n: V3) -> V3 {
    sub(p, scale(n, 2.0 * dot(sub(p, p0), n)))
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

/// Even-odd point-in-region test for a chart point against the group's boundary
/// loops (corner-to-corner segments in chart coordinates). Handles holes (a
/// sphere with several caps removed) via the parity rule.
fn in_loops(uv: [f64; 2], segs: &[([f64; 2], [f64; 2])]) -> bool {
    let mut c = false;
    for &(a, b) in segs {
        if (a[1] > uv[1]) != (b[1] > uv[1]) {
            let xint = a[0] + (uv[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if uv[0] < xint {
                c = !c;
            }
        }
    }
    c
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
pub fn surface_mesh(plc: &TaggedPlc, params: &MeshParams) -> SurfaceMesh {
    let domain = DomainTree::build(plc, params);
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

    // ---- planar patches: relax + triangulate in the patch plane -------------
    for (li, &pi) in planar.iter().enumerate() {
        let patch = &patches[pi];
        let (p0, n) = patch_plane(plc, patch);
        let drop = drop_axis(n);
        let mut loc2: Vec<[f64; 2]> = Vec::new();
        let mut gidx: Vec<usize> = Vec::new();
        let mut seen: DSet<usize> = DSet::default();
        for &(a, b) in &pbe[li] {
            for cv in [a, b] {
                if seen.insert(cv) {
                    loc2.push(project2(points[cv], drop));
                    gidx.push(cv);
                }
            }
            for &gi in &edge_pts[&sorted2(a, b)] {
                if seen.insert(gi) {
                    loc2.push(project2(points[gi], drop));
                    gidx.push(gi);
                }
            }
        }
        if loc2.len() < 3 {
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
        let inside2 =
            |uv: [f64; 2]| point_in_patch(plc, patch, &Point3::Explicit(lift3(uv, drop, p0, n)));
        let step = SURFACE_OVERSAMPLE * domain.finest();
        let target = |uv: [f64; 2]| SURFACE_OVERSAMPLE * domain.h_at(lift3(uv, drop, p0, n));
        for uv in cvt_fill(&loc2[..nb], lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2) {
            points.push(lift3(uv, drop, p0, n));
            loc2.push(uv);
            gidx.push(points.len() - 1);
        }
        for t in crate::surf2d::delaunay2(&loc2) {
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
        let chord = (8.0 * SURF_CHORD_FRAC).sqrt();
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
            SURFACE_OVERSAMPLE * domain.h_at(xyz).min(hc)
        };
        for uv in cvt_fill(&loc2[..nb], lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2) {
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
    use num_rational::BigRational;
    use num_traits::Zero;
    use rapidmesh_geom::{extrude_spline_profile, icosphere, solid_box, NurbsCurve, Scene};
    use rapidmesh_testutil::rat;

    fn region_volume6(m: &TetMesh, r: u32) -> BigRational {
        let mut acc = BigRational::zero();
        for (t, tr) in m.tets.iter().zip(&m.tet_regions) {
            if tr.0 != r {
                continue;
            }
            let p: Vec<[BigRational; 3]> = t.iter().map(|&i| m.points[i].map(rat)).collect();
            let row: Vec<[BigRational; 3]> =
                (0..3).map(|k| std::array::from_fn(|j| &p[k][j] - &p[3][j])).collect();
            let det = &row[0][0] * (&row[1][1] * &row[2][2] - &row[1][2] * &row[2][1])
                - &row[0][1] * (&row[1][0] * &row[2][2] - &row[1][2] * &row[2][0])
                + &row[0][2] * (&row[1][0] * &row[2][1] - &row[1][1] * &row[2][0]);
            acc += det;
        }
        acc
    }

    fn watertight(m: &TetMesh) -> bool {
        let mut faces: HashMap<[usize; 3], usize> = HashMap::new();
        for t in &m.tets {
            for fv in &TET_FACES {
                let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
                f.sort_unstable();
                *faces.entry(f).or_default() += 1;
            }
        }
        faces.values().all(|&c| c <= 2)
    }

    #[test]
    fn box_meshes_exactly() {
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams { maxh: 0.8, ..Default::default() });
        assert_eq!(region_volume6(&mesh, r.0), rat(144.0), "exact box volume");
        assert!(watertight(&mesh), "watertight boundary");
    }

    #[test]
    fn unsized_box_is_valid() {
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams::default());
        assert_eq!(region_volume6(&mesh, r.0), rat(6.0), "exact unit cube");
        assert!(watertight(&mesh));
    }

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

    #[test]
    fn extruded_spline_surface_in_volume_is_on_the_analytic_surface() {
        // The curved Lloyd now feeds the VOLUME path: a half-cylinder (semicircle
        // profile extruded) gets curved boundary faces whose interior vertices
        // land EXACTLY on the cylinder, recovered as real tet faces.
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
        let reg = scene.add_solid(solid);
        let plc = scene.assemble();
        let n_plc = plc.vertices.len();
        let mesh = mesh(&plc, &MeshParams { maxh: 0.4, ..Default::default() });

        let mut curved = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &mesh.faces {
            if matches!(mesh.surfaces[f.surface as usize], SurfaceKind::Extruded { .. }) {
                curved += 1;
                for &v in &f.tri {
                    let p = mesh.points[v];
                    let dev = ((p[0] * p[0] + p[1] * p[1]).sqrt() - r).abs();
                    max_dev = max_dev.max(dev);
                    if v >= n_plc && dev < 1e-7 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved > 0, "expected curved boundary faces in the volume mesh");
        assert!(exact_on > 0, "curved volume boundary vertices should be exactly on the cylinder");
        assert!(max_dev < 0.02, "no curved vertex grossly off radius, max_dev {max_dev}");
        // every output face is a tet face (conformity) and the boundary is closed.
        let mut tet_faces: HashMap<[usize; 3], usize> = HashMap::new();
        for t in &mesh.tets {
            for fv in &TET_FACES {
                let mut k = [t[fv[0]], t[fv[1]], t[fv[2]]];
                k.sort_unstable();
                *tet_faces.entry(k).or_default() += 1;
            }
        }
        for f in &mesh.faces {
            let mut k = f.tri;
            k.sort_unstable();
            assert!(tet_faces.contains_key(&k), "surface face is not a tet face");
        }
        assert!(tet_faces.values().all(|&c| c <= 2), "non-manifold tet face");
        assert!(region_volume6(&mesh, reg.0) > rat(0.0), "nonempty region");
    }

    #[test]
    fn nested_two_region_meshes_exactly() {
        let mut scene = Scene::new();
        let air = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
        let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams { maxh: 1.0, ..Default::default() });
        assert_eq!(region_volume6(&mesh, diel.0), rat(24.0), "diel volume");
        assert_eq!(region_volume6(&mesh, air.0), rat(360.0), "air volume");
        assert!(watertight(&mesh));
    }
}
