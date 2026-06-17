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
use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind, TaggedPlc};
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

/// The planar patch a tet face lies on: all three vertices within `eps` of the
/// patch plane AND the barycenter on a member facet. Used ONLY as the face-tag
/// FALLBACK for faces with no interior surface vertex to carry the tile (a coarse
/// box face of only corners) and to recover embedded sheets; it labels, it does
/// not decide manifoldness (the region interface already did), so it cannot
/// double-emit. `None` if on no planar patch.
fn patch_of_face(plc: &TaggedPlc, patches: &[Patch], pts: &[V3], f: [usize; 3], eps: f64) -> Option<usize> {
    let c: V3 = std::array::from_fn(|k| (pts[f[0]][k] + pts[f[1]][k] + pts[f[2]][k]) / 3.0);
    let bary = Point3::Explicit(c);
    for (pi, p) in patches.iter().enumerate() {
        let (p0, n) = patch_plane(plc, p);
        let coplanar = f.iter().all(|&v| dot(sub(pts[v], p0), n).abs() < eps);
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
    face_tag: FaceTag,
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
    for ((sid, _r_lo, _r_hi, tag), members) in list {
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
            face_tag: FaceTag(tag),
            members,
            boundary,
            chart,
            min_radius,
        });
    }
    out
}


/// NURBS-native edge curve: the analytic profile of an `Extruded` surface,
/// mapped to 3D at a fixed extrusion height `z`. Points distributed on THIS (via
/// `curve::distribute`) come from the exact profile curvature, independent of the
/// input facet tessellation -- the principled way to consume the geometry, vs
/// re-sampling a faceted polyline. The trailing edge falls out as the profile's
/// own parameter endpoints.
struct ExtrudedEdgeCurve {
    profile: std::sync::Arc<rapidmesh_geom::nurbs::NurbsCurve>,
    base: V3,
    u: V3,
    v: V3,
    a: V3,
    z: f64,
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ExtrudedEdgeCurve {
    /// Builds the edge curve over the profile parameter range `[t0, t1]` (the part
    /// the chain covers) at extrusion height `z`. Distributing the FULL profile for
    /// every chain would duplicate points where the profile is split into chains.
    fn new(kind: &SurfaceKind, z: f64, t0: f64, t1: f64) -> Option<ExtrudedEdgeCurve> {
        let SurfaceKind::Extruded { profile, base, udir, vdir, axis } = kind else {
            return None;
        };
        let norm = |a: V3| {
            let l = dot(a, a).sqrt();
            if l > 0.0 { scale(a, 1.0 / l) } else { a }
        };
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        if !(hi > lo) {
            return None;
        }
        let n = 256usize;
        let (mut ts, mut ss) = (vec![lo], vec![0.0f64]);
        let mut prev = lo;
        let mut acc = 0.0;
        for i in 1..=n {
            let t = lo + (hi - lo) * i as f64 / n as f64;
            acc += profile.arc_length(prev, t, 2);
            ts.push(t);
            ss.push(acc);
            prev = t;
        }
        Some(ExtrudedEdgeCurve {
            profile: profile.clone(),
            base: *base,
            u: norm(*udir),
            v: norm(*vdir),
            a: norm(*axis),
            z,
            ts,
            ss,
        })
    }
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self.ss.partition_point(|&x| x < s).clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
    fn at3(&self, t: f64) -> V3 {
        let c = self.profile.eval(t);
        let mut p = self.base;
        for k in 0..3 {
            p[k] += self.a[k] * self.z + self.u[k] * c[0] + self.v[k] * c[1];
        }
        p
    }
}

impl crate::curve::Curve for ExtrudedEdgeCurve {
    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }
    fn point_at(&self, s: f64) -> V3 {
        self.at3(self.s_to_t(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        let k = self.profile.curvature(self.s_to_t(s));
        if k > 1e-12 { 1.0 / k } else { f64::INFINITY }
    }
}

/// Chains the PLC's feature edges (the boundary edges of all surface patches and
/// curved groups) into ordered curves, split at corners (a junction, degree != 2,
/// or a turn sharper than 45 deg). A corner-less loop (a circular rim) is anchored
/// at its lowest-index vertex. Each chain is one curve for the 1D distributor; the
/// chains tile every feature edge once, so adjacent patches share edge points.
fn feature_edge_chains(
    plc: &TaggedPlc,
    pbe: &[Vec<(usize, usize)>],
    cgroups: &[ChartGroup],
    skip_edges: &DSet<(usize, usize)>,
) -> Vec<Vec<usize>> {
    use std::collections::{HashMap, HashSet};
    const CORNER_COS: f64 = 0.707;
    let unit = |a: V3| {
        let l = dot(a, a).sqrt();
        if l > 0.0 { scale(a, 1.0 / l) } else { a }
    };
    // `skip_edges` are not chained: edges interior to a chart group (its vertical
    // facet seams -- they would give every outline vertex degree 3 and split the
    // smooth profile into 1-edge chains, killing the curvature grading) and edges
    // of a faceted closed-curved surface (kept as input tessellation).
    let mut eset: HashSet<(usize, usize)> = HashSet::new();
    for p in pbe {
        for &(a, b) in p {
            if !skip_edges.contains(&sorted2(a, b)) {
                eset.insert(sorted2(a, b));
            }
        }
    }
    for g in cgroups {
        for &(a, b) in &g.boundary {
            if !skip_edges.contains(&sorted2(a, b)) {
                eset.insert(sorted2(a, b));
            }
        }
    }
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in &eset {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    if adj.is_empty() {
        return Vec::new();
    }
    let is_corner = |n: usize, adj: &HashMap<usize, Vec<usize>>| -> bool {
        let ns = &adj[&n];
        if ns.len() != 2 {
            return true;
        }
        let d0 = unit(sub(plc.vertices[n], plc.vertices[ns[0]]));
        let d1 = unit(sub(plc.vertices[ns[1]], plc.vertices[n]));
        dot(d0, d1) < CORNER_COS
    };
    let mut corners: Vec<usize> = adj.keys().copied().filter(|&n| is_corner(n, &adj)).collect();
    corners.sort_unstable();
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut done: HashSet<(usize, usize)> = HashSet::new();
    let walk = |c0: usize, start: usize, adj: &HashMap<usize, Vec<usize>>, done: &mut HashSet<(usize, usize)>| -> Vec<usize> {
        let mut chain = vec![c0];
        let (mut prev, mut cur) = (c0, start);
        loop {
            chain.push(cur);
            done.insert((prev, cur));
            done.insert((cur, prev));
            if is_corner(cur, adj) || cur == c0 {
                break;
            }
            let ns = &adj[&cur];
            let nxt = if ns[0] == prev { ns[1] } else { ns[0] };
            prev = cur;
            cur = nxt;
            if chain.len() > adj.len() + 2 {
                break;
            }
        }
        chain
    };
    for &c0 in &corners {
        for &start in &adj[&c0].clone() {
            if !done.contains(&(c0, start)) {
                out.push(walk(c0, start, &adj, &mut done));
            }
        }
    }
    let mut keys: Vec<usize> = adj.keys().copied().collect();
    keys.sort_unstable();
    for &a in &keys {
        for &b in &adj[&a].clone() {
            if !done.contains(&(a, b)) {
                let mut ch = walk(a, b, &adj, &mut done);
                if ch.last() != Some(&a) {
                    ch.push(a);
                }
                out.push(ch);
            }
        }
    }
    out
}

/// Meshes a tagged PLC into a region-tagged tet mesh by the bottom-up sizing
/// hierarchy: feature edges (`curve.rs`) seed the surface size field, the surface
/// points seed the volume size field, each driving a weighted Lloyd. The size
/// field is gradient-limited (`sizefield.rs`) -- smooth and narrow, no heuristic
/// grading. Region classification and the restricted-Delaunay boundary are kept.
pub fn mesh(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    use rayon::prelude::*;
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
    let grad = if params.grading > 0.0 { params.grading } else { 0.5 };

    // ---- stages 1+2: surface points from the B-rep -----------------------
    // Build the boundary representation and let it drive the surface: distribute
    // on every analytic edge curve (shared across faces), then a randomized
    // Poisson-disk fill of each face on its analytic surface (planar faces keep an
    // exact on-plane carrier). One mechanism for planar, curved and closed faces;
    // the volume stages below are unchanged.
    let t_surf = std::time::Instant::now();
    let brep = rapidmesh_brep::build::from_plc(plc);
    let ss = crate::brep_mesh::surface_sites(&brep, plc, params, &domain);
    let mut sites = ss.sites;
    let mut point_tile = ss.point_tile;
    let tiles = ss.tiles;
    let plc_points = ss.plc_points;

    let n_surf = sites.len();
    rmlog::stat("mesh.brep_faces", brep.faces.len() as f64);
    rmlog::stat("mesh.brep_edges", brep.edges.len() as f64);
    rmlog::stage("mesh.surface", t_surf.elapsed().as_secs_f64());



    // ---- stage 3 input: the VOLUME size field = the SIZE FIELD `H` --------
    // The volume is meshed at the geometry's size field `H` (`domain.h_at`: caps +
    // curvature, graded), DECOUPLED from the surface. The surface was meshed FINER
    // (at `OVERSAMPLE * H`), so the surface is finer than the volume -- the
    // restricted-Delaunay boundary recovers cleanly and volume tets cannot straddle
    // the exact PLC boundary (conformity). Sampling `H` at the surface points and
    // gradient-limiting reproduces the field for the volume stages.
    let surf_pos: Vec<V3> = sites[..n_surf].iter().map(|s| s.pos()).collect();
    let vol_sources: Vec<(V3, f64)> =
        (0..n_surf).map(|i| (surf_pos[i], domain.h_at(surf_pos[i]).max(1e-9))).collect();
    let vol_field = crate::sizefield::SizeField::new(vol_sources, grad, params.maxh);
    // A per-region size cap: `region_maxh` is a region-WIDE size that the surface
    // field (which grows coarse from the boundary inward) does not enforce in the
    // interior, so cap the volume size by the region of the query point.
    let region_cap = |r: u32| -> f64 {
        params
            .region_maxh
            .iter()
            .find(|(rr, _)| *rr == r)
            .map(|&(_, h)| h)
            .unwrap_or(params.maxh)
            .min(params.maxh)
    };
    // ---- stage 3: interior volume points, graded by the domain octree -----
    // One seed per interior leaf center: dense near the fine boundary, coarse in
    // the bulk (the octree is refined to h(x)). Keep each seed clear of the
    // fixed surface by the LOCAL spacing, and respect `max_points`.
    let t_seed = std::time::Instant::now();
    {
        let pos = positions(&sites);
        let surf_tree = Octree::build(&pos);
        let budget_pts = params.max_points.saturating_sub(sites.len());
        // Local size at a point: the volume field (grown from the surface points),
        // capped by the region size.
        let hloc = |p: V3| vol_field.at(p).min(region_cap(domain.region_at(p))).max(1e-9);
        // Dart budget = the density integral integral(1/h^3) dV over a coarse
        // sample of the inside, so the count matches the graded target (a sharp
        // feature does not inflate it). Randomized darts then pack uniformly at
        // that density -- the size-field BCC alone is ~1.5x coarser than the
        // (oversampled) surface, leaving the interior visibly under-seeded.
        let span: V3 = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let ns = 24usize;
        let cellvol = (span[0] * span[1] * span[2]).max(0.0) / (ns * ns * ns) as f64;
        let mut density = 0.0f64;
        for i in 0..ns {
            for j in 0..ns {
                for k in 0..ns {
                    let p: V3 = [
                        lo[0] + (i as f64 + 0.5) / ns as f64 * span[0],
                        lo[1] + (j as f64 + 0.5) / ns as f64 * span[1],
                        lo[2] + (k as f64 + 0.5) / ns as f64 * span[2],
                    ];
                    if inside(p) {
                        let h = hloc(p);
                        density += cellvol / (h * h * h);
                    }
                }
            }
        }
        let budget = ((density * 6.0) as usize).min(budget_pts);
        // Spatial hash for interior separation; cell >= the largest separation
        // radius (0.65*maxh) so a +-1 cell query is exact.
        let cell = (0.65 * params.maxh).max(1e-9);
        let ckey = |p: V3| {
            ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)
        };
        let mut igrid: DMap<(i64, i64, i64), Vec<V3>> = DMap::default();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut added = 0usize;
        let attempts = (budget * 6).max(64);
        for _ in 0..attempts {
            if added >= budget_pts {
                break;
            }
            let p: V3 = [lo[0] + next() * span[0], lo[1] + next() * span[1], lo[2] + next() * span[2]];
            if !inside(p) {
                continue;
            }
            let h = hloc(p);
            // Surface clearance = the FULL local size: a volume seed within one
            // element of the surface makes a boundary tet with a volume vertex,
            // which the restricted-Delaunay boundary cannot emit -> a hole (breaks
            // watertightness / exact volumes). Non-negotiable.
            if surf_tree.nearest(p).map(|j| dist(p, pos[j as usize]) < h).unwrap_or(false) {
                continue;
            }
            // Interior separation.
            let r = 0.65 * h;
            let (kx, ky, kz) = ckey(p);
            let r2 = r * r;
            let mut clear = true;
            'scan: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = igrid.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &q in v {
                                if dot(sub(p, q), sub(p, q)) < r2 {
                                    clear = false;
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
            }
            if clear {
                igrid.entry((kx, ky, kz)).or_default().push(p);
                sites.push(Site::free(p));
                added += 1;
            }
        }
        rmlog::stat("mesh.vol_seeds", added as f64);
    }
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // ---- volume (3D) CVT Lloyd: relax INTERIOR sites only -----------------
    // Bottom-up: the surface points were already relaxed in their own dimension
    // (1D edges, 2D per-face Lloyd in `surface_sites`), so here they are FIXED --
    // they ARE the boundary the volume conforms to. The volume sites move to their
    // density-weighted tet centroid; the full Delaunay is rebuilt each pass. The
    // octree of surface points is the mirror-in fallback for an out-of-domain
    // volume centroid.
    let surf_tree = Octree::build(&sites[..n_surf].iter().map(|s| s.pos()).collect::<Vec<_>>());
    let build_full = |all_pos: &[V3]| -> Vec<[usize; 4]> {
        let order = crate::spatial::morton_order(all_pos);
        let mut db = DelaunayBuilder::enclosing(lo, hi);
        // `orig[k]` = original index of the k-th SUCCESSFULLY inserted point;
        // near-duplicates are skipped (robust to near-tangent intersections).
        let mut orig: Vec<usize> = Vec::with_capacity(order.len());
        for &i in &order {
            if db.try_insert(all_pos[i]).is_some() {
                orig.push(i);
            }
        }
        db.tets().into_iter().map(|t| std::array::from_fn(|j| orig[t[j]])).collect()
    };

    let t_lloyd = std::time::Instant::now();
    let mut lloyd_passes = 0usize;
    for _ in 0..LLOYD_ITERS {
        lloyd_passes += 1;
        let pos = positions(&sites);
        let tets = build_full(&pos);
        let mut num = vec![[0.0f64; 3]; sites.len()];
        let mut den = vec![0.0f64; sites.len()];
        for t in &tets {
            let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
            let c = centroid4(p);
            // DENSITY-WEIGHTED CVT (adaptive): weight by volume * rho, rho = 1/h^3,
            // pulling sites toward finer regions (gated; uniform h => rho const).
            let w = if params.density_weighted {
                let h = vol_field.at(c).min(region_cap(domain.region_at(c))).max(1e-9);
                tet_det(p).abs() / (h * h * h)
            } else {
                tet_det(p).abs()
            };
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * c[k];
                }
                den[i] += w;
            }
        }
        domain.rebucket(&pos);
        let mut max_move = 0.0f64;
        for i in 0..sites.len() {
            // ONLY interior (volume) sites relax; surface sites (corners, edges,
            // faces) are fixed -- they were meshed bottom-up in their own dimension.
            if !sites[i].is_volume() || den[i] == 0.0 {
                continue;
            }
            let mut tgt: V3 = std::array::from_fn(|k| num[i][k] / den[i]);
            // A volume centroid outside the domain: mirror it back in across the
            // nearest carrier plane.
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
            let h = vol_field.at(tgt).min(region_cap(domain.region_at(tgt)));
            let sep = SEPARATION_FRAC * h;
            let crowded = domain.neighbors(tgt, sep).into_iter().any(|j| j as usize != i);
            if !crowded {
                sites[i].move_to(tgt);
                max_move = max_move.max(dist(pos[i], sites[i].pos()));
            }
        }
        if max_move < LLOYD_CONVERGE_FRAC * spacing {
            break;
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());
    rmlog::stat("mesh.lloyd_passes", lloyd_passes as f64);

    // ---- final triangulation, tilings, region classification --------------
    let t_build = std::time::Instant::now();
    let all_tets = build_full(&positions(&sites));
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

    // ---- boundary surface = the RESTRICTED DELAUNAY (region interface) ------
    // A tet face is a boundary/interface face IFF its two incident tets lie in
    // different regions (outside = region 0). This is principled (no coplanar
    // epsilon, no proximity tolerance) and manifold BY CONSTRUCTION: every tet
    // face appears once, shared by its two tets, so an edge carries exactly two
    // faces per region -- no double-tagging at multi-surface junctions. The
    // region pair comes from the two tets (exact); the surface/tag is the tile
    // of an interior surface vertex (`point_tile`), the principled carrier-based
    // label (shared corner/edge vertices defer to the interior one).
    let t_faces = std::time::Instant::now();
    let eps = 1e-9 * diag.max(1.0);
    let mut faces: Vec<SurfaceFace> = Vec::new();
    for (key, owners) in &face_owners {
        if key.iter().any(|&v| v >= n_surf) {
            continue; // a boundary face lives on the seeded surface
        }
        let (ra, rb) = match owners.as_slice() {
            [a, b] => (region[*a as usize], region[*b as usize]),
            [a] => (region[*a as usize], 0),
            _ => continue,
        };
        // Output tag of the face: from an interior surface vertex's B-rep face
        // (`point_tile` -> `tiles`), else the geometric planar patch's OWN
        // surface/tag (coarse all-corner faces). These are different id spaces,
        // so resolve to `(surface, face_tag)` here rather than indexing `tiles`
        // with a patch id (which would be out of range).
        let from_vertex = key.iter().map(|&v| point_tile[v]).find(|&t| t != u32::MAX);

        if ra != rb {
            // Region interface (boundary or material interface): manifold by
            // construction. Region pair from the tets (exact); label from the tile.
            let (surface, face_tag, patch) = if let Some(t) = from_vertex {
                let (s, ft) = tiles[t as usize];
                (s, ft, t)
            } else if let Some(pi) = patch_of_face(plc, &patches, &pts, *key, eps) {
                (patches[pi].surface, patches[pi].face_tag, u32::MAX)
            } else {
                (0, FaceTag(0), u32::MAX)
            };
            faces.push(SurfaceFace {
                tri: *key,
                face_tag,
                regions: [RegionTag(ra.min(rb)), RegionTag(ra.max(rb))],
                patch,
                surface,
            });
        } else if ra != 0 {
            // Embedded sheet: a tagged internal surface with the SAME region on
            // both sides (not a region interface) -- recover it where the face
            // sits on a planar sheet patch (equal regions, tagged).
            if let Some(pi) = patch_of_face(plc, &patches, &pts, *key, eps) {
                let p = &patches[pi];
                if p.regions[0] == p.regions[1] && p.face_tag.0 != 0 {
                    faces.push(SurfaceFace {
                        tri: *key,
                        face_tag: p.face_tag,
                        regions: p.regions,
                        patch: pi as u32,
                        surface: p.surface,
                    });
                }
            }
        }
    }
    rmlog::stage("mesh.faces", t_faces.elapsed().as_secs_f64());

    // Per-point local target size (the graded sizing field), so the optimizer
    // coarsens to the LOCAL size, not one region-uniform floor that would erase
    // curvature-fine detail.
    let point_size: Vec<f64> =
        pts.iter().map(|&p| vol_field.at(p).min(region_cap(domain.region_at(p)))).collect();
    let mesh = TetMesh {
        points: pts,
        tets: kept,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points,
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
        for uv in cvt_fill(&loc2[..nb], lo2, hi2, step, target, SURF_LLOYD_ITERS, inside2, params.density_weighted) {
            points.push(lift3(uv, drop, p0, n));
            loc2.push(uv);
            gidx.push(points.len() - 1);
        }
        for t in crate::surf2d::triangulate_constrained(&loc2, &bsegs, inside2) {
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
        let chord = (8.0 * params.surface_deflection).sqrt();
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

/// The frozen Stage-2 surface in the exact form Stage 3 consumes: the surface
/// mesh plus, per vertex and per facet, the geometric CARRIER. The volume stage
/// inserts each vertex exactly ([`Site::exact`]) and recovers each curved facet
/// onto its carrier ([`crate::cdt3`]); planar facets carry their plane (so a
/// Steiner, if ever needed, stays on it) and need no recovery.
pub struct FrozenSurface {
    /// Surface vertex positions (parallel to [FrozenSurface::vert_carrier]).
    pub points: Vec<V3>,
    /// The carrier each vertex lives on (plane / analytic surface / corner).
    pub vert_carrier: Vec<Carrier>,
    /// The surface triangulation, tagged (region pair, face tag, surface, patch).
    pub faces: Vec<SurfaceFace>,
    /// The carrier each facet lies on: a plane (`Plane`) for planar patches,
    /// the analytic surface (`Surface`) for curved groups.
    pub face_carrier: Vec<Carrier>,
    /// Analytic surfaces, parallel to [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid, parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
}

/// Builds the [`FrozenSurface`] by attaching carriers to the Stage-2 surface
/// mesh. A planar facet's carrier is the plane through its (exact, on-plane)
/// vertices: for an axis-aligned face the normal is exact, so `Site::exact`
/// keeps every vertex bit-exactly on it. A curved facet carries its analytic
/// surface. A vertex on one plane gets that plane; a vertex where several
/// surfaces meet (a corner/feature edge) stays an explicit (`Vertex`) carrier,
/// which on the exact fixtures is already its exact coordinate.
pub fn frozen_surface(plc: &TaggedPlc, params: &MeshParams) -> FrozenSurface {
    let sm = surface_mesh(plc, params);
    let is_planar = |sid: u32| matches!(sm.surfaces[sid as usize], SurfaceKind::Plane);
    let unit = |v: V3| {
        let l = dot(v, v).sqrt();
        if l > 0.0 {
            scale(v, 1.0 / l)
        } else {
            [0.0, 0.0, 0.0]
        }
    };

    // Per-facet carrier.
    let face_carrier: Vec<Carrier> = sm
        .faces
        .iter()
        .map(|f| {
            if is_planar(f.surface) {
                let (a, b, c) = (sm.points[f.tri[0]], sm.points[f.tri[1]], sm.points[f.tri[2]]);
                Carrier::Plane { p0: a, n: unit(cross(sub(b, a), sub(c, a))) }
            } else {
                Carrier::Surface(sm.surfaces[f.surface as usize].clone())
            }
        })
        .collect();

    // Per-vertex carrier: aggregate the incident facets. Any curved incidence
    // pins it to that surface; a single plane pins it to that plane; several
    // distinct planes (a corner or feature edge) stay an explicit vertex.
    let n = sm.points.len();
    let mut curved_kind: Vec<Option<SurfaceKind>> = vec![None; n];
    let mut plane: Vec<Option<(V3, V3)>> = vec![None; n];
    let mut multi_plane = vec![false; n];
    for (fi, f) in sm.faces.iter().enumerate() {
        for &v in &f.tri {
            match &face_carrier[fi] {
                Carrier::Surface(k) => curved_kind[v] = Some(k.clone()),
                Carrier::Plane { p0, n } => match plane[v] {
                    None => plane[v] = Some((*p0, *n)),
                    Some((_, n0)) if dot(*n, n0).abs() < 0.999 => multi_plane[v] = true,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    let vert_carrier: Vec<Carrier> = (0..n)
        .map(|v| {
            if let Some(k) = &curved_kind[v] {
                Carrier::Surface(k.clone())
            } else if let (Some((p0, nrm)), false) = (plane[v], multi_plane[v]) {
                Carrier::Plane { p0, n: nrm }
            } else {
                Carrier::Vertex
            }
        })
        .collect();

    FrozenSurface {
        points: sm.points,
        vert_carrier,
        faces: sm.faces,
        face_carrier,
        surfaces: sm.surfaces,
        surface_owners: sm.surface_owners,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::Zero;
    use rapidmesh_geom::{extrude_spline_profile, icosphere, solid_box, NurbsCurve, Scene};
    use rapidmesh_testutil::rat;

    #[test]
    fn frozen_surface_box_carries_exact_planes() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let fs = frozen_surface(&plc, &MeshParams { maxh: 1.0, ..Default::default() });
        // Every facet of a box is planar -> a Plane carrier (recovery-free).
        assert!(fs.face_carrier.iter().all(|c| matches!(c, Carrier::Plane { .. })));
        assert!(!fs.faces.is_empty());
        // Every vertex's exact() reconstructs its position and lands on a box face
        // (bit-exact for these axis-aligned planes).
        for i in 0..fs.points.len() {
            let e = Site::at(fs.vert_carrier[i].clone(), fs.points[i]).exact();
            let p = e.approx().expect("valid");
            assert!(dist(p, fs.points[i]) < 1e-9, "carrier moved the vertex");
            let on = p[0] == 0.0 || p[0] == 2.0 || p[1] == 0.0 || p[1] == 3.0 || p[2] == 0.0 || p[2] == 4.0;
            assert!(on, "frozen vertex {p:?} is not on the box surface");
        }
    }

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
