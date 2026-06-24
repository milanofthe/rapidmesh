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

// All tuning constants are centralised in crate::constants.
use crate::constants::{
    BOX_PAD_FRAC, DEFAULT_SUBDIV, LLOYD_CONVERGE_FRAC, LLOYD_ITERS, LLOYD_QUALITY_STALL,
    SEPARATION_FRAC, SLIVER_DEG, SURFACE_OVERSAMPLE, SURF_LLOYD_ITERS, TET_FACES,
};

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
    let mut domain = DomainTree::build(plc, params, &[]);
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
    let surf_point_size = ss.point_size; // per-surface-point generation size
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
    // Each surface point seeds the volume field at the size it was GENERATED at
    // (its per-entity edge/face target, finite even where `domain.h_at` is the
    // coarse bulk): a refined edge keeps a fine field around it, so the quality
    // post-pass does not coarsen it back.
    let vol_sources: Vec<(V3, f64)> = (0..n_surf)
        .map(|i| (surf_pos[i], surf_point_size[i].min(domain.h_at(surf_pos[i])).max(1e-9)))
        .collect();
    let vol_field = crate::sizefield::SizeField::new(vol_sources, grad, params.vol_cap());
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
            let sum4: V3 = std::array::from_fn(|k| p[0][k] + p[1][k] + p[2][k] + p[3][k]);
            // DENSITY-WEIGHTED ODT (adaptive): weight by volume * rho (rho = 1/h^3,
            // pulling sites toward finer regions). The ODT vertex update is
            // x* = (Σ_T w · sum of T's OTHER three vertices) / (3 Σ_T w) -- the
            // optimal-Delaunay relocation (sympy-derived), far more sliver-resistant
            // than the CVT centroid it replaces.
            let w = if params.density_weighted {
                let h = vol_field.at(c).min(region_cap(domain.region_at(c))).max(1e-9);
                tet_det(p).abs() / (h * h * h)
            } else {
                tet_det(p).abs()
            };
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * (sum4[k] - pos[i][k]);
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
            let mut tgt: V3 = std::array::from_fn(|k| num[i][k] / (3.0 * den[i]));
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

        // ---- adaptive insertion: fill the COARSE spots -------------------
        // Where a tet edge is longer than the LOCAL target `h`, the interior is
        // under-seeded (a per-region/per-face refinement the initial grid was too
        // coarse to honour). Drop a free site at that edge's midpoint; the next
        // pass relaxes the new density into place (the report's relax-insert
        // loop). Insert in the bulk only (one local element clear of the fixed
        // surface), separation-guarded against existing and newly added sites.
        let mut inserted = 0usize;
        if sites.len() < params.max_points {
            domain.rebucket(&positions(&sites));
            let cell = (SEPARATION_FRAC * spacing).max(1e-9);
            let nkey = |p: V3| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
            let mut newgrid: DMap<(i64, i64, i64), Vec<V3>> = DMap::default();
            for t in &tets {
                if sites.len() + inserted >= params.max_points {
                    break;
                }
                let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
                // The longest edge of this tet.
                let mut best = (0.0f64, [0.0; 3]);
                for &(a, b) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                    let len = dist(p[a], p[b]);
                    if len > best.0 {
                        best = (len, [(p[a][0] + p[b][0]) * 0.5, (p[a][1] + p[b][1]) * 0.5, (p[a][2] + p[b][2]) * 0.5]);
                    }
                }
                let (len, mid) = best;
                if !inside(mid) {
                    continue;
                }
                let h = vol_field.at(mid).min(region_cap(domain.region_at(mid))).max(1e-9);
                if len <= h {
                    continue; // already fine enough here
                }
                // Clear of the fixed surface by one local element (a seed nearer
                // than that makes a boundary tet the restricted boundary cannot emit).
                if surf_tree.nearest(mid).map(|j| dist(mid, sites[j as usize].pos()) < h).unwrap_or(false) {
                    continue;
                }
                let r = SEPARATION_FRAC * h;
                if domain.neighbors(mid, r).into_iter().next().is_some() {
                    continue; // too close to an existing site
                }
                // Separation from sites added THIS pass (query enough rings for r).
                let (kx, ky, kz) = nkey(mid);
                let reach = (r / cell).ceil() as i64 + 1;
                let r2 = r * r;
                let mut clear = true;
                'ins: for dx in -reach..=reach {
                    for dy in -reach..=reach {
                        for dz in -reach..=reach {
                            if let Some(v) = newgrid.get(&(kx + dx, ky + dy, kz + dz)) {
                                if v.iter().any(|&q| dot(sub(mid, q), sub(mid, q)) < r2) {
                                    clear = false;
                                    break 'ins;
                                }
                            }
                        }
                    }
                }
                if clear {
                    newgrid.entry((kx, ky, kz)).or_default().push(mid);
                    sites.push(Site::free(mid));
                    inserted += 1;
                }
            }
        }
        rmlog::stat("mesh.lloyd_inserts", inserted as f64);

        if inserted == 0 && max_move < LLOYD_CONVERGE_FRAC * spacing {
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
    // curvature-fine detail. A surface point keeps its per-entity generation size
    // (so an intentionally refined edge/face is not coarsened back).
    let point_size: Vec<f64> = pts
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let field = vol_field.at(p).min(region_cap(domain.region_at(p)));
            if i < n_surf {
                field.min(surf_point_size[i])
            } else {
                field
            }
        })
        .collect();
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
        n_nonconformal_faces: 0,
    };

    let q = quality_stats(&mesh);
    rmlog::stat("mesh.points", mesh.points.len() as f64);
    rmlog::stat("mesh.tets", mesh.tets.len() as f64);
    rmlog::stat("mesh.surface_points", n_surf as f64);
    rmlog::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    rmlog::stage("mesh.total", t_start.elapsed().as_secs_f64());
    mesh
}

/// Reconcile benign band flips so flood classification is leak-free. A curved
/// surface quad split by the frozen diagonal d1 but tetrahedralized by the other
/// diagonal d2 leaves d1's two frozen faces NOT tet faces (forcing them would be a
/// flat sliver -- recovery correctly leaves them). The flood oracle then has no
/// wall on that quad and leaks. The fix: replace those frozen faces with the
/// quad's OTHER-diagonal triangles, which ARE tet faces -- same geometry, same
/// region, just the volume's chosen diagonal. Returns the number of frozen faces
/// replaced. After this, the only remaining non-tet frozen faces are genuine
/// unrecovered defects (creases), so flood-exactness reduces to "no defect left".
fn reconcile_benign_bands(surf_faces: &mut Vec<SurfaceFace>, pts: &[V3], tets: &[[usize; 4]]) -> usize {
    use std::collections::HashSet;
    let mut tetface: DMap<[usize; 3], ()> = DMap::default();
    for t in tets {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            tetface.insert(f, ());
        }
    }
    let is_tf = |a: usize, b: usize, c: usize| {
        let mut f = [a, b, c];
        f.sort_unstable();
        tetface.contains_key(&f)
    };
    // Edge -> incident frozen faces that are NOT tet faces, with the apex vertex.
    let mut em: DMap<(usize, usize), Vec<(usize, usize)>> = DMap::default();
    for (fi, sf) in surf_faces.iter().enumerate() {
        let t = sf.tri;
        if is_tf(t[0], t[1], t[2]) {
            continue;
        }
        for k in 0..3 {
            let (u, w) = (t[k], t[(k + 1) % 3]);
            let apex = t[(k + 2) % 3];
            em.entry((u.min(w), u.max(w))).or_default().push((fi, apex));
        }
    }
    let mut remove: HashSet<usize> = HashSet::new();
    let mut add: Vec<SurfaceFace> = Vec::new();
    for (&(u, w), faces) in &em {
        if faces.len() != 2 {
            continue;
        }
        let (fi1, p1) = faces[0];
        let (fi2, p2) = faces[1];
        if p1 == p2 || remove.contains(&fi1) || remove.contains(&fi2) {
            continue;
        }
        // Quad {u,w,p1,p2}: current diagonal (u,w), alternate (p1,p2). If the
        // alternate's two triangles are tet faces, flip the surface to them.
        if is_tf(u, p1, p2) && is_tf(w, p1, p2) {
            let f1 = surf_faces[fi1].clone();
            let n1 = cross(sub(pts[f1.tri[1]], pts[f1.tri[0]]), sub(pts[f1.tri[2]], pts[f1.tri[0]]));
            for tri0 in [[u, p1, p2], [w, p1, p2]] {
                let mut tri = tri0;
                let n = cross(sub(pts[tri[1]], pts[tri[0]]), sub(pts[tri[2]], pts[tri[0]]));
                if dot(n, n1) < 0.0 {
                    tri.swap(1, 2); // wind outward, matching the replaced face
                }
                add.push(SurfaceFace { tri, face_tag: f1.face_tag, regions: f1.regions, patch: f1.patch, surface: f1.surface });
            }
            remove.insert(fi1);
            remove.insert(fi2);
        }
    }
    let n = remove.len();
    if n > 0 {
        let mut out: Vec<SurfaceFace> =
            surf_faces.iter().enumerate().filter(|(i, _)| !remove.contains(i)).map(|(_, f)| f.clone()).collect();
        out.extend(add);
        *surf_faces = out;
    }
    n
}

/// Boundary-constrained variant of [`mesh`] (the report's Stage 3): build the
/// frozen Stage-2 surface ([`frozen_surface`]), fill the interior at the sizing
/// density, and tetrahedralize with the surface as a HARD constraint
/// ([`crate::cdt3`]) so the boundary is watertight by construction (no straddling
/// tetrahedra). Region tags are by centroid, which is exact now that no tet
/// straddles; boundary faces are the ones between differing regions. Built
/// alongside [`mesh`] and validated incrementally before it becomes the default.
pub fn mesh_cdt(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    use rayon::prelude::*;
    let t_start = std::time::Instant::now();

    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
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
    // The brep is built up front so its faces' resolved per-face `surf_maxh` can
    // size the VOLUME field too (a finely sized face must refine the volume behind
    // it, else optimize collapses the fine surface back to the coarse bulk target).
    let brep = rapidmesh_brep::build::from_plc(plc);
    // Per-face `surf_maxh` -> per-facet volume target (see `DomainTree::build`).
    let mut facet_surf = vec![f64::INFINITY; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        if let Some(&(_, h)) = params.surf_maxh.iter().find(|&&(i, _)| i as usize == fid) {
            for &ti in &f.facets {
                facet_surf[ti as usize] = facet_surf[ti as usize].min(h);
            }
        }
    }
    // Per-edge `edge_maxh` -> point sources sampled along the brep edge's chain, so
    // the volume field stays fine along a refined edge (the same growth-from-feature
    // mechanism as faces; the chain is sampled at ~h so the field holds `h` along
    // it). Only clones the params when an edge override is actually present.
    let domain = if params.edge_maxh.is_empty() {
        DomainTree::build(plc, params, &facet_surf)
    } else {
        let mut pa = params.clone();
        for (eid, e) in brep.edges.iter().enumerate() {
            let Some(&(_, h)) = params.edge_maxh.iter().find(|&&(i, _)| i as usize == eid) else {
                continue;
            };
            for w in e.chain.windows(2) {
                let n = ((dist(w[0], w[1]) / h).ceil() as usize).max(1);
                for k in 0..n {
                    let t = k as f64 / n as f64;
                    pa.size_points.push((
                        std::array::from_fn(|c| w[0][c] + t * (w[1][c] - w[0][c])),
                        h,
                    ));
                }
            }
            if let Some(&last) = e.chain.last() {
                pa.size_points.push((last, h));
            }
        }
        DomainTree::build(plc, &pa, &facet_surf)
    };
    let patches = build_patches(plc);

    // ---- stages 1+2: the UNIFIED surface (B1) -----------------------------
    // The one chart/oracle-driven surface (`brep_mesh::surface_sites`) IS the frozen
    // surface for the constrained volume: it already returns `Site`s (carrier + pos),
    // its constrained per-face triangles, and a carrier per triangle, all carrying
    // the per-entity sizing. mesh_cdt freezes these as hard constraints (watertight by
    // construction) -- no separate `frozen_surface` patch path.
    let t_surf = std::time::Instant::now();
    let ss = crate::brep_mesh::surface_sites(&brep, plc, params, &domain);
    let surf_points: Vec<V3> = ss.sites.iter().map(|s| s.pos()).collect();
    let surf_tris: Vec<[usize; 3]> = ss.tris.iter().map(|f| f.tri).collect();
    let surf_sites: Vec<Site> = ss.sites;
    let face_carrier: Vec<Carrier> = ss.tri_carrier;
    let mut surf_faces: Vec<SurfaceFace> = ss.tris;
    rmlog::stat("mesh_cdt.surf_points", surf_points.len() as f64);
    rmlog::stat("mesh_cdt.surf_faces", surf_faces.len() as f64);
    rmlog::stage("mesh_cdt.surface", t_surf.elapsed().as_secs_f64());

    // ---- stage 3 seeding: graded interior, clear of the surface -----------
    let t_seed = std::time::Instant::now();
    let surf_tree = Octree::build(&surf_points);
    // Local target size: the graded field, capped by the region's own maxh (a
    // region-wide size the boundary-grown field does not enforce in the interior).
    let region_cap = |r: u32| -> f64 {
        params
            .region_maxh
            .iter()
            .find(|(rr, _)| *rr == r)
            .map(|&(_, h)| h)
            .unwrap_or(params.maxh)
            .min(params.maxh)
    };
    let hloc = |p: V3| domain.h_at(p).min(region_cap(domain.region_at(p))).min(params.vol_cap()).max(1e-9);
    let step = (0.7 * spacing).max(1e-9);
    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let ncell = (0.6 * spacing).max(1e-9);
    let ckey = |p: V3| ((p[0] / ncell).floor() as i64, (p[1] / ncell).floor() as i64, (p[2] / ncell).floor() as i64);
    let mut igrid: DMap<(i64, i64, i64), Vec<V3>> = DMap::default();
    let mut interior: Vec<V3> = Vec::new();
    let (nx, ny, nz) = (
        ((span[0] / step).ceil() as i64).max(1),
        ((span[1] / step).ceil() as i64).max(1),
        ((span[2] / step).ceil() as i64).max(1),
    );
    for i in 1..nx {
        for j in 1..ny {
            for k in 1..nz {
                let p: V3 = [lo[0] + i as f64 * step, lo[1] + j as f64 * step, lo[2] + k as f64 * step];
                if !inside(p) {
                    continue;
                }
                let h = hloc(p);
                // Clearance from the surface: a seed within one local element of
                // the boundary makes a tet the recovery cannot emit. Non-negotiable.
                if surf_tree.nearest(p).map(|q| dist(p, surf_points[q as usize]) < h).unwrap_or(false) {
                    continue;
                }
                let r2 = (0.6 * h).powi(2);
                let (kx, ky, kz) = ckey(p);
                let mut clear = true;
                'scan: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(v) = igrid.get(&(kx + dx, ky + dy, kz + dz)) {
                                if v.iter().any(|&q| dot(sub(p, q), sub(p, q)) < r2) {
                                    clear = false;
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
                if clear {
                    igrid.entry((kx, ky, kz)).or_default().push(p);
                    interior.push(p);
                }
            }
        }
    }
    rmlog::stat("mesh_cdt.vol_seeds", interior.len() as f64);
    rmlog::stage("mesh_cdt.seed", t_seed.elapsed().as_secs_f64());

    // ---- 3D Lloyd: relax the interior toward its CVT layout ----------------
    // The surface is frozen (it was meshed in its own dimension), so only the
    // interior points move, each to the (uniform-weighted) centroid of its
    // incident tets in an UNCONSTRAINED Delaunay rebuilt per pass (the relaxation
    // only needs approximate cells; the final build below is constrained). Out-of
    // domain or too-close-to-surface targets are rejected.
    let t_lloyd = std::time::Instant::now();
    let n_surf = surf_points.len();
    let build_full = |all_pos: &[V3]| -> Vec<[usize; 4]> {
        let order = crate::spatial::morton_order(all_pos);
        let mut db = DelaunayBuilder::enclosing(lo, hi);
        let mut orig: Vec<usize> = Vec::with_capacity(order.len());
        for &i in &order {
            if db.try_insert(all_pos[i]).is_some() {
                orig.push(i);
            }
        }
        db.tets().into_iter().map(|t| std::array::from_fn(|j| orig[t[j]])).collect()
    };
    // Adaptive termination state: the fewest interior slivers seen so far and a
    // plateau counter (only armed once insertions stop). LLOYD_ITERS is the hard
    // cap; the quality plateau ends easy geometries well before it.
    let mut lloyd_passes = 0usize;
    let mut best_slivers = usize::MAX;
    let mut q_stall = 0usize;
    for _ in 0..LLOYD_ITERS {
        lloyd_passes += 1;
        let mut all: Vec<V3> = surf_points.clone();
        all.extend_from_slice(&interior);
        let tets = build_full(&all);
        let mut num = vec![[0.0f64; 3]; all.len()];
        let mut den = vec![0.0f64; all.len()];
        let mut slivers = 0usize;
        for t in &tets {
            let p = [all[t[0]], all[t[1]], all[t[2]], all[t[3]]];
            let sum4: V3 = std::array::from_fn(|k| p[0][k] + p[1][k] + p[2][k] + p[3][k]);
            let w = tet_det(p).abs();
            // Quality monitor: count IN-DOMAIN slivers of this pass's layout to
            // drive adaptive termination (exterior hull tets are excluded so the
            // count tracks the mesh we actually keep).
            if crate::diagnostics::tet_min_dihedral(p) < SLIVER_DEG && inside(centroid4(p)) {
                slivers += 1;
            }
            // ODT relocation: x* = (Σ_T |T| · sum of T's OTHER three verts) /
            // (3 Σ_T |T|) -- the optimal-Delaunay update (sympy-derived), far more
            // sliver-resistant than the CVT/Voronoi centroid it replaces.
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * (sum4[k] - all[i][k]);
                }
                den[i] += w;
            }
        }
        // Rebuild the crowding hash from the surface + interior each pass.
        let mut grid: DMap<(i64, i64, i64), Vec<V3>> = DMap::default();
        for &p in surf_points.iter().chain(interior.iter()) {
            let (kx, ky, kz) = ckey(p);
            grid.entry((kx, ky, kz)).or_default().push(p);
        }
        let mut max_move = 0.0f64;
        for k in 0..interior.len() {
            let i = n_surf + k;
            if den[i] == 0.0 {
                continue;
            }
            let tgt: V3 = std::array::from_fn(|c| num[i][c] / (3.0 * den[i]));
            if !inside(tgt) {
                continue;
            }
            let h = hloc(tgt);
            if surf_tree.nearest(tgt).map(|q| dist(tgt, surf_points[q as usize]) < h).unwrap_or(false) {
                continue;
            }
            // Crowding: keep clear of every OTHER point by the local radius.
            let r2 = (0.6 * h).powi(2);
            let (kx, ky, kz) = ckey(tgt);
            let cur = interior[k];
            let mut clear = true;
            'scan2: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &q in v {
                                if q != cur && dot(sub(tgt, q), sub(tgt, q)) < r2 {
                                    clear = false;
                                    break 'scan2;
                                }
                            }
                        }
                    }
                }
            }
            if clear {
                max_move = max_move.max(dist(cur, tgt));
                interior[k] = tgt;
            }
        }

        // ---- adaptive insertion at the COARSEST spots --------------------
        // Insert at the longest edge of any tet whose length exceeds the local
        // target (where the grid is too coarse, NOT where an error peaks -- that
        // would over-densify), then let the next pass relax the new density in:
        // robust coarse-to-fine refinement (the report's relax-insert loop).
        let mut inserted = 0usize;
        if n_surf + interior.len() < params.max_points {
            let mut newgrid: DMap<(i64, i64, i64), Vec<V3>> = DMap::default();
            let mut adds: Vec<V3> = Vec::new();
            for t in &tets {
                if n_surf + interior.len() + adds.len() >= params.max_points {
                    break;
                }
                let p = [all[t[0]], all[t[1]], all[t[2]], all[t[3]]];
                let mut best = (0.0f64, [0.0; 3]);
                for &(a, b) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                    let len = dist(p[a], p[b]);
                    if len > best.0 {
                        best = (len, [(p[a][0] + p[b][0]) * 0.5, (p[a][1] + p[b][1]) * 0.5, (p[a][2] + p[b][2]) * 0.5]);
                    }
                }
                let (len, mid) = best;
                if !inside(mid) {
                    continue;
                }
                let h = hloc(mid);
                if len <= h {
                    continue; // already fine enough here
                }
                if surf_tree.nearest(mid).map(|q| dist(mid, surf_points[q as usize]) < h).unwrap_or(false) {
                    continue; // one local element clear of the fixed surface
                }
                let r2 = (SEPARATION_FRAC * h).powi(2);
                let (kx, ky, kz) = ckey(mid);
                let mut clear = true;
                'ins: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            for g in [&grid, &newgrid] {
                                if let Some(v) = g.get(&(kx + dx, ky + dy, kz + dz)) {
                                    if v.iter().any(|&q| dot(sub(mid, q), sub(mid, q)) < r2) {
                                        clear = false;
                                        break 'ins;
                                    }
                                }
                            }
                        }
                    }
                }
                if clear {
                    newgrid.entry((kx, ky, kz)).or_default().push(mid);
                    adds.push(mid);
                }
            }
            inserted = adds.len();
            interior.extend(adds);
        }

        // Adaptive termination. While the field is still being seeded keep
        // going (insertions transiently raise the sliver count); once seeding
        // stops, end when the layout has settled geometrically OR stopped
        // shedding slivers for LLOYD_QUALITY_STALL passes.
        if inserted > 0 {
            best_slivers = best_slivers.min(slivers);
            q_stall = 0;
        } else {
            if slivers < best_slivers {
                best_slivers = slivers;
                q_stall = 0;
            } else {
                q_stall += 1;
            }
            if max_move < LLOYD_CONVERGE_FRAC * spacing || q_stall >= LLOYD_QUALITY_STALL {
                break;
            }
        }
    }
    rmlog::stage("mesh_cdt.lloyd", t_lloyd.elapsed().as_secs_f64());
    if std::env::var_os("RAPIDMESH_LLOYD_TRACE").is_some() {
        eprintln!("LLOYD_PASSES {lloyd_passes} surf {} interior {} slivers {} time_ms {:.0}", n_surf, interior.len(), best_slivers, t_lloyd.elapsed().as_secs_f64() * 1000.0);
    }

    // ---- constrained tetrahedralization -----------------------------------
    let t_build = std::time::Instant::now();
    let _ = (&inside, &hloc); // (inside oracle / size field; refinement TBD)
    let con = crate::cdt3::tetrahedralize_constrained(
        &surf_sites, &surf_tris, &face_carrier, &interior, lo, hi,
    );
    // Steiner points cdt3 had to insert during facet recovery (beyond surface +
    // interior): the health signal for curved recovery -- small/bounded = GO.
    let steiner = con.points.len().saturating_sub(con.n_surf_verts + interior.len());
    rmlog::stat("mesh_cdt.steiner", steiner as f64);
    let pts = con.points;
    rmlog::stage("mesh_cdt.tetrahedralize", t_build.elapsed().as_secs_f64());

    // Path A (exact flood + frozen-faces boundary) is leak-free iff EVERY frozen
    // face is a tet face. Recovery seals the crease bridges; benign band flips are
    // sealed by reconciling them to the volume's diagonal (no degenerate sliver).
    // If any frozen face remains non-tet (an unrecovered defect), flood would leak
    // there, so fall back to centroid classification + region-difference boundary
    // (the prior default) -- no regression on geometries we cannot fully seal.
    // Path A (flood + frozen boundary) is an opt-in mode (RAPIDMESH_PATHA). It is
    // enabled per geometry only when every frozen face is a tet face (so flood
    // cannot leak): recovery seals crease bridges, reconciliation seals benign band
    // flips. It fixes the straddler geometries (fused_unequal, rf_toroid, chain ->
    // watertight + straddler-free) but still introduces boundary slivers on some
    // curved bodies (a fixed-boundary sliver problem for the sliver stage) and a
    // non-manifold reconciliation on a few crossing-body geometries -- so it is NOT
    // the default yet. Outside Path A, surf_faces is untouched (exact prior path).
    let path_a = if std::env::var_os("RAPIDMESH_PATHA").is_some() {
        let n_recon = reconcile_benign_bands(&mut surf_faces, &pts, &con.tets);
        let tetfaces: std::collections::HashSet<[usize; 3]> = con
            .tets
            .iter()
            .flat_map(|t| {
                TET_FACES.iter().map(move |fv| {
                    let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
                    f.sort_unstable();
                    f
                })
            })
            .collect();
        let ok = surf_faces.iter().all(|sf| {
            let mut f = sf.tri;
            f.sort_unstable();
            tetfaces.contains(&f)
        });
        if std::env::var_os("RAPIDMESH_PATHA_TRACE").is_some() {
            eprintln!("[path_a] reconciled={n_recon} path_a={ok} surf_faces={}", surf_faces.len());
        }
        ok
    } else {
        false
    };

    // ---- region classification --------------------------------------------
    // Default: centroid inside/outside test against the faceted PLC. This keeps
    // tets whose centroid is inside the faceting but outside the true (finer)
    // surface, so the kept volume bulges past the bottom-up surface (the n>=2
    // conformity gap). RAPIDMESH_FLOOD instead floods the region tag blocked by
    // the frozen surface faces (cdt3::classify_regions): the kept volume then
    // conforms exactly to that surface -- no bulge -- provided every frozen face
    // is a tet face (facet recovery seals the remaining leaks).
    let region: Vec<u32> = if path_a {
        let mut oracle: DMap<[usize; 3], (u32, u32, V3)> = DMap::default();
        for f in &surf_faces {
            let p = [pts[f.tri[0]], pts[f.tri[1]], pts[f.tri[2]]];
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let mut k = f.tri;
            k.sort_unstable();
            oracle.insert(k, (f.regions[0].0, f.regions[1].0, n));
        }
        crate::cdt3::classify_regions(&con.tets, &pts, |f| oracle.get(f).copied())
    } else {
        con.tets
            .par_iter()
            .map(|t| domain.region_at(centroid4([pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]])))
            .collect()
    };
    let mut kept: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for (t, &r) in con.tets.iter().zip(&region) {
        if r != 0 {
            kept.push(*t);
            tet_regions.push(RegionTag(r));
        }
    }

    // ---- boundary faces: between differing regions, tagged from the frozen
    // surface (curved facets match exactly; planar fall back to the patch) -----
    let mut tag: DMap<[usize; 3], (u32, FaceTag, u32)> = DMap::default();
    // surface index -> a representative face tag (for the curved fallback below).
    let mut surf_face_tag: DMap<u32, FaceTag> = DMap::default();
    for f in &surf_faces {
        let mut s = f.tri;
        s.sort_unstable();
        tag.insert(s, (f.surface, f.face_tag, f.patch));
        surf_face_tag.entry(f.surface).or_insert(f.face_tag);
    }
    // C3b: tag a curved boundary face (extracted by region difference, so NOT an
    // exact surf_faces tri and not coplanar with a planar patch) with the analytic
    // surface MOST of its vertices lie on -- so the surface-deviation / straddler
    // diagnostics measure against the TRUE carrier (a torus interface as a torus,
    // not the surface-0 plane), and a real straddler is counted against the surface
    // it should sit on. Without this the fallback was surface 0 -> dev ~ 0 always.
    let dist3 = |a: V3, b: V3| ((a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)+(a[2]-b[2]).powi(2)).sqrt();
    let curved_surface_of = |key: &[usize; 3], pts: &[V3]| -> Option<u32> {
        let tri = [pts[key[0]], pts[key[1]], pts[key[2]]];
        let longest = (0..3).map(|k| dist3(tri[k], tri[(k + 1) % 3])).fold(0.0, f64::max);
        let tol = 0.1 * longest.max(1e-12);
        let mut best: Option<(u32, usize, f64)> = None; // (surface, #on, sum dev)
        for (si, kind) in plc.surfaces.iter().enumerate() {
            if matches!(kind, rapidmesh_geom::SurfaceKind::Plane) {
                continue; // planes handled by patch_of_face
            }
            let devs = tri.map(|q| dist3(q, crate::project::closest_on_surface(kind, q)));
            let on = devs.iter().filter(|&&d| d < tol).count();
            let tot: f64 = devs.iter().sum();
            if on >= 1 && best.map_or(true, |(_, bon, btot)| on > bon || (on == bon && tot < btot)) {
                best = Some((si as u32, on, tot));
            }
        }
        best.map(|(si, _, _)| si)
    };
    let mut face_owners: DMap<[usize; 3], Vec<u32>> = DMap::default();
    for (ti, t) in con.tets.iter().enumerate() {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            face_owners.entry(f).or_default().push(ti as u32);
        }
    }
    let eps = 1e-9 * diag.max(1.0);
    let mut faces: Vec<SurfaceFace> = Vec::new();
    for (key, owners) in &face_owners {
        let (ra, rb) = match owners.as_slice() {
            [a, b] => (region[*a as usize], region[*b as usize]),
            [a] => (region[*a as usize], 0),
            _ => continue,
        };
        if ra == rb {
            continue;
        }
        let (surface, face_tag, patch) = if let Some(&(s, ft, p)) = tag.get(key) {
            (s, ft, p)
        } else if let Some(pi) = patch_of_face(plc, &patches, &pts, *key, eps) {
            (patches[pi].surface, patches[pi].face_tag, pi as u32)
        } else if let Some(si) = curved_surface_of(key, &pts) {
            (si, surf_face_tag.get(&si).copied().unwrap_or(FaceTag(0)), u32::MAX)
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
    }

    // The bottom-up surface stage already produced a clean, watertight surface
    // (surf_faces, 0 straddlers). Taking it directly as the boundary -- instead of
    // re-extracting it from the volume by region difference, which bridges concave
    // creases -- gives an exact boundary. The frozen tris index the same points (the
    // surface vertices lead `pts`), so they need no remapping. (Volume conformity to
    // these faces is the job of facet recovery; this is the extraction side.)
    let faces = if path_a {
        if std::env::var_os("RAPIDMESH_CONFORM_PROBE").is_some() {
            // Conformity: how many frozen boundary faces are actually a face of a
            // KEPT tet. A gap means the volume does not conform there (the facet
            // recovery's real job, hidden by the clean face-set metrics).
            let mut kept_faces: DMap<[usize; 3], usize> = DMap::default();
            for t in &kept {
                for fv in &TET_FACES {
                    let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
                    f.sort_unstable();
                    *kept_faces.entry(f).or_insert(0) += 1;
                }
            }
            let (mut n1, mut n0, mut n2) = (0usize, 0usize, 0usize);
            for sf in &surf_faces {
                let mut k = sf.tri;
                k.sort_unstable();
                match kept_faces.get(&k).copied().unwrap_or(0) {
                    1 => n1 += 1,        // conformal boundary face
                    0 => n0 += 1,        // not a kept-tet face at all (missing/discarded)
                    _ => n2 += 1,        // interior to kept volume (volume pokes outside)
                }
            }
            eprintln!(
                "[conform] frozen_faces={} conformal(n=1)={n1} missing(n=0)={n0} interior(n>=2)={n2}",
                surf_faces.len(),
            );
        }
        surf_faces.clone()
    } else {
        faces
    };

    // Conformity (the Path-A north-star metric): each FROZEN surf_face must be a
    // real face of the KEPT volume with the incidence its region pair demands --
    // an outer-boundary face (one region 0) borders exactly one kept tet, an
    // interface face (both regions != 0) exactly two, one per side. A shortfall
    // (n=0) is an un-recovered/missing facet; an excess or wrong-region set is the
    // volume poking past the surface. Either way the boundary != the frozen
    // surface there, which is EXACTLY a straddler. Measured against `kept` +
    // `surf_faces` (the only place both coexist); 0 here <=> straddler-free.
    let n_nonconformal_faces = {
        let mut inc: DMap<[usize; 3], Vec<u32>> = DMap::default();
        for (t, r) in kept.iter().zip(&tet_regions) {
            for fv in &TET_FACES {
                let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
                f.sort_unstable();
                inc.entry(f).or_default().push(r.0);
            }
        }
        let frozen: std::collections::HashSet<[usize; 3]> = surf_faces
            .iter()
            .map(|sf| {
                let mut k = sf.tri;
                k.sort_unstable();
                k
            })
            .collect();
        // MISSING: a frozen face whose kept-tet incidence does not match its region
        // pair (an outer face wants one kept tet of its nonzero region, an interface
        // two, one per side). EXTRA: a kept-VOLUME boundary face (one kept tet, or
        // two of differing region) that is NOT a frozen face -- the volume poking
        // past the surface (a straddler-bearing face). nc = missing + extra is 0
        // iff the kept boundary equals the frozen surface exactly, in ANY
        // classification mode (so it cannot read 0 while a poke straddler exists).
        let mut missing = 0usize;
        for sf in &surf_faces {
            let mut k = sf.tri;
            k.sort_unstable();
            let mut want: Vec<u32> = sf.regions.iter().map(|r| r.0).filter(|&r| r != 0).collect();
            want.sort_unstable();
            let mut got = inc.get(&k).cloned().unwrap_or_default();
            got.sort_unstable();
            if got != want {
                missing += 1;
            }
        }
        let mut extra = 0usize;
        for (f, regs) in &inc {
            let is_boundary = regs.len() == 1 || (regs.len() == 2 && regs[0] != regs[1]);
            if is_boundary && !frozen.contains(f) {
                extra += 1;
            }
        }
        missing + extra
    };
    if std::env::var_os("RAPIDMESH_CONFORM_TRACE").is_some() {
        eprintln!("[conform] frozen_faces={} nonconformal={n_nonconformal_faces}", surf_faces.len());
    }

    let point_size: Vec<f64> = pts.iter().map(|&p| domain.h_at(p)).collect();
    let mesh = TetMesh {
        points: pts,
        tets: kept,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points: plc.vertices.len(),
        point_size,
        n_nonconformal_faces,
    };
    let q = quality_stats(&mesh);
    rmlog::stat("mesh_cdt.tets", mesh.tets.len() as f64);
    rmlog::stat("mesh_cdt.min_dihedral_deg", q.min_dihedral_deg);
    rmlog::stage("mesh_cdt.total", t_start.elapsed().as_secs_f64());
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
    let domain = DomainTree::build(plc, params, &[]);
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
        let target = |uv: [f64; 2]| SURFACE_OVERSAMPLE * domain.h_at(lift3(uv, drop, p0, n)).min(params.surf_cap());
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
            SURFACE_OVERSAMPLE * domain.h_at(xyz).min(hc).min(params.surf_cap())
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

    /// PROBE (Q1 investigation): does `optimize` clean mesh_cdt's slivers, and are
    /// the residual slivers on the BOUNDARY (constrained, untouchable) or interior?
    #[test]
    #[ignore]
    fn probe_optimize_on_slivers() {
        use crate::diagnostics::{diagnose, SLIVER_DEG};
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
        scene.add_void(rapidmesh_geom::cylinder([2.0, 2.0, -0.1], [0.0, 0.0, 2.2], 1.0, 32));
        let plc = scene.assemble();
        let params = MeshParams { maxh: 0.4, ..Default::default() };
        let mut m = mesh_cdt(&plc, &params);

        // boundary vertices = those on any surface face
        let bnd: std::collections::HashSet<usize> =
            m.faces.iter().flat_map(|f| f.tri).collect();
        let classify = |mesh: &TetMesh| -> (usize, usize, usize) {
            let (mut total, mut on_bnd, mut interior) = (0, 0, 0);
            for t in &mesh.tets {
                let p = [mesh.points[t[0]], mesh.points[t[1]], mesh.points[t[2]], mesh.points[t[3]]];
                if crate::diagnostics::tet_min_dihedral(p) < SLIVER_DEG {
                    total += 1;
                    if t.iter().any(|v| bnd.contains(v)) { on_bnd += 1; } else { interior += 1; }
                }
            }
            (total, on_bnd, interior)
        };
        let d0 = diagnose(&m);
        let (s0, b0, i0) = classify(&m);
        let opt = crate::optimize::OptimizeParams { maxh: 0.4, ..Default::default() };
        crate::optimize::optimize(&mut m, &opt);
        let d1 = diagnose(&m);
        let (s1, b1, i1) = classify(&m);
        eprintln!("BEFORE optimize: tets={} minDih={:.2} slivers={} (boundary {} / interior {})",
            d0.n_tets, d0.min_dihedral_deg, s0, b0, i0);
        eprintln!("AFTER  optimize: tets={} minDih={:.2} slivers={} (boundary {} / interior {})",
            d1.n_tets, d1.min_dihedral_deg, s1, b1, i1);
        // characterise the residual boundary slivers: how many verts on the
        // boundary (4 = surface-on-surface; 3 = one free interior apex), to choose
        // the exudation strategy.
        let bnd2: std::collections::HashSet<usize> = m.faces.iter().flat_map(|f| f.tri).collect();
        let (mut on4, mut on3, mut on_le2) = (0, 0, 0);
        for t in &m.tets {
            let p = [m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]];
            if crate::diagnostics::tet_min_dihedral(p) >= SLIVER_DEG {
                continue;
            }
            match t.iter().filter(|v| bnd2.contains(v)).count() {
                4 => on4 += 1,
                3 => on3 += 1,
                _ => on_le2 += 1,
            }
        }
        eprintln!("residual boundary slivers by #boundary-verts: 4-on-bnd={on4}  3-on-bnd={on3}  <=2={on_le2}");

        // For the 4-on-boundary slivers: do their 4 verts lie on ONE surface
        // (bad surface tris) or span MULTIPLE (thin region / sharp edge)?
        let mut vsurf: std::collections::HashMap<usize, std::collections::BTreeSet<u32>> =
            std::collections::HashMap::new();
        for f in &m.faces {
            for &v in &f.tri {
                vsurf.entry(v).or_default().insert(f.surface);
            }
        }
        let mut by_distinct: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut examples = 0;
        for t in &m.tets {
            let p = [m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]];
            if crate::diagnostics::tet_min_dihedral(p) >= SLIVER_DEG {
                continue;
            }
            if t.iter().filter(|v| bnd2.contains(v)).count() != 4 {
                continue;
            }
            let mut surfs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
            for v in t {
                if let Some(s) = vsurf.get(v) {
                    surfs.extend(s.iter().copied());
                }
            }
            *by_distinct.entry(surfs.len()).or_default() += 1;
            if examples < 4 {
                let c: V3 = std::array::from_fn(|k| p.iter().map(|q| q[k]).sum::<f64>() / 4.0);
                eprintln!("  ex 4-on-bnd sliver: surfaces={:?} centroid=[{:.2},{:.2},{:.2}]", surfs, c[0], c[1], c[2]);
                examples += 1;
            }
        }
        eprintln!("4-on-bnd slivers by #distinct-surfaces-touched: {:?}", by_distinct);
    }

    #[test]
    fn mesh_cdt_box_is_watertight_and_volume_correct() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let m = mesh_cdt(&plc, &MeshParams { maxh: 1.0, ..Default::default() });
        assert!(!m.tets.is_empty(), "box produced tets");
        // Filled volume = box volume (24).
        let vol: f64 = m
            .tets
            .iter()
            .map(|t| tet_det([m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]]).abs() / 6.0)
            .sum();
        assert!((vol - 24.0).abs() < 1e-6, "box volume should be 24, got {vol}");
        // Single region inside the box.
        assert!(m.tet_regions.iter().all(|r| r.0 == 1), "all tets in region 1");
        // Watertight boundary: the surface faces tile the box (area 52).
        let area: f64 = m
            .faces
            .iter()
            .map(|f| {
                let (a, b, c) = (m.points[f.tri[0]], m.points[f.tri[1]], m.points[f.tri[2]]);
                0.5 * dot(cross(sub(b, a), sub(c, a)), cross(sub(b, a), sub(c, a))).sqrt()
            })
            .sum();
        assert!((area - 52.0).abs() < 1e-6, "box surface area should be 52, got {area}");
    }

    #[test]
    fn mesh_cdt_sphere_recovers_curved_facets() {
        // A curved single region: exercises flip-based curved-facet recovery
        // end to end. The result must be a closed, single-region mesh whose
        // volume approaches the ball (the PL surface slightly undershoots).
        let mut scene = Scene::new();
        scene.add_solid(icosphere([0.0, 0.0, 0.0], 1.0, 2));
        let plc = scene.assemble();
        let m = mesh_cdt(&plc, &MeshParams { maxh: 0.4, ..Default::default() });
        assert!(!m.tets.is_empty(), "sphere produced tets");
        assert!(m.tet_regions.iter().all(|r| r.0 == 1), "all tets in region 1");
        let vol: f64 = m
            .tets
            .iter()
            .map(|t| tet_det([m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]]).abs() / 6.0)
            .sum();
        let ball = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(vol > 0.80 * ball && vol < 1.02 * ball, "sphere volume {vol} vs ball {ball}");
        // Closed manifold: every boundary edge is shared by exactly two faces.
        let mut edge: DMap<(usize, usize), usize> = DMap::default();
        for f in &m.faces {
            for k in 0..3 {
                let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edge.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edge.values().all(|&c| c == 2), "surface is not a closed manifold");
    }

    /// B2 GATE: a RADIAL UV `sphere()` (not a geodesic input) must mesh watertight
    /// through `mesh_cdt`. This proves the geodesic icosphere that `surface_sites`
    /// now generates for a closed sphere flows through `cdt3` curved-facet recovery
    /// as a hard constraint -- watertight, single region, volume near the ball, and
    /// WITHOUT a Steiner blow-up (recovery converges).
    #[test]
    fn gate_uv_sphere_meshes_watertight_via_cdt() {
        let mut scene = Scene::new();
        scene.add_solid(rapidmesh_geom::sphere([0.2, -0.1, 0.3], 1.0, 24, 12)); // radial input
        let plc = scene.assemble();
        let m = mesh_cdt(&plc, &MeshParams { maxh: 0.4, tol_surf: 1e-2, ..Default::default() });
        assert!(!m.tets.is_empty(), "uv sphere produced tets");
        assert!(m.tet_regions.iter().all(|r| r.0 == 1), "single region");
        let vol: f64 = m
            .tets
            .iter()
            .map(|t| tet_det([m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]]).abs() / 6.0)
            .sum();
        let ball = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(vol > 0.80 * ball && vol < 1.02 * ball, "uv sphere volume {vol} vs ball {ball}");
        let mut edge: DMap<(usize, usize), usize> = DMap::default();
        for f in &m.faces {
            for k in 0..3 {
                let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edge.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edge.values().all(|&c| c == 2), "uv sphere boundary is a closed manifold");
    }

    /// A full cylinder meshes WATERTIGHT through `mesh_cdt`. After the C3 pivot the
    /// curved barrel boundary is the restricted Delaunay of the surface points (no
    /// forced recovery, no Steiner): ~0.8s, down from 87s.
    #[test]
    fn cylinder_meshes_watertight_via_cdt() {
        let (r, hgt) = (1.0, 3.0);
        let mut scene = Scene::new();
        scene.add_solid(rapidmesh_geom::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, hgt], r, 24));
        let plc = scene.assemble();
        let m = mesh_cdt(&plc, &MeshParams { maxh: 0.5, ..Default::default() });
        assert!(!m.tets.is_empty(), "cylinder produced tets");
        assert!(m.tet_regions.iter().all(|rr| rr.0 == 1), "single region");
        let vol: f64 = m
            .tets
            .iter()
            .map(|t| tet_det([m.points[t[0]], m.points[t[1]], m.points[t[2]], m.points[t[3]]]).abs() / 6.0)
            .sum();
        let exact = std::f64::consts::PI * r * r * hgt;
        assert!(vol > 0.90 * exact && vol < 1.001 * exact, "cylinder volume {vol} vs {exact}");
        let mut edge: DMap<(usize, usize), usize> = DMap::default();
        for f in &m.faces {
            for k in 0..3 {
                let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edge.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edge.values().all(|&c| c == 2), "cylinder boundary is a closed manifold");
    }

    /// B2 GATE: surface sizing flows through `mesh_cdt` -- a finer `maxh_surf` makes a
    /// denser boundary triangulation (the per-entity sizing that previously did NOT
    /// reach mesh_cdt because it used `frozen_surface`).
    #[test]
    fn gate_surf_sizing_flows_through_cdt() {
        let build = |hs: f64| {
            let mut scene = Scene::new();
            scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
            let plc = scene.assemble();
            let m = mesh_cdt(
                &plc,
                &MeshParams { maxh: 4.0, maxh_surf: hs, ..Default::default() },
            );
            m.faces.len()
        };
        let coarse = build(4.0);
        let fine = build(1.0);
        assert!(fine > 2 * coarse, "finer maxh_surf must densify the boundary ({coarse} -> {fine})");
    }

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
