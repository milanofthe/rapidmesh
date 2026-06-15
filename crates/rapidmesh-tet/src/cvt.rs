//! CVT (centroidal Voronoi) tetrahedral meshing of a tagged PLC.
//!
//! Replaces the constrained-Delaunay + Steiner boundary recovery of the old
//! pipeline. The exact CSG arrangement (the `TaggedPlc`) is untouched; this
//! stage fills it with a variational (Lloyd-relaxed) tetrahedralization.
//!
//! Strict density-driven hierarchy (1D -> 2D -> 3D), each level fixed before the
//! next, as in established surface+volume CVT remeshing. Stage 1: corners fixed,
//! feature edges populated at the target density (straight edges uniform, the 1D
//! CVT optimum) and fixed. Stage 2: each planar patch filled by a 2D Lloyd
//! (`surf2d`) with its edge points fixed; interface patches are meshed once and
//! shared by both regions. Stage 3: the interior is filled and a 3D Lloyd relaxes
//! the volume points with the whole surface fixed.
//! A small protection/recovery pass then inserts on-plane points where a volume
//! tet still straddles a fixed face (the restricted-Delaunay condition is not
//! free), so interfaces and concave (void) boundaries conform exactly. Regions
//! are assigned by exact flood-fill (`classify_tet_regions`).
//!
//! All topological decisions use the exact predicates (`orient2d`/`orient3d`,
//! `incircle2d`, `Tri::contains_coplanar`, `point_inside_solid`); float is used
//! only for non-decision quantities (relaxation moves, centroid weights).

use crate::conform::{
    build_patches, classify_tet_regions, quality_stats, MeshParams, Patch, SurfaceFace, TetMesh,
};
use crate::delaunay::DelaunayBuilder;
use crate::seed::{bcc_lattice, SizingField};
use crate::spatial::Octree;
use crate::surf2d::cvt_fill;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{RegionTag, TaggedPlc};
use std::collections::{HashMap, HashSet};

type V3 = [f64; 3];

/// The four vertex-index triples spanning a tet's faces.
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

/// Lloyd relaxation passes (per stage).
const LLOYD_ITERS: usize = 12;
/// Bounding-box subdivisions for the default spacing when no `maxh` is given.
const DEFAULT_SUBDIV: f64 = 8.0;
/// Per-triangle bounding-box pad for the inside test, fraction of the diagonal.
const BOX_PAD_FRAC: f64 = 1e-6;
/// Minimum separation of a seeded/moved site from any other, fraction of spacing.
const SEPARATION_FRAC: f64 = 0.45;
/// Interface-recovery rounds cap (a divergence backstop; it converges in a few).
const MAX_RECOVER_ROUNDS: usize = 64;
/// A tet straddles a patch plane if vertices sit on both sides by more than this
/// (fraction of the scene diagonal); on-plane points sit exactly at 0.
const STRADDLE_EPS_FRAC: f64 = 1e-9;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
fn tet_det(p: [V3; 4]) -> f64 {
    dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])))
}
fn centroid4(p: [V3; 4]) -> V3 {
    std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]))
}
/// Orthogonal projection of `c` onto the plane (point `p0`, unit normal `n`).
fn project_plane(c: V3, p0: V3, n: V3) -> V3 {
    sub(c, scale(n, dot(sub(c, p0), n)))
}

fn tri_of(plc: &TaggedPlc, t: [u32; 3]) -> Tri {
    Tri::new(
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    )
}

/// The OUTER boundary of the meshable domain: PLC facets with background
/// (region 0) on one side. Internal material interfaces (both sides nonzero)
/// are excluded, so a parity ray-cast against this soup answers "inside the
/// meshable union" correctly even for nested regions and voids.
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

/// Interior subdivision points (excluding endpoints) of segment a->b at spacing.
fn subdivide(a: V3, b: V3, spacing: f64) -> Vec<V3> {
    let n = ((dist(a, b) / spacing).round() as usize).max(1);
    (1..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            std::array::from_fn(|d| a[d] + t * (b[d] - a[d]))
        })
        .collect()
}

/// Boundary edges of a patch: corner pairs appearing once among its facets.
fn patch_boundary_edges(plc: &TaggedPlc, patch: &Patch) -> Vec<(usize, usize)> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for &fi in &patch.member_indices {
        let t = plc.triangles[fi];
        let c = [t[0] as usize, t[1] as usize, t[2] as usize];
        for e in 0..3 {
            *count.entry(sorted2(c[e], c[(e + 1) % 3])).or_insert(0) += 1;
        }
    }
    count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect()
}

/// True if `p` (assumed on the patch plane) lies on a member facet (exact).
fn point_in_patch(plc: &TaggedPlc, patch: &Patch, p: V3) -> bool {
    let pp = Point3::Explicit(p);
    patch.member_indices.iter().any(|&fi| {
        let tri = tri_of(plc, plc.triangles[fi]);
        let (ax, or) = tri.projection_axis();
        tri.contains_coplanar(&pp, ax, or)
    })
}

/// The patch a tet face tiles: coplanar with the patch plane (exact `orient3d`)
/// and its centroid on a member facet (exact). `None` if interior to no patch.
fn patch_of_face(plc: &TaggedPlc, patches: &[Patch], sites: &[V3], f: [usize; 3]) -> Option<usize> {
    let fp = [
        Point3::Explicit(sites[f[0]]),
        Point3::Explicit(sites[f[1]]),
        Point3::Explicit(sites[f[2]]),
    ];
    let centroid: V3 =
        std::array::from_fn(|k| (sites[f[0]][k] + sites[f[1]][k] + sites[f[2]][k]) / 3.0);
    for (pi, p) in patches.iter().enumerate() {
        let rt = plc.triangles[p.member_indices[0]];
        let (a, b, c) = (
            Point3::Explicit(plc.vertices[rt[0] as usize]),
            Point3::Explicit(plc.vertices[rt[1] as usize]),
            Point3::Explicit(plc.vertices[rt[2] as usize]),
        );
        let coplanar = fp.iter().all(|v| orient3d(&a, &b, &c, v) == Some(Sign::Zero));
        if coplanar && point_in_patch(plc, p, centroid) {
            return Some(pi);
        }
    }
    None
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

    let patches = build_patches(plc);
    let planes: Vec<(V3, V3)> = patches.iter().map(|p| patch_plane(plc, p)).collect();
    let sep = SEPARATION_FRAC * spacing;

    // ---- stage 1: corners + feature edges (1D), fixed --------------------
    let t_surf = std::time::Instant::now();
    let nb = plc.vertices.len();
    let mut sites: Vec<V3> = plc.vertices.clone();
    let mut fixed: Vec<bool> = vec![true; nb];
    let pbe: Vec<Vec<(usize, usize)>> = patches.iter().map(|p| patch_boundary_edges(plc, p)).collect();
    let mut edge_pts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for edges in &pbe {
        for &e in edges {
            if edge_pts.contains_key(&e) {
                continue;
            }
            let idx: Vec<usize> = subdivide(plc.vertices[e.0], plc.vertices[e.1], spacing)
                .into_iter()
                .map(|p| {
                    sites.push(p);
                    fixed.push(true);
                    sites.len() - 1
                })
                .collect();
            edge_pts.insert(e, idx);
        }
    }

    // ---- stage 2: per-patch 2D Lloyd (faces), edges fixed ----------------
    for (pi, patch) in patches.iter().enumerate() {
        let (p0, n) = planes[pi];
        let drop = drop_axis(n);
        // Fixed 2D boundary: the patch's boundary corners and edge points.
        let mut bnd: Vec<[f64; 2]> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for &(a, b) in &pbe[pi] {
            for c in [a, b] {
                if seen.insert(c) {
                    bnd.push(project2(plc.vertices[c], drop));
                }
            }
            for &i in &edge_pts[&sorted2(a, b)] {
                bnd.push(project2(sites[i], drop));
            }
        }
        if bnd.len() < 3 {
            continue;
        }
        let mut lo2 = bnd[0];
        let mut hi2 = bnd[0];
        for &p in &bnd {
            for k in 0..2 {
                lo2[k] = lo2[k].min(p[k]);
                hi2[k] = hi2[k].max(p[k]);
            }
        }
        let inside2 = |uv: [f64; 2]| point_in_patch(plc, patch, lift3(uv, drop, p0, n));
        let interior = cvt_fill(&bnd, lo2, hi2, spacing, LLOYD_ITERS, inside2);
        // Lift back to the plane; keep clear of existing sites.
        let tree = Octree::build(&sites);
        for uv in interior {
            let q = lift3(uv, drop, p0, n);
            let near = tree.nearest(q).map(|j| dist(q, sites[j as usize]) < sep).unwrap_or(false);
            if !near {
                sites.push(q);
                fixed.push(true);
            }
        }
    }
    let n_surf = sites.len();
    rmlog::stage("mesh.surface", t_surf.elapsed().as_secs_f64());

    // ---- stage 3: interior volume points, free ---------------------------
    let t_seed = std::time::Instant::now();
    if spacing.is_finite() && spacing > 0.0 {
        let surf_tree = Octree::build(&sites);
        for p in bcc_lattice(lo, hi, spacing) {
            if !inside(p) {
                continue;
            }
            let near = surf_tree.nearest(p).map(|j| dist(p, sites[j as usize]) < sep).unwrap_or(false);
            if !near {
                sites.push(p);
                fixed.push(false);
            }
        }
    }
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // ---- 3D Lloyd on the volume points (surface fixed) -------------------
    let t_lloyd = std::time::Instant::now();
    for _ in 0..LLOYD_ITERS {
        if !fixed.iter().any(|&f| !f) {
            break;
        }
        let tets = delaunay_of(&sites, lo, hi);
        let mut num = vec![[0.0f64; 3]; sites.len()];
        let mut den = vec![0.0f64; sites.len()];
        for t in &tets {
            let p = [sites[t[0]], sites[t[1]], sites[t[2]], sites[t[3]]];
            let w = tet_det(p).abs();
            let c = centroid4(p);
            for &i in t {
                for k in 0..3 {
                    num[i][k] += w * c[k];
                }
                den[i] += w;
            }
        }
        let tree = Octree::build(&sites);
        for i in 0..sites.len() {
            if fixed[i] || den[i] == 0.0 {
                continue;
            }
            let tgt: V3 = std::array::from_fn(|k| num[i][k] / den[i]);
            if !inside(tgt) {
                continue;
            }
            let crowded = tree.within_radius(tgt, sep).into_iter().any(|j| j as usize != i);
            if !crowded {
                sites[i] = tgt;
            }
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());

    // ---- interface/boundary recovery -------------------------------------
    let t_recover = std::time::Instant::now();
    let eps = STRADDLE_EPS_FRAC * diag.max(1.0);
    // Recovery must insert the crossing point even when it sits near an existing
    // site (that is often exactly the point needed to kill the straddle); only
    // an actual near-duplicate would panic the Delaunay, so guard just that.
    let dup_tol = 1e-7 * diag.max(1.0);
    let mut recover_rounds = 0;
    loop {
        let tets = delaunay_of(&sites, lo, hi);
        let mut adds: Vec<(V3, usize)> = Vec::new();
        for t in &tets {
            let pv = [sites[t[0]], sites[t[1]], sites[t[2]], sites[t[3]]];
            for (pi, &(p0, n)) in planes.iter().enumerate() {
                let d: [f64; 4] = std::array::from_fn(|k| dot(sub(pv[k], p0), n));
                if !(d.iter().any(|&x| x > eps) && d.iter().any(|&x| x < -eps)) {
                    continue;
                }
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        let cross = (d[i] > eps && d[j] < -eps) || (d[j] > eps && d[i] < -eps);
                        if !cross {
                            continue;
                        }
                        let tt = d[i] / (d[i] - d[j]);
                        let raw = add(pv[i], scale(sub(pv[j], pv[i]), tt));
                        let x = project_plane(raw, p0, n);
                        if point_in_patch(plc, &patches[pi], x)
                            && sites.iter().all(|&q| dist(x, q) >= dup_tol)
                            && adds.iter().all(|&(q, _)| dist(x, q) >= dup_tol)
                        {
                            adds.push((x, pi));
                        }
                    }
                }
            }
        }
        if adds.is_empty() {
            break;
        }
        for (x, _pi) in adds {
            sites.push(x);
            fixed.push(true);
        }
        recover_rounds += 1;
        assert!(
            recover_rounds <= MAX_RECOVER_ROUNDS,
            "interface recovery did not converge in {recover_rounds} rounds"
        );
    }
    rmlog::stat("mesh.recover_rounds", recover_rounds as f64);
    rmlog::stage("mesh.recover", t_recover.elapsed().as_secs_f64());

    // ---- final triangulation, tilings, region classification --------------
    let t_classify = std::time::Instant::now();
    let all_tets = delaunay_of(&sites, lo, hi);
    let mut face_owners: HashMap<[usize; 3], Vec<u32>> = HashMap::new();
    for (ti, t) in all_tets.iter().enumerate() {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            face_owners.entry(f).or_default().push(ti as u32);
        }
    }
    // A face can tile a patch only if all three vertices are fixed surface sites.
    let mut tilings: Vec<Vec<[usize; 3]>> = vec![Vec::new(); patches.len()];
    for key in face_owners.keys() {
        if !key.iter().all(|&v| fixed[v]) {
            continue;
        }
        if let Some(pi) = patch_of_face(plc, &patches, &sites, *key) {
            tilings[pi].push(*key);
        }
    }
    let region = classify_tet_regions(&sites, &all_tets, &patches, &tilings, &face_owners, (lo, hi));
    let mut kept: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for (t, &r) in all_tets.iter().zip(&region) {
        if r != 0 {
            kept.push(*t);
            tet_regions.push(RegionTag(r));
        }
    }
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

    let mesh = TetMesh {
        points: sites,
        tets: kept,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        abandoned_patches: Vec::new(),
        plc_points: nb,
    };

    let q = quality_stats(&mesh);
    rmlog::stat("mesh.points", mesh.points.len() as f64);
    rmlog::stat("mesh.tets", mesh.tets.len() as f64);
    rmlog::stat("mesh.surface_points", n_surf as f64);
    rmlog::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    rmlog::stage("mesh.total", t_start.elapsed().as_secs_f64());
    mesh
}

/// A fresh Delaunay over the current sites; returns real tets in site indices.
fn delaunay_of(sites: &[V3], lo: V3, hi: V3) -> Vec<[usize; 4]> {
    let mut db = DelaunayBuilder::enclosing(lo, hi);
    for &p in sites {
        db.insert(p);
    }
    db.tets()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::Zero;
    use rapidmesh_geom::{solid_box, Scene};
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

    fn max_edge(m: &TetMesh) -> f64 {
        let mut e = 0.0_f64;
        for t in &m.tets {
            for i in 0..4 {
                for j in i + 1..4 {
                    e = e.max(dist(m.points[t[i]], m.points[t[j]]));
                }
            }
        }
        e
    }

    // WP4/WP5 correctness gates: EXACT region volume + watertight. Element
    // quality (min dihedral, edge bound) is the WP8 optimizer's job.

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

    #[test]
    fn sized_box_refines_boundary() {
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams { maxh: 0.6, ..Default::default() });
        assert_eq!(region_volume6(&mesh, r.0), rat(36.0), "exact volume");
        assert!(watertight(&mesh));
        assert!(max_edge(&mesh) <= 2.0 * 0.6, "boundary not refined: {}", max_edge(&mesh));
    }
}
