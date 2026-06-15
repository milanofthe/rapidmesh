//! CVT (centroidal Voronoi) tetrahedral meshing of a tagged PLC.
//!
//! Replaces the constrained-Delaunay + Steiner boundary recovery of the old
//! pipeline. The exact CSG arrangement (the `TaggedPlc`) is untouched; this
//! stage fills it with a variational (Lloyd-relaxed) tetrahedralization.
//!
//! ONE density-driven constrained CVT loop (the WP0 spike validated this keeps
//! exact planar conformity). Every site carries a constraint that pins it to its
//! geometric carrier, so the boundary stays exact while the interior relaxes:
//! corners are Fixed, feature-edge points move only along their edge (shared
//! between the two adjacent patches so faces agree on the edge), face points
//! move only in the patch plane, and interior points are Free. The single 3D
//! Lloyd loop runs on the exact incremental Delaunay (`DelaunayBuilder`); all
//! geometric decisions use the exact predicates (`orient2d`/`orient3d`,
//! `Tri::contains_coplanar`, `point_inside_solid`). No hand-rolled float
//! geometry kernel.
//!
//! Scope landed: single non-background region with a convex boundary. Multi-
//! region shared interfaces (WP5), non-convex / curved boundary (WP6), explicit
//! grading via density (WP7), quality post-pass (WP8) build on this.

use crate::conform::{build_patches, quality_stats, MeshParams, Patch, SurfaceFace, TetMesh};
use crate::delaunay::DelaunayBuilder;
use crate::seed::{bcc_lattice, SizingField};
use crate::spatial::Octree;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{RegionTag, TaggedPlc};
use std::collections::{HashMap, HashSet};

type V3 = [f64; 3];

/// Lloyd relaxation passes.
const LLOYD_ITERS: usize = 12;
/// Bounding-box subdivisions for the default spacing when no `maxh` is given.
const DEFAULT_SUBDIV: f64 = 8.0;
/// Per-triangle bounding-box pad for the inside test, fraction of the diagonal.
const BOX_PAD_FRAC: f64 = 1e-6;
/// Minimum separation of a seeded/moved site from any other, fraction of spacing.
const SEPARATION_FRAC: f64 = 0.45;

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

/// The geometric carrier a site is pinned to during relaxation.
#[derive(Clone)]
enum Con {
    Fixed,
    Edge(V3, V3),
    Plane(V3, V3), // (point on plane, unit normal)
    Free,
}

impl Con {
    /// Constrains a proposed target to the carrier. `Fixed` returns `None`.
    fn apply(&self, c: V3) -> Option<V3> {
        match *self {
            Con::Fixed => None,
            Con::Free => Some(c),
            Con::Edge(a, b) => {
                let ab = sub(b, a);
                let t = (dot(sub(c, a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
                Some(add(a, scale(ab, t)))
            }
            Con::Plane(p0, n) => {
                let d = dot(sub(c, p0), n);
                Some(sub(c, scale(n, d)))
            }
        }
    }
}

fn boundary_soup(plc: &TaggedPlc) -> Vec<Tri> {
    plc.triangles
        .iter()
        .map(|t| {
            Tri::new(
                plc.vertices[t[0] as usize],
                plc.vertices[t[1] as usize],
                plc.vertices[t[2] as usize],
            )
        })
        .collect()
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

/// Feature edges of the PLC: corner pairs that bound a patch (appear once among
/// a patch's facets). Shared between the two patches meeting at the edge.
fn feature_edges(plc: &TaggedPlc, patches: &[Patch]) -> Vec<(usize, usize)> {
    let mut feats: HashSet<(usize, usize)> = HashSet::new();
    for patch in patches {
        let mut count: HashMap<(usize, usize), usize> = HashMap::new();
        for &fi in &patch.member_indices {
            let t = plc.triangles[fi];
            let c = [t[0] as usize, t[1] as usize, t[2] as usize];
            for e in 0..3 {
                *count.entry(sorted2(c[e], c[(e + 1) % 3])).or_insert(0) += 1;
            }
        }
        for (&e, &c) in &count {
            if c == 1 {
                feats.insert(e);
            }
        }
    }
    let mut v: Vec<(usize, usize)> = feats.into_iter().collect();
    v.sort_unstable();
    v
}

/// Meshes a tagged PLC into a region-tagged tet mesh by constrained CVT.
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
    let primary = regions.first().copied().unwrap_or(0);
    let field = SizingField::new(params);
    let cap = field.finest_cap(&regions);
    let spacing = if cap.is_finite() && cap > 0.0 { cap } else { diag / DEFAULT_SUBDIV };

    let soup = boundary_soup(plc);
    let boxes = TriBoxes::build(&soup, BOX_PAD_FRAC * diag.max(1.0));
    let inside = |p: V3| point_inside_solid(&Point3::Explicit(p), p, &soup, &boxes, (lo, hi));

    let patches = build_patches(plc);

    // ---- seed sites with carriers ----------------------------------------
    let t_seed = std::time::Instant::now();
    let nb = plc.vertices.len();
    let mut sites: Vec<V3> = plc.vertices.clone();
    let mut cons: Vec<Con> = vec![Con::Fixed; nb]; // corners are fixed

    let sep2 = (SEPARATION_FRAC * spacing).powi(2);
    let push_site = |sites: &mut Vec<V3>, cons: &mut Vec<Con>, p: V3, con: Con| {
        if sites.iter().all(|&q| dist(p, q).powi(2) >= sep2) {
            sites.push(p);
            cons.push(con);
        }
    };

    // Feature-edge points, pinned to the edge line (shared across patches).
    for (a, b) in feature_edges(plc, &patches) {
        let (pa, pb) = (plc.vertices[a], plc.vertices[b]);
        for p in subdivide(pa, pb, spacing) {
            push_site(&mut sites, &mut cons, p, Con::Edge(pa, pb));
        }
    }

    // Face points: scatter a grid inside each patch (exact containment via the
    // patch's facet triangles), pinned to the patch plane.
    if spacing.is_finite() && spacing > 0.0 {
        for patch in &patches {
            let (p0, n) = patch_plane(plc, patch);
            let drop = drop_axis(n);
            let (k1, k2) = kept_axes(drop);
            // Patch facets as Tris with their exact projection.
            let facets: Vec<(Tri, rapidmesh_exact::Axis, Sign)> = patch
                .member_indices
                .iter()
                .map(|&fi| {
                    let t = plc.triangles[fi];
                    let tri = Tri::new(
                        plc.vertices[t[0] as usize],
                        plc.vertices[t[1] as usize],
                        plc.vertices[t[2] as usize],
                    );
                    let (ax, or) = tri.projection_axis();
                    (tri, ax, or)
                })
                .collect();
            let mut blo = [f64::MAX; 2];
            let mut bh = [f64::MIN; 2];
            for &fi in &patch.member_indices {
                let t = plc.triangles[fi];
                for &vi in &[t[0], t[1], t[2]] {
                    let v = plc.vertices[vi as usize];
                    blo[0] = blo[0].min(v[k1]);
                    blo[1] = blo[1].min(v[k2]);
                    bh[0] = bh[0].max(v[k1]);
                    bh[1] = bh[1].max(v[k2]);
                }
            }
            let nx = (((bh[0] - blo[0]) / spacing).ceil() as usize).max(1);
            let ny = (((bh[1] - blo[1]) / spacing).ceil() as usize).max(1);
            for i in 1..nx {
                for j in 1..ny {
                    let uv = [blo[0] + i as f64 * spacing, blo[1] + j as f64 * spacing];
                    let q = lift3(uv, drop, p0, n);
                    let qp = Point3::Explicit(q);
                    let inside_patch = facets
                        .iter()
                        .any(|(tri, ax, or)| tri.contains_coplanar(&qp, *ax, *or));
                    if inside_patch {
                        push_site(&mut sites, &mut cons, q, Con::Plane(p0, n));
                    }
                }
            }
        }
    }
    let n_surf = sites.len();

    // Interior volume points: BCC inside the domain, clear of the surface.
    if spacing.is_finite() && spacing > 0.0 {
        let surf_tree = Octree::build(&sites);
        let sep = SEPARATION_FRAC * spacing;
        for p in bcc_lattice(lo, hi, spacing) {
            if !inside(p) {
                continue;
            }
            let near = surf_tree
                .nearest(p)
                .map(|j| dist(p, sites[j as usize]) < sep)
                .unwrap_or(false);
            if !near {
                sites.push(p);
                cons.push(Con::Free);
            }
        }
    }
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // ---- constrained Lloyd relaxation ------------------------------------
    let t_lloyd = std::time::Instant::now();
    let sep = SEPARATION_FRAC * spacing;
    for _ in 0..LLOYD_ITERS {
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
            if den[i] == 0.0 {
                continue;
            }
            let raw: V3 = std::array::from_fn(|k| num[i][k] / den[i]);
            let tgt = match cons[i].apply(raw) {
                Some(t) => t,
                None => continue, // Fixed
            };
            // Interior moves must stay in the domain; all moves must stay clear
            // of other sites (no collapse / sliver seeding).
            if matches!(cons[i], Con::Free) && !inside(tgt) {
                continue;
            }
            let crowded = tree
                .within_radius(tgt, sep)
                .into_iter()
                .any(|j| j as usize != i);
            if !crowded {
                sites[i] = tgt;
            }
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());

    // ---- final triangulation + single-region classification ---------------
    let t_classify = std::time::Instant::now();
    let all_tets = delaunay_of(&sites, lo, hi);
    let mut kept: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    for t in &all_tets {
        let c = centroid4([sites[t[0]], sites[t[1]], sites[t[2]], sites[t[3]]]);
        if inside(c) {
            kept.push(*t);
            tet_regions.push(RegionTag(primary));
        }
    }
    rmlog::stage("mesh.classify", t_classify.elapsed().as_secs_f64());

    let faces = tag_boundary_faces(plc, &patches, &sites, &kept, n_surf);

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

/// Boundary faces (shared by exactly one kept tet), each tagged to the PLC patch
/// whose plane contains it (exact `orient3d` coplanarity). Face vertices must be
/// surface sites (index < n_surf).
fn tag_boundary_faces(
    plc: &TaggedPlc,
    patches: &[Patch],
    sites: &[V3],
    kept: &[[usize; 4]],
    n_surf: usize,
) -> Vec<SurfaceFace> {
    const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    let mut owners: HashMap<[usize; 3], usize> = HashMap::new();
    for t in kept {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            *owners.entry(f).or_default() += 1;
        }
    }
    let patch_tri = |p: &Patch| -> [Point3; 3] {
        let t = plc.triangles[p.member_indices[0]];
        [
            Point3::Explicit(plc.vertices[t[0] as usize]),
            Point3::Explicit(plc.vertices[t[1] as usize]),
            Point3::Explicit(plc.vertices[t[2] as usize]),
        ]
    };
    let mut out = Vec::new();
    for (f, &c) in &owners {
        if c != 1 || f.iter().any(|&v| v >= n_surf) {
            continue;
        }
        let mut chosen: Option<usize> = None;
        for (pi, p) in patches.iter().enumerate() {
            let [a, b, cc] = patch_tri(p);
            if f.iter().all(|&v| {
                orient3d(&a, &b, &cc, &Point3::Explicit(sites[v])) == Some(Sign::Zero)
            }) {
                chosen = Some(pi);
                break;
            }
        }
        if let Some(pi) = chosen {
            let p = &patches[pi];
            out.push(SurfaceFace {
                tri: *f,
                face_tag: p.face_tag,
                regions: p.regions,
                patch: pi as u32,
                surface: p.surface,
            });
        }
    }
    out
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
        const TF: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
        let mut faces: HashMap<[usize; 3], usize> = HashMap::new();
        for t in &m.tets {
            for fv in &TF {
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

    // Correctness gates for the WP4 scope: EXACT region volume and a watertight
    // boundary. Element quality (min dihedral, edge bound) is the WP8 optimizer's
    // job and is gated there, not here.

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
    fn sized_box_refines_boundary() {
        // The density-driven seeding refines the boundary: the edge length is in
        // the ballpark of the target (a loose pre-optimizer bound).
        let mut scene = Scene::new();
        let r = scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]));
        let plc = scene.assemble();
        let mesh = mesh(&plc, &MeshParams { maxh: 0.6, ..Default::default() });
        assert_eq!(region_volume6(&mesh, r.0), rat(36.0), "exact volume");
        assert!(watertight(&mesh));
        assert!(max_edge(&mesh) <= 2.0 * 0.6, "boundary not refined: {}", max_edge(&mesh));
    }
}
