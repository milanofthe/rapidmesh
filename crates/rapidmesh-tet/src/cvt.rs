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
//! the boundary as tet faces); regions are assigned by raytracing
//! (`classify_tet_regions`). Out-of-domain volume moves are rejected (kept).

use crate::conform::{build_patches, classify_tet_regions, quality_stats, MeshParams, Patch, SurfaceFace, TetMesh};
use crate::delaunay::DelaunayBuilder;
use crate::seed::{bcc_lattice, SizingField};
use crate::site::{Carrier, Site};
use crate::spatial::Octree;
use crate::surf2d::cvt_fill;
use rapidmesh_csg::classify::{point_inside_solid, TriBoxes};
use rapidmesh_csg::Tri;
use rapidmesh_exact::Point3;
use rapidmesh_geom::{RegionTag, TaggedPlc};
use std::collections::HashMap;

type V3 = [f64; 3];

/// The four vertex-index triples spanning a tet's faces.
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

/// Volume (3D) Lloyd relaxation passes.
const LLOYD_ITERS: usize = 8;
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
const SURFACE_OVERSAMPLE: f64 = 0.5;
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
    let sep = SEPARATION_FRAC * spacing;
    let surf_spacing = spacing * SURFACE_OVERSAMPLE;
    let surf_sep = SEPARATION_FRAC * surf_spacing;

    // ---- stage 1: corners + feature edges (1D), fixed --------------------
    let t_surf = std::time::Instant::now();
    let nb = plc.vertices.len();
    let mut sites: Vec<Site> = plc.vertices.iter().map(|&v| Site::vertex(v)).collect();
    let pbe: Vec<Vec<(usize, usize)>> = patches.iter().map(|p| patch_boundary_edges(plc, p)).collect();
    let mut edge_pts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for edges in &pbe {
        for &e in edges {
            if edge_pts.contains_key(&e) {
                continue;
            }
            let (va, vb) = (plc.vertices[e.0], plc.vertices[e.1]);
            let n = ((dist(va, vb) / surf_spacing).round() as usize).max(1);
            let idx: Vec<usize> = (1..n)
                .map(|k| {
                    sites.push(Site::on_edge(va, vb, k as f64 / n as f64));
                    sites.len() - 1
                })
                .collect();
            edge_pts.insert(e, idx);
        }
    }

    // ---- stage 2: per-patch 2D Lloyd (faces), edges fixed ----------------
    for (pi, patch) in patches.iter().enumerate() {
        let (p0, n) = patch_plane(plc, patch);
        let drop = drop_axis(n);
        // Fixed 2D boundary: the patch's boundary corners and edge points.
        let mut bnd: Vec<[f64; 2]> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
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
            continue;
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
        let interior = cvt_fill(&bnd, lo2, hi2, surf_spacing, SURF_LLOYD_ITERS, inside2);
        let pos = positions(&sites);
        let tree = Octree::build(&pos);
        for uv in interior {
            let q = lift3(uv, drop, p0, n);
            let near = tree.nearest(q).map(|j| dist(q, pos[j as usize]) < surf_sep).unwrap_or(false);
            if !near {
                sites.push(Site::on_plane(p0, n, q));
            }
        }
    }
    let n_surf = sites.len();
    rmlog::stage("mesh.surface", t_surf.elapsed().as_secs_f64());

    // ---- stage 3: interior volume points, free ---------------------------
    let t_seed = std::time::Instant::now();
    if spacing.is_finite() && spacing > 0.0 {
        let pos = positions(&sites);
        let surf_tree = Octree::build(&pos);
        for p in bcc_lattice(lo, hi, spacing) {
            if !inside(p) {
                continue;
            }
            let near = surf_tree.nearest(p).map(|j| dist(p, pos[j as usize]) < sep).unwrap_or(false);
            if !near {
                sites.push(Site::free(p));
            }
        }
    }
    rmlog::stage("mesh.seed", t_seed.elapsed().as_secs_f64());

    // ---- 3D Lloyd on the volume sites (surface fixed) --------------------
    // The surface is fixed now, so its octree (for mirror-in) is built once.
    let surf_pos: Vec<V3> = sites[..n_surf].iter().map(|s| s.pos()).collect();
    let surf_tree = Octree::build(&surf_pos);
    let t_lloyd = std::time::Instant::now();
    for _ in 0..LLOYD_ITERS {
        if !sites.iter().any(|s| s.is_volume()) {
            break;
        }
        let pos = positions(&sites);
        let tets = delaunay_of(&sites, lo, hi);
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
        let tree = Octree::build(&pos);
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
            let crowded = tree.within_radius(tgt, sep).into_iter().any(|j| j as usize != i);
            if !crowded {
                sites[i].move_to(tgt);
            }
        }
    }
    rmlog::stage("mesh.lloyd", t_lloyd.elapsed().as_secs_f64());

    // ---- final triangulation, tilings, region classification --------------
    let t_classify = std::time::Instant::now();
    let all_tets = delaunay_of(&sites, lo, hi);
    let pts = positions(&sites);
    let mut face_owners: HashMap<[usize; 3], Vec<u32>> = HashMap::new();
    for (ti, t) in all_tets.iter().enumerate() {
        for fv in &TET_FACES {
            let mut f = [t[fv[0]], t[fv[1]], t[fv[2]]];
            f.sort_unstable();
            face_owners.entry(f).or_default().push(ti as u32);
        }
    }
    // A face tiles a patch only if all three vertices are surface sites.
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
    let region = classify_tet_regions(&pts, &all_tets, &patches, &tilings, &face_owners, (lo, hi));
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
        points: pts,
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

/// A fresh Delaunay over the current site positions; returns real tets in site
/// indices. Points are inserted in Morton (space-filling) order so the builder's
/// location walk stays short (near-linear construction); the returned tets are
/// remapped from insertion order back to site indices.
fn delaunay_of(sites: &[Site], lo: V3, hi: V3) -> Vec<[usize; 4]> {
    let pos: Vec<V3> = sites.iter().map(|s| s.pos()).collect();
    let order = crate::spatial::morton_order(&pos);
    let mut db = DelaunayBuilder::enclosing(lo, hi);
    for &si in &order {
        db.insert(pos[si]);
    }
    // Insertion index k corresponds to site `order[k]`.
    db.tets()
        .into_iter()
        .map(|t| std::array::from_fn(|j| order[t[j]]))
        .collect()
}

/// Reflection of `p` across the plane (`p0`, unit `n`).
fn reflect(p: V3, p0: V3, n: V3) -> V3 {
    sub(p, scale(n, 2.0 * dot(sub(p, p0), n)))
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
}
