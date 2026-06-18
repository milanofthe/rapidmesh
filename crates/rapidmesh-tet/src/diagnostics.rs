//! Mesh diagnostics: quality metrics plus spatially-located defects.
//!
//! This is the instrument that turns the corpus run into a MAP rather than a
//! pass/fail light, and -- crucially -- it locates WHERE the mesh is wrong, which
//! is exactly the trigger the curved-surface refinement (straddler repair) needs:
//! a straddler is a boundary face whose vertex is OFF its analytic surface (an
//! interior point leaked into the boundary), and its position is where a surface
//! point must be inserted. Quality (dihedral histogram, slivers, radius-edge) and
//! conformity (non-manifold edges, surface deviation, region volumes) round out
//! the picture so every later step is measurable and visible.

use crate::conform::TetMesh;
use crate::project::closest_on_surface;
use rapidmesh_geom::SurfaceKind;

type V3 = [f64; 3];

/// The kind of a located mesh defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectKind {
    /// A tet whose smallest dihedral angle is below [`SLIVER_DEG`] (poorly
    /// conditioned; `value` = the angle in degrees).
    Sliver,
    /// A surface edge not shared by exactly two faces (a boundary leak /
    /// non-manifold incidence; `value` = the incidence count).
    NonManifoldEdge,
    /// A boundary face with a vertex OFF its analytic surface: an interior point
    /// leaked into the boundary (the restricted Delaunay under-sampled the
    /// surface). `value` = the off-surface distance. The repair site for refinement.
    Straddler,
}

/// A defect with its 3D location and a severity `value` (units per [`DefectKind`]).
#[derive(Debug, Clone)]
pub struct Defect {
    pub kind: DefectKind,
    pub pos: V3,
    pub value: f64,
}

/// Quality + conformity diagnostics of a tet mesh, with located defects.
#[derive(Debug, Clone)]
pub struct MeshDiagnostics {
    pub n_tets: usize,
    pub n_points: usize,
    pub n_faces: usize,
    /// Smallest dihedral angle over all tets (degrees).
    pub min_dihedral_deg: f64,
    /// Mean of the per-tet smallest dihedral angle (degrees).
    pub mean_min_dihedral_deg: f64,
    /// Count of per-tet min-dihedral in each 10-degree bin `[0,10),...,[170,180)`.
    pub dihedral_histogram: [usize; 18],
    /// Tets with min dihedral below [`SLIVER_DEG`].
    pub n_slivers: usize,
    /// Largest circumradius / shortest-edge ratio.
    pub max_radius_edge: f64,
    /// Every surface edge is shared by exactly two faces.
    pub watertight: bool,
    pub n_nonmanifold_edges: usize,
    pub n_straddlers: usize,
    /// Largest distance of a curved boundary face's centroid from its analytic
    /// surface (the chord sagitta -- the realised geometric accuracy vs `tol`).
    pub max_surface_deviation: f64,
    /// Total tet volume per region, ascending region tag.
    pub region_volumes: Vec<(u32, f64)>,
    /// Located defects (slivers, non-manifold edges, straddlers).
    pub defects: Vec<Defect>,
}

/// Tets below this smallest-dihedral angle (degrees) are counted as slivers.
pub const SLIVER_DEG: f64 = 10.0;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dist(a: V3, b: V3) -> f64 {
    sub(a, b).iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn centroid(ps: &[V3]) -> V3 {
    let n = ps.len() as f64;
    std::array::from_fn(|k| ps.iter().map(|p| p[k]).sum::<f64>() / n)
}

/// Smallest dihedral angle (degrees) of one tet, or `f64::MAX` if degenerate.
pub(crate) fn tet_min_dihedral(p: [V3; 4]) -> f64 {
    let mut m = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            let others: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
            let (a, b) = (p[i], p[j]);
            let tlen = dist(a, b);
            if tlen == 0.0 {
                continue;
            }
            let tv: V3 = std::array::from_fn(|k| (b[k] - a[k]) / tlen);
            let perp = |q: V3| -> V3 {
                let w: V3 = std::array::from_fn(|k| q[k] - a[k]);
                let s: f64 = (0..3).map(|k| w[k] * tv[k]).sum();
                std::array::from_fn(|k| w[k] - s * tv[k])
            };
            let (u, v) = (perp(p[others[0]]), perp(p[others[1]]));
            let (nu, nv) = (dist(u, [0.0; 3]), dist(v, [0.0; 3]));
            if nu * nv == 0.0 {
                continue;
            }
            let cosang = ((0..3).map(|k| u[k] * v[k]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
            m = m.min(cosang.acos().to_degrees());
        }
    }
    m
}

fn tet_volume(p: [V3; 4]) -> f64 {
    let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
    let (ab, ac, ad) = (sub(b, a), sub(c, a), sub(d, a));
    let cr = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    (ab[0] * cr[0] + ab[1] * cr[1] + ab[2] * cr[2]).abs() / 6.0
}

/// Circumradius / shortest-edge of one tet (`None` if degenerate).
fn radius_edge(p: [V3; 4]) -> Option<f64> {
    // Circumcenter via the linear system on squared-distance differences.
    let a = p[0];
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for i in 0..3 {
        let q = p[i + 1];
        for k in 0..3 {
            m[i][k] = 2.0 * (q[k] - a[k]);
        }
        rhs[i] = (0..3).map(|k| q[k] * q[k] - a[k] * a[k]).sum();
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let mut cc = [0.0f64; 3];
    for k in 0..3 {
        let mut mk = m;
        for r in 0..3 {
            mk[r][k] = rhs[r];
        }
        let dk = mk[0][0] * (mk[1][1] * mk[2][2] - mk[1][2] * mk[2][1])
            - mk[0][1] * (mk[1][0] * mk[2][2] - mk[1][2] * mk[2][0])
            + mk[0][2] * (mk[1][0] * mk[2][1] - mk[1][1] * mk[2][0]);
        cc[k] = dk / det;
    }
    let r = dist(cc, a);
    let mut lmin = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            lmin = lmin.min(dist(p[i], p[j]));
        }
    }
    (lmin > 0.0).then(|| r / lmin)
}

/// Computes quality + conformity diagnostics with located defects.
pub fn diagnose(mesh: &TetMesh) -> MeshDiagnostics {
    let pt = |i: usize| mesh.points[i];

    // ---- quality: dihedral histogram, slivers, radius-edge ----------------
    let mut hist = [0usize; 18];
    let mut min_dih = f64::MAX;
    let mut sum_dih = 0.0;
    let mut max_re = 0.0f64;
    let mut n_slivers = 0usize;
    let mut defects: Vec<Defect> = Vec::new();
    let mut region_vol: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    for (ti, t) in mesh.tets.iter().enumerate() {
        let p = [pt(t[0]), pt(t[1]), pt(t[2]), pt(t[3])];
        let md = tet_min_dihedral(p);
        if md.is_finite() {
            min_dih = min_dih.min(md);
            sum_dih += md;
            let bin = ((md / 10.0).floor() as usize).min(17);
            hist[bin] += 1;
            if md < SLIVER_DEG {
                n_slivers += 1;
                defects.push(Defect { kind: DefectKind::Sliver, pos: centroid(&p), value: md });
            }
        }
        if let Some(re) = radius_edge(p) {
            max_re = max_re.max(re);
        }
        *region_vol.entry(mesh.tet_regions[ti].0).or_insert(0.0) += tet_volume(p);
    }
    let mean_dih = if mesh.tets.is_empty() { 0.0 } else { sum_dih / mesh.tets.len() as f64 };

    // ---- conformity: non-manifold surface edges ---------------------------
    let mut edge_count: std::collections::HashMap<(usize, usize), (u32, V3)> =
        std::collections::HashMap::new();
    for f in &mesh.faces {
        for k in 0..3 {
            let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            let mid = centroid(&[pt(a), pt(b)]);
            let e = edge_count.entry(key).or_insert((0, mid));
            e.0 += 1;
        }
    }
    let mut n_nonmanifold = 0usize;
    for (_, &(cnt, mid)) in &edge_count {
        if cnt != 2 {
            n_nonmanifold += 1;
            defects.push(Defect { kind: DefectKind::NonManifoldEdge, pos: mid, value: cnt as f64 });
        }
    }

    // ---- conformity: straddlers + surface deviation (curved faces) --------
    // A boundary face whose vertex sits far OFF its analytic surface means an
    // interior point leaked into the boundary (under-sampling). The chord sagitta
    // (centroid off-surface) is the realised geometric accuracy.
    let mut max_dev = 0.0f64;
    let mut n_straddlers = 0usize;
    for f in &mesh.faces {
        let kind = &mesh.surfaces[f.surface as usize];
        if matches!(kind, SurfaceKind::Plane) {
            continue; // planar faces are exact; deviation is 0
        }
        let v = [pt(f.tri[0]), pt(f.tri[1]), pt(f.tri[2])];
        let longest = (0..3).map(|k| dist(v[k], v[(k + 1) % 3])).fold(0.0f64, f64::max);
        // straddler: a VERTEX off the surface (surface points are on it, so any
        // sizeable offset is an interior point that leaked in).
        let vmax_off = v.iter().map(|&q| dist(q, closest_on_surface(kind, q))).fold(0.0f64, f64::max);
        if longest > 0.0 && vmax_off > 0.25 * longest {
            n_straddlers += 1;
            defects.push(Defect { kind: DefectKind::Straddler, pos: centroid(&v), value: vmax_off });
        }
        // accuracy: chord sagitta = centroid off-surface.
        let c = centroid(&v);
        max_dev = max_dev.max(dist(c, closest_on_surface(kind, c)));
    }

    MeshDiagnostics {
        n_tets: mesh.tets.len(),
        n_points: mesh.points.len(),
        n_faces: mesh.faces.len(),
        min_dihedral_deg: if min_dih.is_finite() { min_dih } else { 0.0 },
        mean_min_dihedral_deg: mean_dih,
        dihedral_histogram: hist,
        n_slivers,
        max_radius_edge: max_re,
        watertight: n_nonmanifold == 0,
        n_nonmanifold_edges: n_nonmanifold,
        n_straddlers,
        max_surface_deviation: max_dev,
        region_volumes: region_vol.into_iter().collect(),
        defects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    #[test]
    fn box_is_clean_and_watertight() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let m = crate::cvt::mesh_cdt(&plc, &crate::conform::MeshParams { maxh: 1.0, ..Default::default() });
        let d = diagnose(&m);
        assert!(d.watertight, "box must be watertight ({} non-manifold)", d.n_nonmanifold_edges);
        assert_eq!(d.n_straddlers, 0, "a box has no curved straddlers");
        assert!(d.max_surface_deviation < 1e-9, "planar faces have zero deviation");
        assert!(d.min_dihedral_deg > 0.0, "well-defined dihedral");
        let vol: f64 = d.region_volumes.iter().map(|&(_, v)| v).sum();
        assert!((vol - 24.0).abs() < 1e-6, "box volume 24, got {vol}");
    }

    #[test]
    fn sphere_is_watertight_and_on_surface() {
        use rapidmesh_geom::sphere;
        let mut scene = Scene::new();
        scene.add_solid(sphere([0.0, 0.0, 0.0], 1.0, 24, 12));
        let plc = scene.assemble();
        let m = crate::cvt::mesh_cdt(&plc, &crate::conform::MeshParams { maxh: 0.4, tol_surf: 1e-2, ..Default::default() });
        let d = diagnose(&m);
        assert!(d.watertight, "sphere must be watertight ({} non-manifold)", d.n_nonmanifold_edges);
        assert_eq!(d.n_straddlers, 0, "a well-sampled sphere has no straddlers");
        // chord sagitta of a radius-1 sphere at this density is small but nonzero
        assert!(d.max_surface_deviation < 0.1, "sphere deviation within tol, got {}", d.max_surface_deviation);
    }
}
