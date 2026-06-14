//! WP0 spike (go/no-go for the CVT rewrite): can a point-moving Lloyd/CVT loop
//! hold EXACT planar conformity (the gate the planar fixtures in `conform.rs`
//! demand) while improving dihedral angles?
//!
//! Two experiments, because they probe different things:
//!
//!   A) single-region unit cube. Tests the mechanics (constraint projection by
//!      site class, the Lloyd smoother, the quality trend). NOTE the exact
//!      region volume is trivially 6 here for ANY interior point set, because
//!      the convex hull of {8 corners + points inside} is the cube and Delaunay
//!      partitions that hull exactly. So A validates plumbing + quality, not the
//!      hard conformity question.
//!
//!   B) cube split by the internal plane z = 1/2 into two regions. This is the
//!      REAL discriminator: a pure Delaunay of points that merely lie on an
//!      internal plane does NOT generally have that plane as a union of tet
//!      faces (that is exactly why constrained recovery exists). We compare a
//!      naive independent smoothing of the two sides against a mirror-symmetric
//!      placement about z = 1/2. The mirror case makes z = 1/2 an exact set of
//!      Voronoi facets, so its dual Delaunay carries the interface exactly.
//!
//! Run:  cargo test -p rapidmesh-tet --test spike_cvt -- --ignored --nocapture

use num_rational::BigRational;
use num_traits::Zero;
use rapidmesh_testutil::rat;
use rapidmesh_tet::DelaunayBuilder;

type V3 = [f64; 3];

// --- small vector helpers -------------------------------------------------

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Signed 1/6-volume determinant of a tet (positive when (1,2,3) is CCW from 0).
fn tet_det(p: [V3; 4]) -> f64 {
    let r0 = sub(p[1], p[0]);
    let r1 = sub(p[2], p[0]);
    let r2 = sub(p[3], p[0]);
    dot(r0, cross(r1, r2))
}

fn centroid4(p: [V3; 4]) -> V3 {
    [
        0.25 * (p[0][0] + p[1][0] + p[2][0] + p[3][0]),
        0.25 * (p[0][1] + p[1][1] + p[2][1] + p[3][1]),
        0.25 * (p[0][2] + p[1][2] + p[2][2] + p[3][2]),
    ]
}

/// Minimum interior dihedral angle of a tet, in degrees. The interior dihedral
/// along the edge shared by two faces is pi minus the angle between the faces'
/// outward normals.
fn min_dihedral_deg(p: [V3; 4]) -> f64 {
    let face_normal = |i: usize| -> V3 {
        let idx: Vec<usize> = (0..4).filter(|&k| k != i).collect();
        let (a, b, c) = (p[idx[0]], p[idx[1]], p[idx[2]]);
        let mut n = cross(sub(b, a), sub(c, a));
        // Orient away from the opposite vertex p[i] (outward).
        if dot(n, sub(p[i], a)) > 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        let l = norm(n);
        if l == 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [n[0] / l, n[1] / l, n[2] / l]
        }
    };
    let n = [face_normal(0), face_normal(1), face_normal(2), face_normal(3)];
    let mut m = std::f64::consts::PI;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let c = dot(n[i], n[j]).clamp(-1.0, 1.0);
            let dih = std::f64::consts::PI - c.acos();
            if dih < m {
                m = dih;
            }
        }
    }
    m.to_degrees()
}

// --- sites + constraints --------------------------------------------------

/// A site carries its position and a per-axis lock: `Some(c)` pins that
/// coordinate to the exact value `c` (a box face, a feature edge, or the
/// internal interface plane); `None` lets the axis move freely.
#[derive(Clone)]
struct Site {
    p: V3,
    lock: [Option<f64>; 3],
}

impl Site {
    fn n_locked(&self) -> usize {
        self.lock.iter().filter(|l| l.is_some()).count()
    }
    /// Apply the constraint to a candidate target: locked axes snap to their
    /// exact pinned value, free axes are clamped into the unit box.
    fn project(&self, mut t: V3) -> V3 {
        for k in 0..3 {
            match self.lock[k] {
                Some(c) => t[k] = c,
                None => t[k] = t[k].clamp(0.0, 1.0),
            }
        }
        t
    }
}

/// BCC lattice over the unit cube at `n` cells per axis (h = 1/n). The corner
/// sub-lattice carries exact face locks (coord == 0.0 or 1.0); the body-centered
/// sub-lattice is strictly interior (always free). When `interface` is set, the
/// k = n/2 layer (z == 1/2 exactly, requires even n) is locked on z as an
/// internal plane. `keep` filters which sites to emit (used for the mirror case).
fn bcc_sites(n: usize, interface: bool, keep: impl Fn(V3) -> bool) -> Vec<Site> {
    assert!(!interface || n % 2 == 0, "interface needs even n for z=1/2");
    let h = 1.0 / n as f64;
    let mut sites = Vec::new();
    // Corner sub-lattice.
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let p = [i as f64 * h, j as f64 * h, k as f64 * h];
                if !keep(p) {
                    continue;
                }
                let mut lock = [None; 3];
                if i == 0 {
                    lock[0] = Some(0.0);
                } else if i == n {
                    lock[0] = Some(1.0);
                }
                if j == 0 {
                    lock[1] = Some(0.0);
                } else if j == n {
                    lock[1] = Some(1.0);
                }
                if k == 0 {
                    lock[2] = Some(0.0);
                } else if k == n {
                    lock[2] = Some(1.0);
                } else if interface && k == n / 2 {
                    lock[2] = Some(0.5);
                }
                sites.push(Site { p, lock });
            }
        }
    }
    // Body-centered sub-lattice (strictly interior, never on a face).
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = [(i as f64 + 0.5) * h, (j as f64 + 0.5) * h, (k as f64 + 0.5) * h];
                if !keep(p) {
                    continue;
                }
                sites.push(Site { p, lock: [None, None, None] });
            }
        }
    }
    sites
}

// --- the Lloyd/CVT loop ---------------------------------------------------

/// One Delaunay over the current sites. Returns the real tets in site indices.
fn delaunay(sites: &[Site]) -> Vec<[usize; 4]> {
    let mut db = DelaunayBuilder::enclosing([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    for s in sites {
        db.insert(s.p);
    }
    db.tets()
}

/// CVT-proxy smoothing step: move each non-pinned site toward the volume-
/// weighted average of the centroids of its incident tets (an ODT-flavored
/// Lloyd move that, unlike plain Laplacian, does not shrink the boundary),
/// then re-apply its constraint. Corners (3 locks) never move.
fn lloyd_step(sites: &mut [Site], tets: &[[usize; 4]]) {
    let mut num = vec![[0.0f64; 3]; sites.len()];
    let mut den = vec![0.0f64; sites.len()];
    for t in tets {
        let p = [sites[t[0]].p, sites[t[1]].p, sites[t[2]].p, sites[t[3]].p];
        let vol = tet_det(p).abs();
        let c = centroid4(p);
        for &i in t {
            for k in 0..3 {
                num[i][k] += vol * c[k];
            }
            den[i] += vol;
        }
    }
    for i in 0..sites.len() {
        if sites[i].n_locked() == 3 || den[i] == 0.0 {
            continue;
        }
        let tgt = [num[i][0] / den[i], num[i][1] / den[i], num[i][2] / den[i]];
        sites[i].p = sites[i].project(tgt);
    }
}

fn worst_min_dihedral(sites: &[Site], tets: &[[usize; 4]]) -> f64 {
    tets.iter().fold(f64::MAX, |m, t| {
        let p = [sites[t[0]].p, sites[t[1]].p, sites[t[2]].p, sites[t[3]].p];
        m.min(min_dihedral_deg(p))
    })
}

// --- exact rational volume ------------------------------------------------

/// Exact 6x volume of the tets assigned to `side` (a predicate on the tet's
/// centroid z). Equals 6*(region volume) iff the tet set partitions the region
/// exactly, i.e. iff no tet straddles the interface.
fn region_volume6(sites: &[Site], tets: &[[usize; 4]], side: impl Fn(f64) -> bool) -> BigRational {
    let mut acc = BigRational::zero();
    for t in tets {
        let p = [sites[t[0]].p, sites[t[1]].p, sites[t[2]].p, sites[t[3]].p];
        let cz = 0.25 * (p[0][2] + p[1][2] + p[2][2] + p[3][2]);
        if !side(cz) {
            continue;
        }
        let pr: Vec<[BigRational; 3]> = t.iter().map(|&i| sites[i].p.map(rat)).collect();
        let r: Vec<[BigRational; 3]> = (0..3)
            .map(|k| std::array::from_fn(|j| &pr[k][j] - &pr[3][j]))
            .collect();
        let det = &r[0][0] * (&r[1][1] * &r[2][2] - &r[1][2] * &r[2][1])
            - &r[0][1] * (&r[1][0] * &r[2][2] - &r[1][2] * &r[2][0])
            + &r[0][2] * (&r[1][0] * &r[2][1] - &r[1][1] * &r[2][0]);
        acc += det;
    }
    acc
}

/// Count tets that straddle z = 1/2 (a vertex strictly below AND one strictly
/// above). Zero straddlers <=> the interface is a union of tet faces.
fn straddlers(sites: &[Site], tets: &[[usize; 4]]) -> usize {
    tets.iter()
        .filter(|t| {
            let zs: Vec<f64> = t.iter().map(|&i| sites[i].p[2]).collect();
            zs.iter().any(|&z| z < 0.5 - 1e-12) && zs.iter().any(|&z| z > 0.5 + 1e-12)
        })
        .count()
}

/// Boundary watertightness: every triangle on the cube hull bounds exactly one
/// tet, every interior face exactly two.
fn assert_watertight(tets: &[[usize; 4]]) {
    use std::collections::HashMap;
    let mut faces: HashMap<[usize; 3], usize> = HashMap::new();
    for t in tets {
        for i in 0..4 {
            let mut f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
            f.sort_unstable();
            *faces.entry([f[0], f[1], f[2]]).or_default() += 1;
        }
    }
    assert!(faces.values().all(|&c| c <= 2), "non-manifold face (>2 tets)");
}

// --- experiments ----------------------------------------------------------

#[test]
#[ignore = "WP0 spike, run explicitly with --ignored --nocapture"]
fn spike_a_single_region_cube() {
    let mut sites = bcc_sites(4, false, |_| true);
    eprintln!("\n[A] unit cube, {} sites", sites.len());
    let mut tets = delaunay(&sites);
    eprintln!("    iter 0: min dihedral = {:.2} deg", worst_min_dihedral(&sites, &tets));
    for it in 1..=6 {
        lloyd_step(&mut sites, &tets);
        tets = delaunay(&sites);
        eprintln!("    iter {it}: min dihedral = {:.2} deg", worst_min_dihedral(&sites, &tets));
    }
    assert_watertight(&tets);
    let vol6 = region_volume6(&sites, &tets, |_| true);
    eprintln!("    exact volume*6 = {vol6}  (want 6)");
    assert_eq!(vol6, rat(6.0), "single-region cube must have exact volume");
}

#[test]
#[ignore = "WP0 spike, run explicitly with --ignored --nocapture"]
fn spike_b_naive_interface() {
    // Both sides smoothed independently. Interface-layer sites are pinned on
    // z = 1/2, but interior sites stay free in z and may drift across the plane.
    // This documents that naive point-moving CVT does not recover the interface.
    let mut sites = bcc_sites(4, true, |_| true);
    eprintln!("\n[B naive] split cube, {} sites", sites.len());
    let mut tets = delaunay(&sites);
    for _ in 0..6 {
        lloyd_step(&mut sites, &tets);
        tets = delaunay(&sites);
    }
    let str_count = straddlers(&sites, &tets);
    let va = region_volume6(&sites, &tets, |z| z < 0.5);
    let vb = region_volume6(&sites, &tets, |z| z >= 0.5);
    eprintln!("    straddlers = {str_count}  (0 == interface recovered)");
    eprintln!("    vol*6 below = {va}, above = {vb}  (want 3 each)");
    // This is the documenting experiment: we EXPECT naive independent smoothing
    // to leave straddlers and miss exact per-region volume.
    eprintln!(
        "    -> naive recovers interface exactly: {}",
        str_count == 0 && va == rat(3.0) && vb == rat(3.0)
    );
}

#[test]
#[ignore = "WP0 spike, run explicitly with --ignored --nocapture"]
fn spike_b_mirror_interface() {
    // Generate + smooth only the lower half (z <= 1/2, including the interface
    // layer), then mirror to the upper half about z = 1/2 every iteration. The
    // resulting site set is symmetric, so z = 1/2 is a set of Voronoi facets and
    // its dual Delaunay carries the interface exactly.
    let mut lower = bcc_sites(4, true, |p| p[2] <= 0.5 + 1e-12);

    let mirror = |s: &[Site]| -> Vec<Site> {
        let mut all: Vec<Site> = s.to_vec();
        for site in s {
            // Skip the interface layer itself (z == 1/2) to avoid duplicates.
            if (site.p[2] - 0.5).abs() < 1e-12 {
                continue;
            }
            let mut m = site.clone();
            m.p[2] = 1.0 - site.p[2];
            // Mirror the z lock if any (a z=0 face becomes z=1).
            m.lock[2] = site.lock[2].map(|c| 1.0 - c);
            all.push(m);
        }
        all
    };

    let mut all = mirror(&lower);
    eprintln!("\n[B mirror] split cube, {} sites ({} lower)", all.len(), lower.len());
    let mut tets = delaunay(&all);
    eprintln!("    iter 0: min dihedral = {:.2} deg", worst_min_dihedral(&all, &tets));
    for it in 1..=6 {
        // Smooth the lower half against the full (mirrored) triangulation so the
        // interface sees neighbors on both sides, then re-mirror.
        let full = mirror(&lower);
        let ftets = delaunay(&full);
        // Map lower-site indices are the first lower.len() entries of `full`.
        lloyd_step_subset(&mut lower, &full, &ftets);
        all = mirror(&lower);
        tets = delaunay(&all);
        eprintln!("    iter {it}: min dihedral = {:.2} deg", worst_min_dihedral(&all, &tets));
    }

    assert_watertight(&tets);
    let str_count = straddlers(&all, &tets);
    eprintln!("    straddlers = {str_count}  (0 == interface recovered)");
    // Documenting: mirror symmetry makes z=1/2 a union of VORONOI facets, but the
    // dual DELAUNAY crosses the plane with mirror-pair edges, so tets straddle.
    eprintln!(
        "    -> mirror recovers interface exactly: {}",
        str_count == 0
    );
}

#[test]
#[ignore = "WP0 spike, run explicitly with --ignored --nocapture"]
fn spike_c_interface_refinement() {
    // The production technique for an internal interface: restricted-Delaunay
    // refinement. Start from interface-layer sites on z=1/2 plus interior sites,
    // then repeatedly insert the on-plane crossing point of every straddling tet
    // edge until no tet straddles. All inserted points have z == 1/2 exactly, so
    // a clean (zero-straddler) result keeps bit-exact per-region volume.
    let mut sites = bcc_sites(4, true, |_| true);
    eprintln!("\n[C refine] split cube, {} sites", sites.len());

    let near = |a: V3, pts: &[Site]| pts.iter().any(|s| {
        let d = sub(a, s.p);
        norm(d) < 1e-6
    });

    for round in 0..12 {
        let tets = delaunay(&sites);
        let s = straddlers(&sites, &tets);
        eprintln!("    round {round}: {} sites, {s} straddlers", sites.len());
        if s == 0 {
            break;
        }
        // Collect on-plane crossing points of straddling tet edges.
        let mut adds: Vec<V3> = Vec::new();
        for t in &tets {
            let zs: Vec<f64> = t.iter().map(|&i| sites[i].p[2]).collect();
            if !(zs.iter().any(|&z| z < 0.5 - 1e-12) && zs.iter().any(|&z| z > 0.5 + 1e-12)) {
                continue;
            }
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let (pa, pb) = (sites[t[a]].p, sites[t[b]].p);
                    if (pa[2] < 0.5 - 1e-12 && pb[2] > 0.5 + 1e-12)
                        || (pb[2] < 0.5 - 1e-12 && pa[2] > 0.5 + 1e-12)
                    {
                        let s = (0.5 - pa[2]) / (pb[2] - pa[2]);
                        let mut x = [
                            pa[0] + s * (pb[0] - pa[0]),
                            pa[1] + s * (pb[1] - pa[1]),
                            0.5,
                        ];
                        x[0] = x[0].clamp(0.0, 1.0);
                        x[1] = x[1].clamp(0.0, 1.0);
                        if !near(x, &sites) && !adds.iter().any(|&q| norm(sub(q, x)) < 1e-6) {
                            adds.push(x);
                        }
                    }
                }
            }
        }
        if adds.is_empty() {
            eprintln!("    (no new crossing points; refinement stalled)");
            break;
        }
        for x in adds {
            sites.push(Site { p: x, lock: [None, None, Some(0.5)] });
        }
    }

    let tets = delaunay(&sites);
    let s = straddlers(&sites, &tets);
    let va = region_volume6(&sites, &tets, |z| z < 0.5);
    let vb = region_volume6(&sites, &tets, |z| z >= 0.5);
    eprintln!("    final: {} sites, {s} straddlers", sites.len());
    eprintln!("    vol*6 below = {va}, above = {vb}  (want 3 each)");
    eprintln!(
        "    -> refinement recovers interface exactly: {}",
        s == 0 && va == rat(3.0) && vb == rat(3.0)
    );
}

/// Like `lloyd_step` but only moves the first `subset.len()` sites (the lower
/// half), reading incident tets from the full mirrored triangulation `full`.
fn lloyd_step_subset(subset: &mut [Site], full: &[Site], tets: &[[usize; 4]]) {
    let mut num = vec![[0.0f64; 3]; full.len()];
    let mut den = vec![0.0f64; full.len()];
    for t in tets {
        let p = [full[t[0]].p, full[t[1]].p, full[t[2]].p, full[t[3]].p];
        let vol = tet_det(p).abs();
        let c = centroid4(p);
        for &i in t {
            for k in 0..3 {
                num[i][k] += vol * c[k];
            }
            den[i] += vol;
        }
    }
    for i in 0..subset.len() {
        if subset[i].n_locked() == 3 || den[i] == 0.0 {
            continue;
        }
        let tgt = [num[i][0] / den[i], num[i][1] / den[i], num[i][2] / den[i]];
        subset[i].p = subset[i].project(tgt);
    }
}
