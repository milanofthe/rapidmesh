//! Primitive shapes: solids (closed, outward-oriented) and sheets (open
//! surfaces, e.g. PEC traces and ports, embeddable in volumes).
//!
//! Every builder returns a [`Faceted`] with per-triangle analytic surface
//! back-references. Solid builders guarantee watertightness and outward
//! orientation; sheet builders guarantee consistent winding.

use crate::faceted::{Faceted, SurfaceKind};
use crate::polygon::{polygon_orientation, triangulate_polygon};
use rapidmesh_csg::Tri;
use rapidmesh_exact::Sign;

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Two unit vectors orthogonal to `axis` (and to each other).
fn orthonormal_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = norm(axis);
    assert!(n > 0.0, "axis must be nonzero");
    let a = scale(axis, 1.0 / n);
    // Pick the coordinate axis least aligned with `a`.
    let helper = if a[0].abs() <= a[1].abs() && a[0].abs() <= a[2].abs() {
        [1.0, 0.0, 0.0]
    } else if a[1].abs() <= a[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let e1 = cross3(a, helper);
    let e1 = scale(e1, 1.0 / norm(e1));
    let e2 = cross3(a, e1);
    (e1, e2)
}

/// Embeds a local 2D point into the (base, u, v) frame.
fn embed(base: [f64; 3], u: [f64; 3], v: [f64; 3], p: [f64; 2]) -> [f64; 3] {
    [
        base[0] + p[0] * u[0] + p[1] * v[0],
        base[1] + p[0] * u[1] + p[1] * v[1],
        base[2] + p[0] * u[2] + p[1] * v[2],
    ]
}

// ------------------------------------------------------------------ solids

/// Axis-aligned box. Six `Plane` face groups in the order
/// -z, +z, -y, +y, -x, +x.
pub fn solid_box(min: [f64; 3], max: [f64; 3]) -> Faceted {
    assert!((0..3).all(|k| min[k] < max[k]), "box must have positive extent");
    // Corner index bits: bit0 = x, bit1 = y, bit2 = z.
    let c: [[f64; 3]; 8] = std::array::from_fn(|i| {
        [
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        ]
    });
    let quads: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut f = Faceted::new();
    for q in quads {
        let s = f.add_surface(SurfaceKind::Plane);
        f.push_tri(Tri::new(c[q[0]], c[q[1]], c[q[2]]), s);
        f.push_tri(Tri::new(c[q[0]], c[q[2]], c[q[3]]), s);
    }
    f
}

/// Circular frustum from `base_center` along the full height vector `axis`,
/// radii `r_base` and `r_top` (`r_top == 0` gives a cone). Surface groups:
/// barrel (`Cylinder`/`Cone`), bottom cap, top cap (absent for a cone).
pub fn frustum(
    base_center: [f64; 3],
    axis: [f64; 3],
    r_base: f64,
    r_top: f64,
    segments: usize,
) -> Faceted {
    assert!(segments >= 3, "need at least 3 segments");
    assert!(r_base > 0.0 && r_top >= 0.0, "radii must be positive (top may be 0)");
    let (e1, e2) = orthonormal_basis(axis);
    let top_center = add(base_center, axis);
    let ring = |center: [f64; 3], r: f64| -> Vec<[f64; 3]> {
        (0..segments)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
                add(center, add(scale(e1, r * t.cos()), scale(e2, r * t.sin())))
            })
            .collect()
    };
    let bottom = ring(base_center, r_base);

    let mut f = Faceted::new();
    let barrel_kind = if r_top == r_base {
        SurfaceKind::Cylinder {
            center: base_center,
            axis,
            radius: r_base,
        }
    } else {
        // Apex where the barrel lines meet.
        let factor = r_base / (r_base - r_top);
        let apex = add(base_center, scale(axis, factor));
        SurfaceKind::Cone {
            apex,
            axis: scale(axis, -factor),
            tan_half_angle: r_base / (norm(axis) * factor).abs(),
        }
    };
    let barrel = f.add_surface(barrel_kind);

    if r_top == 0.0 {
        for i in 0..segments {
            let j = (i + 1) % segments;
            f.push_tri(Tri::new(bottom[i], bottom[j], top_center), barrel);
        }
    } else {
        let top = ring(top_center, r_top);
        for i in 0..segments {
            let j = (i + 1) % segments;
            f.push_tri(Tri::new(bottom[i], bottom[j], top[j]), barrel);
            f.push_tri(Tri::new(bottom[i], top[j], top[i]), barrel);
        }
        let cap = f.add_surface(SurfaceKind::Plane);
        for i in 0..segments {
            let j = (i + 1) % segments;
            f.push_tri(Tri::new(top_center, top[i], top[j]), cap);
        }
    }
    let cap = f.add_surface(SurfaceKind::Plane);
    for i in 0..segments {
        let j = (i + 1) % segments;
        f.push_tri(Tri::new(base_center, bottom[j], bottom[i]), cap);
    }
    f
}

/// Circular cylinder from `base_center` along the full height vector `axis`.
pub fn cylinder(base_center: [f64; 3], axis: [f64; 3], radius: f64, segments: usize) -> Faceted {
    frustum(base_center, axis, radius, radius, segments)
}

/// UV sphere: `segments` longitudes (>= 3), `rings` latitude bands (>= 2).
pub fn sphere(center: [f64; 3], radius: f64, segments: usize, rings: usize) -> Faceted {
    assert!(segments >= 3 && rings >= 2, "sphere needs >= 3 segments, >= 2 rings");
    assert!(radius > 0.0);
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Sphere { center, radius });
    let pt = |theta: f64, phi: f64| -> [f64; 3] {
        add(
            center,
            [
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            ],
        )
    };
    let north = add(center, [0.0, 0.0, radius]);
    let south = add(center, [0.0, 0.0, -radius]);
    let row: Vec<Vec<[f64; 3]>> = (1..rings)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / rings as f64;
            (0..segments)
                .map(|i| pt(theta, 2.0 * std::f64::consts::PI * i as f64 / segments as f64))
                .collect()
        })
        .collect();
    for i in 0..segments {
        let j = (i + 1) % segments;
        f.push_tri(Tri::new(north, row[0][i], row[0][j]), s);
        let last = &row[rings - 2];
        f.push_tri(Tri::new(south, last[j], last[i]), s);
    }
    for k in 0..rings - 2 {
        for i in 0..segments {
            let j = (i + 1) % segments;
            let (a, b) = (row[k][i], row[k][j]);
            let (d, c) = (row[k + 1][i], row[k + 1][j]);
            f.push_tri(Tri::new(a, c, b), s);
            f.push_tri(Tri::new(a, d, c), s);
        }
    }
    f
}

/// Prism: a planar polygon (with holes) in the (base, u, v) frame, extruded
/// along `h`. The frame must be right-handed with respect to the extrusion:
/// (u x v) . h > 0. Surface groups: bottom, top, one side group per ring.
pub fn extrude_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    h: [f64; 3],
) -> Faceted {
    assert!(
        dot3(cross3(u, v), h) > 0.0,
        "extrusion frame must be right-handed: (u x v) . h > 0"
    );
    // Normalize ring orientations: outer counterclockwise, holes clockwise
    // (so that the shared wall formula yields outward normals everywhere).
    let normalize = |pts: &[[f64; 2]], want: Sign| -> Vec<[f64; 2]> {
        let o = polygon_orientation(pts);
        assert_ne!(o, Sign::Zero, "degenerate polygon ring");
        if o == want {
            pts.to_vec()
        } else {
            pts.iter().rev().copied().collect()
        }
    };
    let outer_ccw = normalize(outer, Sign::Positive);
    let holes_cw: Vec<Vec<[f64; 2]>> = holes
        .iter()
        .map(|hole| normalize(hole, Sign::Negative))
        .collect();

    let cap = triangulate_polygon(&outer_ccw, &holes_cw);
    let mut f = Faceted::new();

    // Bottom cap: counterclockwise in (u, v) faces along +(u x v); the
    // outward bottom normal is the opposite, so reverse the winding.
    let bottom = f.add_surface(SurfaceKind::Plane);
    for t in &cap {
        f.push_tri(
            Tri::new(
                embed(base, u, v, t[0]),
                embed(base, u, v, t[2]),
                embed(base, u, v, t[1]),
            ),
            bottom,
        );
    }
    let top_base = add(base, h);
    let top = f.add_surface(SurfaceKind::Plane);
    for t in &cap {
        f.push_tri(
            Tri::new(
                embed(top_base, u, v, t[0]),
                embed(top_base, u, v, t[1]),
                embed(top_base, u, v, t[2]),
            ),
            top,
        );
    }

    // Walls: region lies left of every (normalized) ring edge, so the quad
    // (a, b, b+h, a+h) faces outward.
    for ring in std::iter::once(&outer_ccw).chain(holes_cw.iter()) {
        let side = f.add_surface(SurfaceKind::Plane);
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            let a = embed(base, u, v, ring[i]);
            let b = embed(base, u, v, ring[j]);
            let (at, bt) = (add(a, h), add(b, h));
            f.push_tri(Tri::new(a, b, bt), side);
            f.push_tri(Tri::new(a, bt, at), side);
        }
    }
    f
}

/// UV torus around `center` with the major circle normal to `axis`.
/// `segments_major` divides the major circle, `segments_minor` the tube.
pub fn torus(
    center: [f64; 3],
    axis: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
    segments_major: usize,
    segments_minor: usize,
) -> Faceted {
    assert!(segments_major >= 3 && segments_minor >= 3, "torus needs >= 3 segments per ring");
    assert!(
        major_radius > minor_radius && minor_radius > 0.0,
        "torus needs 0 < minor_radius < major_radius"
    );
    let (e1, e2) = orthonormal_basis(axis);
    let a_hat = scale(axis, 1.0 / norm(axis));
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Torus {
        center,
        axis,
        major_radius,
        minor_radius,
    });
    let pt = |i: usize, j: usize| -> [f64; 3] {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / segments_major as f64;
        let phi = 2.0 * std::f64::consts::PI * j as f64 / segments_minor as f64;
        let radial = add(scale(e1, theta.cos()), scale(e2, theta.sin()));
        add(
            center,
            add(
                scale(radial, major_radius + minor_radius * phi.cos()),
                scale(a_hat, minor_radius * phi.sin()),
            ),
        )
    };
    for i in 0..segments_major {
        let i2 = (i + 1) % segments_major;
        for j in 0..segments_minor {
            let j2 = (j + 1) % segments_minor;
            let (a, b, c, d) = (pt(i, j), pt(i2, j), pt(i2, j2), pt(i, j2));
            f.push_tri(Tri::new(a, b, c), s);
            f.push_tri(Tri::new(a, c, d), s);
        }
    }
    f
}

/// Wedge: a box `dx x dy x dz` at `position` whose top edge is shortened to
/// `top_x` along x (0 gives a triangular prism). The trapezoid profile lives
/// in the xz-plane and extrudes along +y.
pub fn wedge(position: [f64; 3], dx: f64, dy: f64, dz: f64, top_x: f64) -> Faceted {
    assert!(dx > 0.0 && dy > 0.0 && dz > 0.0, "wedge must have positive extent");
    assert!((0.0..=dx).contains(&top_x), "top_x must lie in [0, dx]");
    // Profile points as (z, x) pairs in the (u = z-hat, v = x-hat) frame:
    // (u x v) . h = (z x x) . y = +1, right-handed.
    let profile: Vec<[f64; 2]> = if top_x > 0.0 {
        vec![[0.0, 0.0], [0.0, dx], [dz, top_x], [dz, 0.0]]
    } else {
        vec![[0.0, 0.0], [0.0, dx], [dz, 0.0]]
    };
    extrude_polygon(
        &profile,
        &[],
        position,
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, dy, 0.0],
    )
}

/// Signed volume of a closed faceted shape (divergence theorem); used to
/// normalize outward orientation of sweep/loft results.
fn signed_volume(f: &Faceted) -> f64 {
    let mut v6 = 0.0;
    for t in &f.tris {
        let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
        v6 += dot3(a, cross3(b, c));
    }
    v6 / 6.0
}

/// Tube swept along an open polyline path with a circular cross-section
/// (rapidfem's `sweep_along_path`/`helix` substrate). One ring per path
/// node, oriented normal to the local tangent bisector, with a
/// parallel-transported frame (no twist). The path must not double back on
/// itself, and `radius` must stay below the local curvature radius or the
/// tube self-intersects (the caller controls sampling density). Facets carry
/// no analytic surface (fidelity snapping off; sample the path finely).
pub fn pipe(path: &[[f64; 3]], radius: f64, segments: usize) -> Faceted {
    assert!(path.len() >= 2, "pipe path needs at least 2 points");
    assert!(radius > 0.0 && segments >= 3);
    let n = path.len();
    let seg_dir = |i: usize| -> [f64; 3] {
        let d: [f64; 3] = std::array::from_fn(|k| path[i + 1][k] - path[i][k]);
        let l = norm(d);
        assert!(l > 0.0, "pipe path points must be distinct");
        scale(d, 1.0 / l)
    };
    // Node normals: tangent bisectors (segment tangents at the ends).
    let node_normal = |i: usize| -> [f64; 3] {
        let t = if i == 0 {
            seg_dir(0)
        } else if i == n - 1 {
            seg_dir(n - 2)
        } else {
            let s = add(seg_dir(i - 1), seg_dir(i));
            let l = norm(s);
            assert!(l > 1e-9, "pipe path doubles back on itself");
            scale(s, 1.0 / l)
        };
        t
    };
    // Parallel transport the ring frame from node to node (rotation taking
    // one bisector to the next; no roll accumulates).
    let mut frames: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(n);
    let n0 = node_normal(0);
    frames.push(orthonormal_basis(n0));
    for i in 1..n {
        let (prev_n, cur_n) = (node_normal(i - 1), node_normal(i));
        let (u, v) = frames[i - 1];
        let c = dot3(prev_n, cur_n).clamp(-1.0, 1.0);
        let axis = cross3(prev_n, cur_n);
        let s = norm(axis);
        if s < 1e-12 {
            frames.push((u, v));
            continue;
        }
        let k = scale(axis, 1.0 / s);
        let rot = |p: [f64; 3]| -> [f64; 3] {
            // Rodrigues: p c + (k x p) s + k (k . p)(1 - c).
            let kxp = cross3(k, p);
            let kdp = dot3(k, p);
            std::array::from_fn(|m| p[m] * c + kxp[m] * s + k[m] * kdp * (1.0 - c))
        };
        frames.push((rot(u), rot(v)));
    }
    let ring = |i: usize| -> Vec<[f64; 3]> {
        let (u, v) = frames[i];
        (0..segments)
            .map(|j| {
                let t = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
                add(
                    path[i],
                    add(scale(u, radius * t.cos()), scale(v, radius * t.sin())),
                )
            })
            .collect()
    };
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
    let rings: Vec<Vec<[f64; 3]>> = (0..n).map(ring).collect();
    for i in 0..n - 1 {
        for j in 0..segments {
            let j2 = (j + 1) % segments;
            let (a, b) = (rings[i][j], rings[i][j2]);
            let (d, c) = (rings[i + 1][j], rings[i + 1][j2]);
            f.push_tri(Tri::new(a, b, c), s);
            f.push_tri(Tri::new(a, c, d), s);
        }
    }
    let cap0 = f.add_surface(SurfaceKind::Plane);
    for j in 0..segments {
        let j2 = (j + 1) % segments;
        f.push_tri(Tri::new(path[0], rings[0][j2], rings[0][j]), cap0);
    }
    let cap1 = f.add_surface(SurfaceKind::Plane);
    for j in 0..segments {
        let j2 = (j + 1) % segments;
        f.push_tri(Tri::new(path[n - 1], rings[n - 1][j], rings[n - 1][j2]), cap1);
    }
    debug_assert!(signed_volume(&f) > 0.0, "pipe orientation");
    f
}

/// Helical tube around the +z axis through `base`: `radius` of the helix,
/// `pitch` advance per turn, circular wire cross-section. Composition of a
/// sampled helix path and [`pipe`].
pub fn helix(
    base: [f64; 3],
    radius: f64,
    pitch: f64,
    turns: f64,
    wire_radius: f64,
    points_per_turn: usize,
    segments: usize,
) -> Faceted {
    assert!(turns > 0.0 && points_per_turn >= 8);
    let n = ((turns * points_per_turn as f64).ceil() as usize).max(2);
    let path: Vec<[f64; 3]> = (0..=n)
        .map(|i| {
            let t = turns * i as f64 / n as f64;
            let ang = 2.0 * std::f64::consts::PI * t;
            [
                base[0] + radius * ang.cos(),
                base[1] + radius * ang.sin(),
                base[2] + pitch * t,
            ]
        })
        .collect();
    pipe(&path, wire_radius, segments)
}

/// Ruled loft between two planar profiles with the SAME vertex count,
/// corresponded by index (rapidfem's horn workhorse). Caps are fanned from
/// the profile centroids, so each profile must be star-shaped with respect
/// to its centroid (convex profiles always are). Output orientation is
/// normalized to outward via the signed volume.
pub fn loft(profile_a: &[[f64; 3]], profile_b: &[[f64; 3]]) -> Faceted {
    assert!(
        profile_a.len() == profile_b.len() && profile_a.len() >= 3,
        "loft profiles need the same vertex count (>= 3)"
    );
    let n = profile_a.len();
    let centroid = |pts: &[[f64; 3]]| -> [f64; 3] {
        let mut c = [0.0; 3];
        for p in pts {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        scale(c, 1.0 / pts.len() as f64)
    };
    let (ca, cb) = (centroid(profile_a), centroid(profile_b));
    let mut f = Faceted::new();
    let cap_a = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(ca, profile_a[j], profile_a[i]), cap_a);
    }
    let cap_b = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(cb, profile_b[i], profile_b[j]), cap_b);
    }
    let side = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b) = (profile_a[i], profile_a[j]);
        let (d, c) = (profile_b[i], profile_b[j]);
        f.push_tri(Tri::new(a, b, c), side);
        f.push_tri(Tri::new(a, c, d), side);
    }
    let vol = signed_volume(&f);
    assert!(vol.abs() > 0.0, "degenerate loft");
    if vol < 0.0 {
        for t in &mut f.tris {
            t.v.swap(1, 2);
        }
    }
    f
}

// ------------------------------------------------------------------ sheets

/// Parallelogram sheet spanned by `u`, `v` at `corner`.
pub fn sheet_rect(corner: [f64; 3], u: [f64; 3], v: [f64; 3]) -> Faceted {
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
    let b = add(corner, u);
    let c = add(add(corner, u), v);
    let d = add(corner, v);
    f.push_tri(Tri::new(corner, b, c), s);
    f.push_tri(Tri::new(corner, c, d), s);
    f
}

/// Planar polygon sheet (with holes) in the (base, u, v) frame.
pub fn sheet_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
) -> Faceted {
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
    for t in triangulate_polygon(outer, holes) {
        f.push_tri(
            Tri::new(
                embed(base, u, v, t[0]),
                embed(base, u, v, t[1]),
                embed(base, u, v, t[2]),
            ),
            s,
        );
    }
    f
}

/// Elliptic disk sheet at `center` spanned by the radius vectors `e1`, `e2`.
pub fn sheet_disk(center: [f64; 3], e1: [f64; 3], e2: [f64; 3], segments: usize) -> Faceted {
    assert!(segments >= 3);
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
    let ring: Vec<[f64; 3]> = (0..segments)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
            add(center, add(scale(e1, t.cos()), scale(e2, t.sin())))
        })
        .collect();
    for i in 0..segments {
        let j = (i + 1) % segments;
        f.push_tri(Tri::new(center, ring[i], ring[j]), s);
    }
    f
}
