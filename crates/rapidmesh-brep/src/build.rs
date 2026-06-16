//! Builds a [`Brep`] from the exact CSG output (`TaggedPlc`).
//!
//! The tagged triangle soup is the source of truth for TOPOLOGY (which surfaces
//! meet, region labels, exact vertex positions). This step RECONSTRUCTS the
//! boundary representation from it -- groups triangles into faces, chains their
//! boundary edges into B-rep edges, recovers an analytic curve per edge, orders
//! the loops, and radially links faces. Nothing is snapped: positions, regions
//! and incidence come unchanged from the arrangement; only analytic curves are
//! added on top.
//!
//! Both the CSG path and the STEP-import path converge on `TaggedPlc`, so this
//! one function covers both.

use crate::{
    Brep, CoEdge, CoEdgeId, Curve, Edge, EdgeId, Face, FaceId, Loop, PCurve, Surface, SurfaceId,
    Vertex, VertexId,
};
use rapidmesh_geom::{SurfaceKind, TaggedPlc};
use std::collections::HashMap;

type V3 = [f64; 3];

/// Turn cosine below which a degree-2 vertex is still a corner (45 deg), matching
/// the mesher's feature-edge splitter.
const CORNER_COS: f64 = 0.707;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
fn key2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn uf_find(rep: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while rep[r] != r {
        r = rep[r];
    }
    let mut c = x;
    while rep[c] != c {
        let nx = rep[c];
        rep[c] = r;
        c = nx;
    }
    r
}
fn uf_union(rep: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (uf_find(rep, a), uf_find(rep, b));
    if ra != rb {
        rep[ra.max(rb)] = ra.min(rb);
    }
}

/// Build a B-rep from a tagged PLC (pure function; no CSG state, no snapping).
pub fn from_plc(plc: &TaggedPlc) -> Brep {
    let pos: &[V3] = &plc.vertices;
    let tri = |i: usize| {
        let t = plc.triangles[i];
        [t[0] as usize, t[1] as usize, t[2] as usize]
    };
    let n_tri = plc.triangles.len();
    let mut diag = 0.0f64;
    {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        for k in 0..3 {
            diag = diag.max(hi[k] - lo[k]);
        }
    }
    let tol = 1e-9 * diag.max(1.0);

    // ---- B1: group triangles into faces (key + connected component) ----------
    // Key = (analytic surface, unordered region pair, face tag). Within a key,
    // triangles connected through a shared edge are ONE face (two disjoint
    // patches of the same surface stay separate).
    let tkey = |i: usize| -> (u32, u32, u32, u32) {
        let r = plc.region_tags[i];
        (plc.surface_refs[i].0, r[0].0.min(r[1].0), r[0].0.max(r[1].0), plc.face_tags[i].0)
    };
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for i in 0..n_tri {
        let c = tri(i);
        for e in 0..3 {
            edge_tris.entry(key2(c[e], c[(e + 1) % 3])).or_default().push(i);
        }
    }
    let mut frep: Vec<usize> = (0..n_tri).collect();
    for tris in edge_tris.values() {
        for a in 0..tris.len() {
            for b in (a + 1)..tris.len() {
                if tkey(tris[a]) == tkey(tris[b]) {
                    uf_union(&mut frep, tris[a], tris[b]);
                }
            }
        }
    }
    // Component representative -> face id; build the Face records.
    let mut face_of_rep: HashMap<usize, usize> = HashMap::new();
    let mut faces: Vec<Face> = Vec::new();
    let mut tri_face: Vec<usize> = vec![usize::MAX; n_tri];
    for i in 0..n_tri {
        let r = uf_find(&mut frep, i);
        let fid = *face_of_rep.entry(r).or_insert_with(|| {
            let rt = plc.region_tags[i];
            let sid = plc.surface_refs[i].0;
            faces.push(Face {
                surface: SurfaceId(sid),
                loops: Vec::new(),
                regions: rt,
                face_tag: plc.face_tags[i],
                plc_surface: sid,
                owner: plc.surface_owners[sid as usize],
                facets: Vec::new(),
            });
            faces.len() - 1
        });
        tri_face[i] = fid;
        faces[fid].facets.push(i as u32);
    }

    // ---- boundary edges per face, and the radial face set per edge -----------
    // A face's boundary edge is used by exactly one of its triangles (interior
    // edges by two). The set of faces sharing a boundary edge is its radial set.
    let mut bedge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    {
        // count (face, edge) uses
        let mut fe_count: HashMap<(usize, (usize, usize)), usize> = HashMap::new();
        for i in 0..n_tri {
            let c = tri(i);
            let f = tri_face[i];
            for e in 0..3 {
                *fe_count.entry((f, key2(c[e], c[(e + 1) % 3]))).or_insert(0) += 1;
            }
        }
        for ((f, e), cnt) in fe_count {
            if cnt == 1 {
                bedge_faces.entry(e).or_default().push(f);
            }
        }
    }
    for v in bedge_faces.values_mut() {
        v.sort_unstable();
        v.dedup();
    }

    // ---- B3: chain boundary edges into B-rep edges, split at corners ---------
    // The boundary graph: vertices linked by boundary edges. Walk maximal chains
    // that keep the SAME radial face set, splitting at junctions (degree != 2),
    // at a change of the face set, and at sharp turns (> 45 deg).
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in bedge_faces.keys() {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let fset = |a: usize, b: usize| -> &Vec<usize> { &bedge_faces[&key2(a, b)] };
    let is_corner = |v: usize, adj: &HashMap<usize, Vec<usize>>| -> bool {
        let ns = &adj[&v];
        if ns.len() != 2 {
            return true;
        }
        if fset(v, ns[0]) != fset(v, ns[1]) {
            return true;
        }
        let d0 = norm(sub(pos[v], pos[ns[0]]));
        let d1 = norm(sub(pos[ns[1]], pos[v]));
        dot(d0, d1) < CORNER_COS
    };
    let walk = |c0: usize,
                start: usize,
                adj: &HashMap<usize, Vec<usize>>,
                done: &mut std::collections::HashSet<(usize, usize)>|
     -> Vec<usize> {
        let mut chain = vec![c0];
        let (mut prev, mut cur) = (c0, start);
        loop {
            chain.push(cur);
            done.insert(key2(prev, cur));
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
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut done: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut corners: Vec<usize> = adj.keys().copied().filter(|&v| is_corner(v, &adj)).collect();
    corners.sort_unstable();
    for &c0 in &corners {
        for &start in &adj[&c0].clone() {
            if !done.contains(&key2(c0, start)) {
                chains.push(walk(c0, start, &adj, &mut done));
            }
        }
    }
    // Corner-less loops (a smooth rim): anchor at the lowest-index vertex.
    let mut keys: Vec<usize> = adj.keys().copied().collect();
    keys.sort_unstable();
    for &a in &keys {
        for &b in &adj[&a].clone() {
            if !done.contains(&key2(a, b)) {
                let mut ch = walk(a, b, &adj, &mut done);
                if ch.last() != Some(&a) {
                    ch.push(a);
                }
                chains.push(ch);
            }
        }
    }

    // Near-closed chain endpoints (a blunt trailing edge: the two endpoints sit a
    // sub-arc gap apart) merge to one corner -- a topology op, not a PLC weld.
    let mut vrep: Vec<usize> = (0..pos.len()).collect();
    for ch in &chains {
        let (a, b) = (ch[0], *ch.last().unwrap());
        let arc: f64 = ch.windows(2).map(|w| dist(pos[w[0]], pos[w[1]])).sum();
        if a != b && arc > 0.0 && dist(pos[a], pos[b]) < 0.05 * arc {
            uf_union(&mut vrep, a, b);
        }
    }
    for v in 0..vrep.len() {
        uf_find(&mut vrep, v);
    }

    // ---- B2: vertices = unique chain endpoints (through the merge) -----------
    let mut vid: HashMap<usize, VertexId> = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let corner_id = |plc_v: usize,
                         vid: &mut HashMap<usize, VertexId>,
                         vertices: &mut Vec<Vertex>,
                         vrep: &[usize]|
     -> VertexId {
        let r = vrep[plc_v];
        *vid.entry(r).or_insert_with(|| {
            vertices.push(Vertex { pos: pos[r] });
            VertexId((vertices.len() - 1) as u32)
        })
    };

    // ---- B3 cont.: build Edge records (curve recovery), keep radial faces ----
    let mut edges: Vec<Edge> = Vec::new();
    let mut edge_faces: Vec<Vec<FaceId>> = Vec::new();
    for ch in &chains {
        let a = ch[0];
        let b = *ch.last().unwrap();
        let va = corner_id(a, &mut vid, &mut vertices, &vrep);
        let vb = corner_id(b, &mut vid, &mut vertices, &vrep);
        let chain_pts: Vec<V3> = ch.iter().map(|&v| pos[v]).collect();
        // radial faces: the face set of the chain's segments (constant by the
        // same-face-set split, so the first segment suffices).
        let mut rad: Vec<FaceId> = fset(ch[0], ch[1]).iter().map(|&f| FaceId(f as u32)).collect();
        rad.sort_unstable();
        let curve = recover_curve(&chain_pts, &rad, &faces, plc, tol);
        edges.push(Edge { ends: [va, vb], chain: chain_pts, curve, coedges: Vec::new() });
        edge_faces.push(rad);
    }

    // ---- B4/B5: per face, build its self-contained surface, order loops, and
    // make one co-edge per (edge, loop-direction) carrying the edge's PCurve in
    // this face's (u,v). A plane gets its frame from the outer loop's points;
    // every other kind is self-contained from its parameters.
    let mut face_edges: Vec<Vec<usize>> = vec![Vec::new(); faces.len()];
    for (ei, ef) in edge_faces.iter().enumerate() {
        for f in ef {
            face_edges[f.0 as usize].push(ei);
        }
    }
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut coedges: Vec<CoEdge> = Vec::new();
    for fid in 0..faces.len() {
        let signed = order_loops(&face_edges[fid], &edges);
        let frame_pts = signed.first().map(|lp| loop_points(lp, &edges)).unwrap_or_default();
        // Build this face's self-contained surface and point the face at it.
        let kind = plc.surfaces[faces[fid].surface.0 as usize].clone();
        let sid = SurfaceId(surfaces.len() as u32);
        surfaces.push(Surface::from_kind(&kind, &frame_pts));
        faces[fid].surface = sid;
        let surf = &surfaces[sid.0 as usize];
        let mut loops_out: Vec<Loop> = Vec::new();
        for sl in &signed {
            let mut lp = Loop::default();
            for &(ei, fwd) in sl {
                let chain = &edges[ei].chain;
                let uv: Vec<[f64; 2]> = if fwd {
                    chain.iter().map(|&p| surf.project_uv(p)).collect()
                } else {
                    chain.iter().rev().map(|&p| surf.project_uv(p)).collect()
                };
                let cid = CoEdgeId(coedges.len() as u32);
                coedges.push(CoEdge {
                    edge: EdgeId(ei as u32),
                    face: FaceId(fid as u32),
                    forward: fwd,
                    pcurve: PCurve { uv },
                });
                edges[ei].coedges.push(cid);
                lp.coedges.push(cid);
            }
            loops_out.push(lp);
        }
        faces[fid].loops = loops_out;
    }

    Brep { vertices, edges, coedges, faces, surfaces }
}

/// Ordered 3D points along a signed-edge loop (chains concatenated, reversed where
/// the loop runs backward), used to fit a planar face's chart frame.
fn loop_points(sl: &[(usize, bool)], edges: &[Edge]) -> Vec<V3> {
    let mut pts: Vec<V3> = Vec::new();
    for &(ei, fwd) in sl {
        let ch = &edges[ei].chain;
        let seq: Vec<V3> = if fwd { ch.clone() } else { ch.iter().rev().cloned().collect() };
        for p in seq {
            if pts.last().map(|&q| dist(q, p) > 1e-12).unwrap_or(true) {
                pts.push(p);
            }
        }
    }
    pts
}

/// Orders a face's edges into oriented loops by walking shared endpoints, as
/// sequences of `(edge index, forward)`. `forward` is true when the loop
/// traverses the edge from `ends[0]` to `ends[1]`. The largest-perimeter loop is
/// placed first (the outer boundary; the rest are holes).
fn order_loops(eids: &[usize], edges: &[Edge]) -> Vec<Vec<(usize, bool)>> {
    let mut adj: HashMap<u32, Vec<usize>> = HashMap::new();
    for &ei in eids {
        let [a, b] = edges[ei].ends;
        adj.entry(a.0).or_default().push(ei);
        if b.0 != a.0 {
            adj.entry(b.0).or_default().push(ei);
        }
    }
    let mut used = vec![false; edges.len()];
    let mut loops: Vec<(f64, Vec<(usize, bool)>)> = Vec::new();
    for &start in eids {
        if used[start] {
            continue;
        }
        let mut seq: Vec<(usize, bool)> = Vec::new();
        let mut perim = 0.0f64;
        let mut cur = start;
        let mut at = edges[start].ends[0].0; // walk leaving from ends[0]
        loop {
            if used[cur] {
                break;
            }
            used[cur] = true;
            let [a, b] = edges[cur].ends;
            let forward = at == a.0;
            let next_v = if forward { b.0 } else { a.0 };
            seq.push((cur, forward));
            perim += arc_len(&edges[cur].chain);
            let nxt = adj.get(&next_v).and_then(|inc| inc.iter().copied().find(|&e| !used[e]));
            match nxt {
                Some(e) => {
                    cur = e;
                    at = next_v;
                }
                None => break,
            }
        }
        loops.push((perim, seq));
    }
    loops.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    loops.into_iter().map(|(_, l)| l).collect()
}

fn arc_len(chain: &[V3]) -> f64 {
    chain.windows(2).map(|w| dist(w[0], w[1])).sum()
}

/// Recovers the analytic curve of an edge from its vertex chain and the surfaces
/// of its radial faces. Handles the forms our scenes use; everything else falls
/// back to the faceted polyline (`Curve::Polyline`).
fn recover_curve(
    chain: &[V3],
    rad: &[FaceId],
    faces: &[Face],
    plc: &TaggedPlc,
    tol: f64,
) -> Curve {
    if chain.len() < 2 {
        return Curve::Polyline;
    }
    let (p0, pn) = (chain[0], chain[chain.len() - 1]);
    let len = arc_len(chain);

    // Straight: every chain point lies on the segment p0..pn.
    if len > 0.0 && dist(p0, pn) > tol {
        let dir = norm(sub(pn, p0));
        let straight = chain.iter().all(|&p| {
            let t = dot(sub(p, p0), dir);
            let foot: V3 = std::array::from_fn(|k| p0[k] + dir[k] * t);
            dist(p, foot) < tol.max(1e-7 * len)
        });
        if straight {
            return Curve::Line { p0, dir };
        }
    }

    // Circular arc / full circle: a sphere/plane or cylinder/plane intersection, a
    // sphere-sphere intersection, a barrel rim. Only attempted when the edge bounds
    // a CURVED face (so a planar polygon is never mistaken for a circle); the fit
    // tolerance is loose enough to accept a faceted polygon's vertices, which lie
    // approximately on the true circle.
    let circ = rad
        .iter()
        .find_map(|f| {
            let kind = &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
            analytic_circle(chain, kind)
        })
        .or_else(|| {
            let curved = rad.iter().any(|f| {
                !matches!(plc.surfaces[faces[f.0 as usize].plc_surface as usize], SurfaceKind::Plane)
            });
            curved.then(|| fit_circle(chain)).flatten()
        });
    if let Some((center, axis, radius, x)) = circ {
        return Curve::Circle { center, axis, radius, x };
    }

    // On an extruded surface at constant height: the analytic profile curve.
    for f in rad {
        let sid = faces[f.0 as usize].surface;
        if let SurfaceKind::Extruded { profile, base, udir, vdir, axis } =
            &plc.surfaces[sid.0 as usize]
        {
            let (u, v, a) = (norm(*udir), norm(*vdir), norm(*axis));
            let z0 = dot(sub(p0, *base), a);
            // constant extrusion height along the whole chain -> a profile edge
            let const_h = chain.iter().all(|&p| (dot(sub(p, *base), a) - z0).abs() < tol.max(1e-7));
            if const_h {
                let foot = |p: V3| -> f64 {
                    let rel = sub(p, *base);
                    profile_footpoint(profile, [dot(rel, u), dot(rel, v)])
                };
                return Curve::Profile {
                    profile: profile.clone(),
                    base: *base,
                    u,
                    v,
                    axis: a,
                    t: [foot(p0), foot(pn)],
                    z: z0,
                };
            }
        }
    }

    Curve::Polyline
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The chain's best-fit plane `(centroid, unit Newell normal)`; `None` if degenerate.
fn chain_plane(chain: &[V3]) -> Option<(V3, V3)> {
    if chain.len() < 3 {
        return None;
    }
    let n = chain.len() as f64;
    let o: V3 = std::array::from_fn(|k| chain.iter().map(|p| p[k]).sum::<f64>() / n);
    let mut nrm = [0.0f64; 3];
    for i in 0..chain.len() {
        let a = chain[i];
        let b = chain[(i + 1) % chain.len()];
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    if dot(nrm, nrm) < 1e-24 {
        return None;
    }
    Some((o, norm(nrm)))
}

/// Recovers an edge's circle EXACTLY from an adjacent analytic curved surface and
/// the chain's plane: `(center, unit axis, radius, unit in-plane x)`. This keeps
/// the edge on the same analytic radius as the face's surface points (a fitted
/// circle would sit a chord-sagitta inside, mismatching the barrel at the rim).
fn analytic_circle(chain: &[V3], kind: &SurfaceKind) -> Option<(V3, V3, f64, V3)> {
    let (o, n) = chain_plane(chain)?;
    let xref = |c: V3, axis: V3| {
        let d = sub(chain[0], c);
        norm(std::array::from_fn(|k| d[k] - axis[k] * dot(d, axis)))
    };
    match kind {
        SurfaceKind::Sphere { center, radius } => {
            let d = dot(sub(o, *center), n);
            let r2 = radius * radius - d * d;
            if r2 <= 1e-18 {
                return None;
            }
            let c = add(*center, scale(n, d));
            Some((c, n, r2.sqrt(), xref(c, n)))
        }
        SurfaceKind::Cylinder { center, axis, radius } => {
            let a = norm(*axis);
            if dot(n, a).abs() < 0.99 {
                return None; // the plane must cut perpendicular to the axis
            }
            let c = add(*center, scale(a, dot(sub(o, *center), a)));
            Some((c, a, *radius, xref(c, a)))
        }
        SurfaceKind::Cone { apex, axis, tan_half_angle } => {
            let a = norm(*axis);
            if dot(n, a).abs() < 0.99 {
                return None;
            }
            let r = dot(sub(o, *apex), a) * tan_half_angle;
            if r <= 1e-9 {
                return None;
            }
            let c = add(*apex, scale(a, dot(sub(o, *apex), a)));
            Some((c, a, r, xref(c, a)))
        }
        _ => None,
    }
}

/// Fits a circle to a vertex chain (3-point circumcircle of well-separated
/// samples) and returns `(center, unit axis, radius, unit in-plane x)` if EVERY
/// chain point lies on it within ~5% of the radius -- loose enough to accept a
/// faceted polygon's vertices (which sit a chord-sagitta inside the true circle),
/// tight enough that a non-circular chain (an airfoil profile) is rejected. Needs
/// >= 4 points; the caller gates this on curved-face adjacency.
fn fit_circle(chain: &[V3]) -> Option<(V3, V3, f64, V3)> {
    let n = chain.len();
    if n < 4 {
        return None;
    }
    let (a, b, c) = (chain[0], chain[n / 3], chain[2 * n / 3]);
    let (av, bv) = (sub(b, a), sub(c, a));
    let nrm = cross(av, bv);
    let n2 = dot(nrm, nrm);
    if n2 < 1e-24 {
        return None; // collinear sample
    }
    let axis = norm(nrm);
    // circumcenter relative to `a`: (|A|^2 (B x n) + |B|^2 (n x A)) / (2|n|^2)
    let (a2, b2) = (dot(av, av), dot(bv, bv));
    let term: V3 =
        std::array::from_fn(|k| (a2 * cross(bv, nrm)[k] + b2 * cross(nrm, av)[k]) / (2.0 * n2));
    let center: V3 = std::array::from_fn(|k| a[k] + term[k]);
    let radius = dot(term, term).sqrt();
    if !(radius > 1e-12) {
        return None;
    }
    // Reject a near-straight / gently-curved chain: a real circle's radius is
    // comparable to its own extent, but three near-collinear samples fit a huge
    // circle that a loose tolerance would wrongly accept (an airfoil arc).
    let mut ext = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            ext = ext.max(dist(chain[i], chain[j]));
        }
    }
    if radius > 3.0 * ext {
        return None;
    }
    // ~4% of the radius: accepts a faceted polygon's vertices (chord sagitta),
    // rejects a non-circular profile (deviation is far larger).
    let rtol = 0.04 * radius;
    let on_circle = chain.iter().all(|&p| {
        let d = sub(p, center);
        dot(d, axis).abs() < rtol && (dot(d, d).sqrt() - radius).abs() < rtol
    });
    if !on_circle {
        return None;
    }
    Some((center, axis, radius, norm(sub(a, center))))
}

/// Parameter on `profile` nearest to the 2D point `c` (dense sample + refine).
fn profile_footpoint(profile: &rapidmesh_geom::nurbs::NurbsCurve, c: [f64; 2]) -> f64 {
    let (t0, t1) = profile.domain();
    let n = 512usize;
    let mut best_t = t0;
    let mut best_d = f64::INFINITY;
    for i in 0..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        let p = profile.eval(t);
        let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2);
        if d < best_d {
            best_d = d;
            best_t = t;
        }
    }
    best_t
}
