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
use rapidmesh_geom::vec3::{V3, sub, add, scale, dot, cross, dist, normalize as norm};
use rapidmesh_geom::{SurfaceKind, TaggedPlc};
// Deterministic (seedless) hashers: from_plc's map ITERATION order sets the
// B-rep edge / face / vertex order, which flows into the surface point order and
// the mesh -- std's RandomState would make the whole mesh vary run to run.
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Turn cosine below which a degree-2 vertex is still a corner (45 deg), matching
/// the mesher's feature-edge splitter.
const CORNER_COS: f64 = 0.707;

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
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
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
    let mut face_of_rep: HashMap<usize, usize> = HashMap::default();
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
    let mut bedge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    {
        // count (face, edge) uses
        let mut fe_count: HashMap<(usize, (usize, usize)), usize> = HashMap::default();
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
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::default();
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
                done: &mut HashSet<(usize, usize)>|
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
    let mut done: HashSet<(usize, usize)> = HashSet::default();
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
    let mut vid: HashMap<usize, VertexId> = HashMap::default();
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
        // Frame for the surface. For a plane this must be EXACT (so on-plane
        // carriers stay bit-exact on the PLC plane -> exact region volumes): use an
        // originating facet triangle (exact PLC vertices, cross-product normal),
        // not the float edge points. Other kinds ignore the frame (self-contained).
        let frame_pts: Vec<V3> = if let Some(&tfi) = faces[fid].facets.first() {
            let t = plc.triangles[tfi as usize];
            vec![plc.vertices[t[0] as usize], plc.vertices[t[1] as usize], plc.vertices[t[2] as usize]]
        } else {
            signed.first().map(|lp| loop_points(lp, &edges)).unwrap_or_default()
        };
        let mut kind = plc.surfaces[faces[fid].surface.0 as usize].clone();
        // A `Plane` kind whose facets are NOT coplanar is a faceted CURVED
        // face without an analytic recovery -- loft mantles, swept tubes,
        // helix coils all tag their whole side wall as one Plane surface. A
        // plane fit through such a face is a garbage carrier (the refinement
        // core projects and classifies against it, shredding the mesh into
        // fragments). Carry it as a DISCRETE patch of its own facets instead:
        // the same closest-point oracle that remeshes STL imports.
        if matches!(kind, SurfaceKind::Plane) && !face_facets_coplanar(&faces[fid], plc, tol) {
            let mut vmap: HashMap<usize, u32> = HashMap::default();
            let mut dpoints: Vec<V3> = Vec::new();
            let mut dtris: Vec<[u32; 3]> = Vec::new();
            for &tfi in &faces[fid].facets {
                let t = plc.triangles[tfi as usize];
                let ids: [u32; 3] = std::array::from_fn(|k| {
                    *vmap.entry(t[k] as usize).or_insert_with(|| {
                        dpoints.push(plc.vertices[t[k] as usize]);
                        (dpoints.len() - 1) as u32
                    })
                });
                dtris.push(ids);
            }
            kind = SurfaceKind::Discrete(std::sync::Arc::new(
                rapidmesh_geom::DiscreteSurface::new(dpoints, dtris),
            ));
        }
        let sid = SurfaceId(surfaces.len() as u32);
        // One surface per face, in face order: `Curve::Intersection` (built in
        // recover_curve, before this loop) references faces' surfaces by this
        // identity, so it must hold.
        debug_assert_eq!(sid.0 as usize, fid, "surface id must equal face id");
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
    let mut adj: HashMap<u32, Vec<usize>> = HashMap::default();
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
    // Exact sphere-sphere intersection: the circle on the radical plane, derived
    // from BOTH spheres' centres/radii -- NOT a fit to the faceted chain (whose
    // sagitta error is exactly what leaves the straddler slivers). Chain-
    // independent, so the recovered circle lies exactly on both spheres.
    let spheres: Vec<(V3, f64)> = rad
        .iter()
        .filter_map(|f| match &plc.surfaces[faces[f.0 as usize].plc_surface as usize] {
            SurfaceKind::Sphere { center, radius } => Some((*center, *radius)),
            _ => None,
        })
        .collect();
    if spheres.len() >= 2 {
        if let Some((center, axis, radius, x)) =
            sphere_sphere_circle(spheres[0].0, spheres[0].1, spheres[1].0, spheres[1].1, chain)
        {
            return Curve::Circle { center, axis, radius, x };
        }
    }

    // Exact circle from an adjacent analytic surface + the chain plane, VALIDATED
    // against the chain: a non-planar intersection curve (cylinder∩cylinder) can
    // otherwise masquerade as a circle -- its Newell normal aligns with the hole
    // axis, so the perpendicularity gate alone passes, and the mesher would then
    // distribute points on a wrong circle OFF the true carrier (the cross_cyl /
    // drilled_block straddler-sliver mechanism).
    let circ = rad.iter().find_map(|f| {
        let kind = &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
        analytic_circle(chain, kind).filter(|c| circle_fits_chain(chain, c))
    });
    if let Some((center, axis, radius, x)) = circ {
        return Curve::Circle { center, axis, radius, x };
    }

    // Oblique plane section of a cylinder: an exact ELLIPSE (the axis-perpendicular
    // case is the circle above). Derived from the cylinder's parameters and the
    // EXACT facet plane of the adjacent planar face -- chain-independent, so the
    // recovered ellipse lies exactly on both carriers (a fit to the faceted chain
    // would sit a sagitta inside, the straddler-sliver mechanism).
    let planes: Vec<(V3, V3)> = rad
        .iter()
        .filter(|f| {
            matches!(plc.surfaces[faces[f.0 as usize].plc_surface as usize], SurfaceKind::Plane)
        })
        .filter_map(|f| exact_face_plane(&faces[f.0 as usize], plc))
        .collect();
    for f in rad {
        let kind = &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
        if let SurfaceKind::Cylinder { center, axis, radius } = kind {
            for &(po, pn) in &planes {
                if let Some(e) = plane_cylinder_ellipse(chain, po, pn, *center, norm(*axis), *radius, tol)
                {
                    return e;
                }
            }
        }
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

    // Two distinct analytic carriers, no closed form matched: the edge is their
    // intersection curve. The mesher densifies the chain and pulls every sample
    // onto BOTH surfaces (alternating projection), so the edge follows the true
    // curve instead of the coarse arrangement chain. Prefer two curved carriers,
    // else curved + plane; two planes intersect in a line (handled above).
    {
        let mut carriers: Vec<(bool, u32, FaceId)> = rad
            .iter()
            .map(|&f| {
                let sid = faces[f.0 as usize].plc_surface;
                (matches!(plc.surfaces[sid as usize], SurfaceKind::Plane), sid, f)
            })
            .collect();
        carriers.sort_unstable_by_key(|&(is_plane, sid, _)| (is_plane, sid));
        carriers.dedup_by_key(|c| c.1);
        if carriers.len() >= 2 && !carriers[0].0 {
            // NB: Curve::Intersection stores brep SurfaceIds; from_plc assigns one
            // surface per face IN FACE ORDER, so SurfaceId(fid) is that face's
            // surface (asserted in from_plc).
            return Curve::Intersection {
                a: SurfaceId(carriers[0].2 .0),
                b: SurfaceId(carriers[1].2 .0),
            };
        }
    }

    // Heuristic circle fit, LAST resort (after every exact/lazy-analytic form):
    // a circular chain with no recoverable carrier pair (a smooth seam inside one
    // surface). Loose by nature, so it must never shadow an exact recovery.
    let curved = rad.iter().any(|f| {
        !matches!(plc.surfaces[faces[f.0 as usize].plc_surface as usize], SurfaceKind::Plane)
    });
    if curved {
        if let Some((center, axis, radius, x)) = fit_circle(chain) {
            return Curve::Circle { center, axis, radius, x };
        }
    }

    Curve::Polyline
}

/// True if every chain point lies on the circle within 2% of its radius: accepts
/// the chord-sagitta of a faceted carrier (an icosphere equator sits < 1% inside
/// the analytic sphere), rejects a warped non-planar intersection curve (whose
/// out-of-plane deviation scales with the OTHER surface's sagitta, e.g.
/// r_small/(2 r_big) for cylinder∩cylinder).
fn circle_fits_chain(chain: &[V3], c: &(V3, V3, f64, V3)) -> bool {
    let (center, axis, radius, _) = *c;
    let tol = 0.02 * radius;
    chain.iter().all(|&p| {
        let d = sub(p, center);
        let z = dot(d, axis);
        let rho = (dot(d, d) - z * z).max(0.0).sqrt();
        z.abs() < tol && (rho - radius).abs() < tol
    })
}

/// True if every facet vertex of the face lies on the plane of its FIRST
/// facet (within `tol`): the gate that separates a real planar face from a
/// faceted curved side wall mis-tagged as `Plane`.
fn face_facets_coplanar(face: &Face, plc: &TaggedPlc, tol: f64) -> bool {
    let Some((o, n)) = exact_face_plane(face, plc) else {
        return true;
    };
    face.facets.iter().all(|&tfi| {
        let t = plc.triangles[tfi as usize];
        (0..3).all(|k| dot(sub(plc.vertices[t[k] as usize], o), n).abs() <= tol)
    })
}

/// The EXACT carrier plane of a planar face `(origin, unit normal)`, from its
/// first originating PLC facet (exact vertices, cross-product normal) -- not a
/// Newell fit to float edge points.
fn exact_face_plane(face: &Face, plc: &TaggedPlc) -> Option<(V3, V3)> {
    let &tfi = face.facets.first()?;
    let t = plc.triangles[tfi as usize];
    let (a, b, c) = (
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    );
    let n = cross(sub(b, a), sub(c, a));
    if dot(n, n) < 1e-24 {
        return None;
    }
    Some((a, norm(n)))
}

/// The exact ellipse of an oblique plane∩cylinder section, validated against the
/// chain. Plane `(po, pn)`, cylinder `(center c, unit axis ca, radius r)`:
/// the section is an ellipse with center on the cylinder axis, semi-minor `r`
/// along `ca x pn`, semi-major `r/|ca·pn|` along the axis' in-plane projection.
/// `None` when near-perpendicular (a circle, handled elsewhere), near-parallel
/// (no bounded section), or when the chain does not lie on the ellipse.
fn plane_cylinder_ellipse(
    chain: &[V3],
    po: V3,
    pn: V3,
    c: V3,
    ca: V3,
    r: f64,
    _tol: f64,
) -> Option<Curve> {
    let cosphi = dot(ca, pn);
    // Perpendicular cut (circle) or glancing cut (unbounded/degenerate): not ours.
    if cosphi.abs() > 0.99 || cosphi.abs() < 1e-3 {
        return None;
    }
    // Ellipse center: the cylinder axis pierced through the plane.
    let t = dot(sub(po, c), pn) / cosphi;
    let center = add(c, scale(ca, t));
    let minor_dir = norm(cross(ca, pn));
    let major_dir = norm(cross(pn, minor_dir));
    let (a, b) = (r / cosphi.abs(), r);
    // Validate the hypothesis against the chain, tolerating the faceted carrier:
    // chain vertices on triangle-split diagonals of the prism sit up to the chord
    // sagitta INSIDE the analytic cylinder (< 1% of r for a 24-gon), same margin
    // as `circle_fits_chain`. The recovered ellipse itself is exact -- distributing
    // on it pulls the edge ONTO the true carrier, better than the chain.
    for &p in chain {
        let d = sub(p, center);
        if dot(d, pn).abs() > 0.02 * b {
            return None;
        }
        let (x, y) = (dot(d, major_dir) / a, dot(d, minor_dir) / b);
        if ((x * x + y * y).sqrt() - 1.0).abs() > 0.02 {
            return None;
        }
    }
    Some(Curve::Ellipse { center, major: major_dir, minor: minor_dir, a, b })
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
/// Exact intersection circle of two distinct spheres: the circle on the radical
/// plane (perpendicular to the centre line, at the exact offset). Returns
/// `(center, unit axis, radius, unit in-plane x toward the chain start)`. None if
/// the spheres are concentric, tangent, or disjoint.
fn sphere_sphere_circle(c1: V3, r1: f64, c2: V3, r2: f64, chain: &[V3]) -> Option<(V3, V3, f64, V3)> {
    let dvec = sub(c2, c1);
    let d2 = dot(dvec, dvec);
    if d2 < 1e-18 {
        return None; // concentric
    }
    let d = d2.sqrt();
    let axis = scale(dvec, 1.0 / d);
    let a = (d2 + r1 * r1 - r2 * r2) / (2.0 * d);
    let rr = r1 * r1 - a * a;
    if rr <= 1e-18 {
        return None; // tangent or disjoint
    }
    let center = add(c1, scale(axis, a));
    let dchain = sub(chain[0], center);
    let x = norm(std::array::from_fn(|k| dchain[k] - axis[k] * dot(dchain, axis)));
    Some((center, axis, rr.sqrt(), x))
}

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
