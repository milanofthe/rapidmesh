//! Incremental 3D Delaunay tetrahedralization (Bowyer-Watson) with exact
//! predicates and neighbor-pointer connectivity.
//!
//! Performance shape: tets live in a flat slab with per-face neighbor
//! pointers and a free list; point location is a visibility walk from the
//! last touched tet (with a linear-scan fallback for degenerate walks); the
//! cavity is grown through neighbor pointers; all per-insert working memory
//! comes from reusable scratch buffers. No hash maps or allocations on the
//! hot path beyond amortized scratch growth.
//!
//! Robustness shape (unchanged from the first kernel): exact orient3d /
//! insphere predicates, strict-insphere cavity, and a star-shape repair loop
//! that absorbs neighbors whose shared face does not strictly see the new
//! point — which is what makes heavily cospherical / on-face grid geometry
//! safe.

use rapidmesh_exact::{Point3, Sign};
use std::sync::atomic::{AtomicU64, Ordering};

/// Diagnostic counters (RAPIDMESH_CAND_TRACE): linear-scan point locations
/// and guarded-insert outcomes.
pub static LOCATE_SCANS: AtomicU64 = AtomicU64::new(0);
pub static GUARDED_NN_BAILS: AtomicU64 = AtomicU64::new(0);
pub static GUARDED_KEEP_VETOES: AtomicU64 = AtomicU64::new(0);

/// A Delaunay tetrahedralization of a point set.
#[derive(Debug)]
pub struct DelaunayTets {
    /// The input points (super-tet corners excluded).
    pub points: Vec<[f64; 3]>,
    /// Positively oriented tets as point indices.
    pub tets: Vec<[usize; 4]>,
}

const NONE: u32 = u32::MAX;

/// A simplex an insertion would remove from the triangulation (public
/// vertex indices). See [`DelaunayBuilder::insert_guarded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removal {
    /// An interior cavity face, vertices sorted ascending.
    Face([usize; 3]),
    /// A fully interior edge, as (min, max).
    Edge(usize, usize),
}

/// True if the vertex carries an implicit exact position (CDT Steiner
/// points); explicit vertices keep the fast Shewchuk path.
fn is_implicit(exact: &[Option<Point3>], i: u32) -> bool {
    exact.get(i as usize).is_some_and(|e| e.is_some())
}

fn pt3(pts: &[[f64; 3]], exact: &[Option<Point3>], i: u32) -> Point3 {
    match exact.get(i as usize).and_then(|e| e.clone()) {
        Some(p) => p,
        None => Point3::Explicit(pts[i as usize]),
    }
}

fn orient(pts: &[[f64; 3]], exact: &[Option<Point3>], a: u32, b: u32, c: u32, d: u32) -> Sign {
    if !(is_implicit(exact, a)
        || is_implicit(exact, b)
        || is_implicit(exact, c)
        || is_implicit(exact, d))
    {
        return Sign::of_f64(geometry_predicates::orient3d(
            pts[a as usize],
            pts[b as usize],
            pts[c as usize],
            pts[d as usize],
        ));
    }
    rapidmesh_exact::orient3d(
        &pt3(pts, exact, a),
        &pt3(pts, exact, b),
        &pt3(pts, exact, c),
        &pt3(pts, exact, d),
    )
    .expect("implicit DT vertex must be a valid point")
}

fn insphere(pts: &[[f64; 3]], exact: &[Option<Point3>], t: [u32; 4], p: u32) -> Sign {
    if !(is_implicit(exact, t[0])
        || is_implicit(exact, t[1])
        || is_implicit(exact, t[2])
        || is_implicit(exact, t[3])
        || is_implicit(exact, p))
    {
        return Sign::of_f64(geometry_predicates::insphere(
            pts[t[0] as usize],
            pts[t[1] as usize],
            pts[t[2] as usize],
            pts[t[3] as usize],
            pts[p as usize],
        ));
    }
    rapidmesh_exact::insphere3d(
        &pt3(pts, exact, t[0]),
        &pt3(pts, exact, t[1]),
        &pt3(pts, exact, t[2]),
        &pt3(pts, exact, t[3]),
        &pt3(pts, exact, p),
    )
    .expect("implicit DT vertex must be a valid point")
}

/// True if `g` is an even rotation of `f` (same oriented triangle).
fn is_same_cycle(f: [u32; 3], g: [u32; 3]) -> bool {
    g == f || g == [f[1], f[2], f[0]] || g == [f[2], f[0], f[1]]
}

/// True if `g` is the reversal of `f` (same triangle, opposite orientation).
fn is_reversed(f: [u32; 3], g: [u32; 3]) -> bool {
    is_same_cycle([f[0], f[2], f[1]], g)
}

/// Face of a positively oriented tet opposite vertex `i`, wound so the
/// opposite vertex lies on its positive side.
fn face(t: [u32; 4], i: usize) -> [u32; 3] {
    match i {
        0 => [t[1], t[3], t[2]],
        1 => [t[0], t[2], t[3]],
        2 => [t[0], t[3], t[1]],
        _ => [t[0], t[1], t[2]],
    }
}

/// Incremental Delaunay tetrahedralization. Internal indices 0..4 are the
/// super-tet corners; public indices count inserted points from 0.
pub struct DelaunayBuilder {
    /// The super-tet interior: per-axis lower bounds and the upper bound on
    /// the coordinate sum (the four face planes of the super-tet).
    domain: ([f64; 3], f64),
    pts: Vec<[f64; 3]>,
    /// Exact implicit position per vertex (`None` = the f64 in `pts` IS the
    /// point). Predicates touching a `Some` vertex take the staged-exact
    /// path; `pts` then only caches `approx()` for walk heuristics.
    exact: Vec<Option<Point3>>,
    tets: Vec<[u32; 4]>,
    /// neighbors[t][i] = tet across the face opposite vertex i (NONE at the
    /// super-tet hull).
    neighbors: Vec<[u32; 4]>,
    alive: Vec<bool>,
    free: Vec<u32>,
    /// Walk start hint.
    last: u32,
    /// Epoch marks (cavity membership) per tet slot.
    mark: Vec<u32>,
    epoch: u32,
    /// Per vertex (internal index): some alive tet slot containing it,
    /// refreshed on every refill. A vertex removed from a cavity always
    /// reappears in that insert's new tets (otherwise it would vanish from
    /// the triangulation), so hints never go stale.
    vert_hint: Vec<u32>,
    // Scratch buffers (reused across inserts).
    cavity: Vec<u32>,
    /// Boundary faces as (cavity tet, face index).
    boundary: Vec<(u32, u8)>,
    /// Edge -> (new tet, face slot) for wiring new tets to each other.
    edge_map: rustc_hash::FxHashMap<(u32, u32), (u32, u8)>,
    new_tets: Vec<u32>,
    /// Guarded-insert scratch: surviving cavity-boundary edges.
    scratch_edges: rustc_hash::FxHashSet<(u32, u32)>,
    /// Guarded-insert scratch: cavity-boundary faces as (tet, face slot).
    scratch_bfaces: rustc_hash::FxHashSet<(u32, u8)>,
}

impl DelaunayBuilder {
    /// A builder whose super-tetrahedron comfortably encloses the given
    /// bounding box; every inserted point must lie inside that box.
    pub fn enclosing(lo: [f64; 3], hi: [f64; 3]) -> DelaunayBuilder {
        let c: [f64; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let d = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
        let big = 64.0 * d;
        let pts = vec![
            [c[0] - big, c[1] - big, c[2] - big],
            [c[0] + 3.0 * big, c[1] - big, c[2] - big],
            [c[0] - big, c[1] + 3.0 * big, c[2] - big],
            [c[0] - big, c[1] - big, c[2] + 3.0 * big],
        ];
        let mut seed = [0u32, 1, 2, 3];
        if orient(&pts, &[], seed[0], seed[1], seed[2], seed[3]) == Sign::Negative {
            seed.swap(2, 3);
        }
        debug_assert_eq!(orient(&pts, &[], seed[0], seed[1], seed[2], seed[3]), Sign::Positive);
        DelaunayBuilder {
            domain: (
                std::array::from_fn(|k| c[k] - big),
                c[0] + c[1] + c[2] + big,
            ),
            pts,
            exact: vec![None; 4],
            tets: vec![seed],
            neighbors: vec![[NONE; 4]],
            alive: vec![true],
            free: Vec::new(),
            last: 0,
            mark: vec![0],
            epoch: 0,
            vert_hint: vec![0; 4],
            cavity: Vec::new(),
            boundary: Vec::new(),
            edge_map: rustc_hash::FxHashMap::default(),
            new_tets: Vec::new(),
            scratch_edges: rustc_hash::FxHashSet::default(),
            scratch_bfaces: rustc_hash::FxHashSet::default(),
        }
    }

    /// Number of inserted points.
    pub fn len(&self) -> usize {
        self.pts.len() - 4
    }

    /// True if no points were inserted yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn locate_scan(&self, p: u32) -> u32 {
        LOCATE_SCANS.fetch_add(1, Ordering::Relaxed);
        for (ti, &t) in self.tets.iter().enumerate() {
            if !self.alive[ti] {
                continue;
            }
            if (0..4).all(|i| {
                let f = face(t, i);
                orient(&self.pts, &self.exact, f[0], f[1], f[2], p) != Sign::Negative
            }) {
                return ti as u32;
            }
        }
        panic!("point must lie inside the super-tet");
    }

    /// Visibility walk from the last touched tet, with scan fallback for
    /// degenerate walks.
    fn locate(&self, p: u32) -> u32 {
        let mut cur = self.last;
        if !self.alive[cur as usize] {
            return self.locate_scan(p);
        }
        let mut prev = NONE;
        let mut steps = 0usize;
        'walk: loop {
            steps += 1;
            if steps > self.tets.len() + 64 {
                return self.locate_scan(p);
            }
            let t = self.tets[cur as usize];
            let mut fallback: Option<u32> = None;
            for i in 0..4 {
                let nb = self.neighbors[cur as usize][i];
                let f = face(t, i);
                if orient(&self.pts, &self.exact, f[0], f[1], f[2], p) == Sign::Negative {
                    if nb == NONE {
                        return self.locate_scan(p);
                    }
                    if nb != prev {
                        prev = cur;
                        cur = nb;
                        continue 'walk;
                    }
                    fallback = Some(nb);
                }
            }
            if let Some(nb) = fallback {
                // Only the way we came is negative: degenerate ping-pong.
                let _ = nb;
                return self.locate_scan(p);
            }
            return cur;
        }
    }

    fn alloc(&mut self, t: [u32; 4]) -> u32 {
        if let Some(slot) = self.free.pop() {
            self.tets[slot as usize] = t;
            self.neighbors[slot as usize] = [NONE; 4];
            self.alive[slot as usize] = true;
            self.mark[slot as usize] = 0;
            slot
        } else {
            self.tets.push(t);
            self.neighbors.push([NONE; 4]);
            self.alive.push(true);
            self.mark.push(0);
            (self.tets.len() - 1) as u32
        }
    }

    /// Inserts a point and returns its public index.
    pub fn insert(&mut self, point: [f64; 3]) -> usize {
        let p = self.pts.len() as u32;
        self.pts.push(point);
        self.exact.push(None);
        let ok = self.compute_cavity(p, -1.0, usize::MAX);
        debug_assert!(ok);
        self.refill(p)
    }

    /// Inserts an implicit exact point (a CDT Steiner point): every predicate
    /// involving it evaluates the exact position, `pts` only caches the
    /// rounded approximation for walk heuristics. The point must be valid
    /// and its approximation must lie inside the builder's domain box.
    pub fn insert_exact(&mut self, point: Point3) -> usize {
        if let Some(c) = point.as_explicit() {
            return self.insert(c);
        }
        let approx = point.approx().expect("implicit DT vertex must be valid");
        let p = self.pts.len() as u32;
        self.pts.push(approx);
        self.exact.push(Some(point));
        let ok = self.compute_cavity(p, -1.0, usize::MAX);
        debug_assert!(ok);
        self.refill(p)
    }

    /// The exact position of an inserted point (public index): the implicit
    /// point where one was inserted, the explicit f64 otherwise.
    pub fn exact_point(&self, i: usize) -> Point3 {
        pt3(&self.pts, &self.exact, (i + 4) as u32)
    }

    /// The f64 position of an inserted point (public index): the inserted
    /// coordinates for explicit points, the cached approximation for
    /// implicit ones.
    pub fn approx_point(&self, i: usize) -> [f64; 3] {
        self.pts[i + 4]
    }

    /// Some alive slot whose tet contains the internal vertex, from the
    /// refill-maintained hint (with a linear rescue scan as backstop).
    fn star_seed(&self, vi: u32) -> u32 {
        let valid = |s: u32| {
            s != NONE && self.alive[s as usize] && self.tets[s as usize].contains(&vi)
        };
        let hint = self.vert_hint[vi as usize];
        if valid(hint) {
            return hint;
        }
        debug_assert!(false, "stale vertex hint for {vi}");
        (0..self.tets.len() as u32)
            .find(|&s| valid(s))
            .expect("inserted vertex must appear in some alive tet")
    }

    /// All alive tet slots containing the vertex (public index), super-corner
    /// tets included. The star of any inserted vertex is face-connected, so a
    /// BFS through the neighbors sharing the vertex enumerates it completely.
    pub fn star_slots(&self, v: usize) -> Vec<u32> {
        let vi = (v + 4) as u32;
        let mut out = vec![self.star_seed(vi)];
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert(out[0]);
        let mut head = 0;
        while head < out.len() {
            let s = out[head];
            head += 1;
            let t = self.tets[s as usize];
            for (k, &tv) in t.iter().enumerate() {
                if tv == vi {
                    continue;
                }
                let nb = self.neighbors[s as usize][k];
                if nb != NONE && seen.insert(nb) {
                    debug_assert!(self.tets[nb as usize].contains(&vi));
                    out.push(nb);
                }
            }
        }
        out
    }

    /// True if (i, j) is an edge of the current triangulation (public
    /// indices): a BFS over the star of `i` looking for a tet containing `j`.
    pub fn edge_exists(&self, i: usize, j: usize) -> bool {
        let (vi, vj) = ((i + 4) as u32, (j + 4) as u32);
        let mut queue = vec![self.star_seed(vi)];
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert(queue[0]);
        let mut head = 0;
        while head < queue.len() {
            let s = queue[head];
            head += 1;
            let t = self.tets[s as usize];
            if t.contains(&vj) {
                return true;
            }
            for (k, &tv) in t.iter().enumerate() {
                if tv == vi {
                    continue;
                }
                let nb = self.neighbors[s as usize][k];
                if nb != NONE && seen.insert(nb) {
                    queue.push(nb);
                }
            }
        }
        false
    }

    /// Inserts a point unless `keep` rejects one of the faces or edges the
    /// insertion would REMOVE from the triangulation (public vertex indices;
    /// faces sorted ascending, edges as (min, max)). On rejection the
    /// triangulation is left untouched and `None` is returned.
    ///
    /// This is how refinement keeps recovered constraints intact: a quality
    /// insertion whose cavity would swallow a constraint face or crease edge
    /// is simply not performed (encroachment tests on diametral balls are
    /// only sufficient for Gabriel simplices; weakly Delaunay constraints
    /// can be knocked out by points outside every ball).
    /// `min_dist2`: minimum allowed SQUARED distance from the new point to
    /// every existing vertex it would connect to (the cavity boundary).
    /// Numerically corrupted circumcenters of near-degenerate tets land
    /// arbitrarily close to existing points; without this floor they seed
    /// ever-shorter edges and refinement cascades below the input scale.
    pub fn insert_guarded(
        &mut self,
        point: [f64; 3],
        min_dist2: f64,
        mut keep: impl FnMut(Removal) -> bool,
    ) -> Option<usize> {
        // Circumcenters of near-degenerate tets can land anywhere, including
        // outside the super-tet; a guarded insert simply declines those.
        if (0..3).any(|k| point[k] <= self.domain.0[k])
            || point[0] + point[1] + point[2] >= self.domain.1
        {
            return None;
        }
        let p = self.pts.len() as u32;
        self.pts.push(point);
        // Legitimate refinement cavities hold a few dozen tets; pathological
        // ones (corrupted circumcenters skimming the boundary) grow into the
        // hundreds and pay an exact insphere per tet. Capping rejects them
        // early; a wrongly capped legitimate insert is merely skipped.
        if !self.compute_cavity(p, min_dist2, 1024) {
            GUARDED_NN_BAILS.fetch_add(1, Ordering::Relaxed);
            self.pts.pop();
            self.exact.pop();
            return None;
        }

        // Surviving edges: every edge of a cavity-boundary face remains (the
        // face is re-coned to p). An edge of a cavity tet on NO boundary
        // face is interior and vanishes; a face of a cavity tet that is not
        // a boundary face is interior and vanishes.
        self.scratch_edges.clear();
        self.scratch_bfaces.clear();
        for &(ti, fi) in &self.boundary {
            self.scratch_bfaces.insert((ti, fi));
            let f = face(self.tets[ti as usize], fi as usize);
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                self.scratch_edges.insert((a.min(b), a.max(b)));
            }
        }
        for ci in 0..self.cavity.len() {
            let ti = self.cavity[ci];
            let t = self.tets[ti as usize];
            for i in 0..4 {
                if self.scratch_bfaces.contains(&(ti, i as u8)) {
                    continue;
                }
                let f = face(t, i);
                if f.iter().any(|&v| v < 4) {
                    continue; // touches a super-tet corner: never a constraint
                }
                let mut pf = f.map(|v| (v - 4) as usize);
                pf.sort_unstable();
                if !keep(Removal::Face(pf)) {
                    GUARDED_KEEP_VETOES.fetch_add(1, Ordering::Relaxed);
                    self.pts.pop();
            self.exact.pop();
                    return None;
                }
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
                    if a < 4 {
                        continue;
                    }
                    if self.scratch_edges.contains(&(a, b)) {
                        continue;
                    }
                    if !keep(Removal::Edge((a - 4) as usize, (b - 4) as usize)) {
                        GUARDED_KEEP_VETOES.fetch_add(1, Ordering::Relaxed);
                        self.pts.pop();
            self.exact.pop();
                        return None;
                    }
                }
            }
        }
        Some(self.refill(p))
    }

    /// Grows the cavity of `p` (strict circumsphere violations plus the
    /// star-shape repair) into `self.cavity` and its boundary faces into
    /// `self.boundary`. Read-only apart from scratch state. Returns `false`
    /// (cavity state undefined) as soon as a vertex of a cavity tet is
    /// closer to `p` than `min_dist2` allows: the nearest neighbor of `p`
    /// is always among the cavity vertices, so the bail is exact, and it
    /// fires before most of the exact insphere work for rejected points.
    fn compute_cavity(&mut self, p: u32, min_dist2: f64, max_cavity: usize) -> bool {
        let start = self.locate(p);
        let too_close = |pts: &[[f64; 3]], t: [u32; 4]| -> bool {
            t.iter().any(|&v| {
                v >= 4 && {
                    let q = pts[v as usize];
                    let x = pts[p as usize];
                    (0..3).map(|k| (x[k] - q[k]).powi(2)).sum::<f64>() < min_dist2
                }
            })
        };
        if too_close(&self.pts, self.tets[start as usize]) {
            return false;
        }

        // Cavity: strict circumsphere violations, grown through neighbors.
        self.epoch += 1;
        let epoch = self.epoch;
        self.cavity.clear();
        self.cavity.push(start);
        self.mark[start as usize] = epoch;
        let mut head = 0;
        while head < self.cavity.len() {
            let ti = self.cavity[head];
            head += 1;
            for i in 0..4 {
                let nb = self.neighbors[ti as usize][i];
                if nb == NONE || self.mark[nb as usize] == epoch {
                    continue;
                }
                if insphere(&self.pts, &self.exact, self.tets[nb as usize], p) == Sign::Positive {
                    if too_close(&self.pts, self.tets[nb as usize])
                        || self.cavity.len() >= max_cavity
                    {
                        return false;
                    }
                    self.mark[nb as usize] = epoch;
                    self.cavity.push(nb);
                }
            }
        }
        // Star-shape repair: absorb neighbors across faces that do not see p
        // strictly (handles on-face inserts and cospherical clusters).
        loop {
            let mut grew = false;
            let mut idx = 0;
            while idx < self.cavity.len() {
                let ti = self.cavity[idx];
                idx += 1;
                let t = self.tets[ti as usize];
                for i in 0..4 {
                    let nb = self.neighbors[ti as usize][i];
                    if nb != NONE && self.mark[nb as usize] == epoch {
                        continue;
                    }
                    let f = face(t, i);
                    if orient(&self.pts, &self.exact, f[0], f[1], f[2], p) != Sign::Positive {
                        let nb = if nb == NONE {
                            panic!("cavity reached the super-tet hull")
                        } else {
                            nb
                        };
                        if too_close(&self.pts, self.tets[nb as usize]) {
                            return false;
                        }
                        self.mark[nb as usize] = epoch;
                        self.cavity.push(nb);
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }

        // Boundary faces of the cavity.
        self.boundary.clear();
        for ci in 0..self.cavity.len() {
            let ti = self.cavity[ci];
            for i in 0..4 {
                let nb = self.neighbors[ti as usize][i];
                if nb == NONE || self.mark[nb as usize] != epoch {
                    self.boundary.push((ti, i as u8));
                }
            }
        }
        true
    }

    /// Re-fill: cone p to each boundary face, wiring neighbors locally.
    fn refill(&mut self, p: u32) -> usize {
        self.edge_map.clear();
        self.new_tets.clear();
        for bi in 0..self.boundary.len() {
            let (ti, fi) = self.boundary[bi];
            let outside = self.neighbors[ti as usize][fi as usize];
            let f = face(self.tets[ti as usize], fi as usize);
            debug_assert_eq!(orient(&self.pts, &self.exact, f[0], f[1], f[2], p), Sign::Positive);
            let nt = self.alloc([f[0], f[1], f[2], p]);
            self.new_tets.push(nt);
            // Across the boundary face (slot 3 = opposite p).
            self.neighbors[nt as usize][3] = outside;
            if outside != NONE {
                let on = &mut self.neighbors[outside as usize];
                let slot = (0..4)
                    .find(|&k| on[k] == ti)
                    .expect("outside tet must point at the cavity tet");
                on[slot] = nt;
            }
            // Between new tets: shared faces contain p and one boundary edge.
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                let key = (a.min(b), a.max(b));
                // Face of nt containing edge (a, b) and p is opposite the
                // third face vertex, which sits at slot (e + 2) % 3.
                let slot = ((e + 2) % 3) as u8;
                match self.edge_map.entry(key) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        let (ot, oslot) = *o.get();
                        self.neighbors[nt as usize][slot as usize] = ot;
                        self.neighbors[ot as usize][oslot as usize] = nt;
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert((nt, slot));
                    }
                }
            }
        }
        // Retire the cavity.
        for ci in 0..self.cavity.len() {
            let ti = self.cavity[ci];
            self.alive[ti as usize] = false;
            self.free.push(ti);
        }
        // Refresh vertex hints: every vertex of a new tet (cavity-boundary
        // vertices and p itself) points at a tet created this insert.
        self.vert_hint.resize(self.pts.len(), NONE);
        for &nt in &self.new_tets {
            for &v in &self.tets[nt as usize] {
                self.vert_hint[v as usize] = nt;
            }
        }
        self.last = *self.new_tets.last().expect("cavity has boundary faces");
        (p - 4) as usize
    }

    /// Current real tets (super-corner tets excluded), in public indices.
    pub fn tets(&self) -> Vec<[usize; 4]> {
        self.tets
            .iter()
            .enumerate()
            .filter(|&(ti, t)| self.alive[ti] && t.iter().all(|&v| v >= 4))
            .map(|(_, t)| std::array::from_fn(|k| (t[k] - 4) as usize))
            .collect()
    }

    /// Like [`DelaunayBuilder::tets`], but each tet paired with its stable
    /// internal slot id (valid until the slot is freed by a later insert).
    /// Slot-keyed bookkeeping lets callers track tets incrementally through
    /// [`DelaunayBuilder::last_removed`] / [`DelaunayBuilder::last_created`].
    pub fn tets_with_slots(&self) -> Vec<(u32, [usize; 4])> {
        self.tets
            .iter()
            .enumerate()
            .filter(|&(ti, t)| self.alive[ti] && t.iter().all(|&v| v >= 4))
            .map(|(ti, t)| (ti as u32, std::array::from_fn(|k| (t[k] - 4) as usize)))
            .collect()
    }

    /// Number of tet slots ever allocated (alive or free).
    pub fn slot_count(&self) -> usize {
        self.tets.len()
    }

    /// The tet in `slot` as public vertex indices, `None` if the slot is
    /// dead or the tet touches a super corner.
    pub fn tet_at(&self, slot: u32) -> Option<[usize; 4]> {
        let t = self.tets[slot as usize];
        if !self.alive[slot as usize] || t.iter().any(|&v| v < 4) {
            return None;
        }
        Some(std::array::from_fn(|k| (t[k] - 4) as usize))
    }

    /// The tet slots the LAST successful insert removed (its cavity), valid
    /// until the next insert. Vertex data of removed slots stays readable
    /// via [`DelaunayBuilder::removed_verts`] until a later insert reuses
    /// the slot.
    pub fn last_removed(&self) -> &[u32] {
        &self.cavity
    }

    /// The tet slots the LAST successful insert created, parallel to
    /// [`DelaunayBuilder::last_parents`]: created tet i is the cone over
    /// cavity-boundary face i and geometrically replaces (part of) that
    /// face's removed owner.
    pub fn last_created(&self) -> &[u32] {
        &self.new_tets
    }

    /// For each created tet of the last insert, the REMOVED cavity slot it
    /// was coned out of (the inside owner of its base face).
    pub fn last_parents(&self) -> impl Iterator<Item = u32> + '_ {
        self.boundary.iter().map(|&(ti, _)| ti)
    }

    /// Vertex indices of a (possibly dead) slot, `None` per vertex for super
    /// corners. Used to identify constraint faces of removed tets (a removed
    /// super-corner tet still owns one all-real face, e.g. a hull tile).
    pub fn verts_of_slot(&self, slot: u32) -> [Option<usize>; 4] {
        self.tets[slot as usize].map(|v| (v >= 4).then(|| (v - 4) as usize))
    }

    /// Position of the slot's single super corner, `None` for all-real tets
    /// (tets with 2+ super corners never border an all-real face).
    pub fn super_corner(&self, slot: u32) -> Option<[f64; 3]> {
        let t = self.tets[slot as usize];
        let supers: Vec<usize> = (0..4).filter(|&i| t[i] < 4).collect();
        (supers.len() == 1).then(|| self.pts[t[supers[0]] as usize])
    }

    /// The neighbor slot across the face of `slot` opposite its vertex `i`
    /// (`None` at the super-tet hull). Face vertices are the tet's other
    /// three vertices.
    pub fn neighbor_at(&self, slot: u32, i: usize) -> Option<u32> {
        let nb = self.neighbors[slot as usize][i];
        (nb != NONE).then_some(nb)
    }

    /// Replaces a set of alive tets with an explicit retetrahedrization of
    /// the same region (CDT face recovery: cavity removal + gift wrapping).
    /// `removed` are alive slots; `new_tets` are PUBLIC vertex indices.
    ///
    /// The replacement is verified before it is applied, by the 3-chain
    /// degree argument: if every new tet is exactly positively oriented and
    /// the oriented boundary of the new complex equals the oriented boundary
    /// of the removed region (internal faces cancel in opposite pairs,
    /// boundary faces match exactly once), then the covering degree of the
    /// new chain is 1 inside the cavity and 0 outside — an exact tiling,
    /// with no interpenetration possible. Violations panic; the builder is
    /// left untouched in that case only up to this validation (no partial
    /// state is written before it passes).
    pub fn replace_cavity(&mut self, removed: &[u32], new_tets: &[[usize; 4]]) {
        let removed_set: rustc_hash::FxHashSet<u32> = removed.iter().copied().collect();
        assert_eq!(removed_set.len(), removed.len(), "duplicate removed slot");
        for &s in removed {
            assert!(self.alive[s as usize], "removed slot {s} is not alive");
        }

        // Oriented boundary of the removed region, keyed by sorted vertex
        // triple, valued with the oriented face (as seen from inside the
        // region) and the outside neighbor.
        let mut boundary: rustc_hash::FxHashMap<[u32; 3], ([u32; 3], u32)> =
            rustc_hash::FxHashMap::default();
        for &s in removed {
            let t = self.tets[s as usize];
            for i in 0..4 {
                let nb = self.neighbors[s as usize][i];
                if nb != NONE && removed_set.contains(&nb) {
                    continue;
                }
                let f = face(t, i);
                let mut key = f;
                key.sort_unstable();
                let prev = boundary.insert(key, (f, nb));
                assert!(prev.is_none(), "removed region has a pinched boundary face");
            }
        }

        // Validate the new complex: positive orientation and oriented face
        // balance (internal faces appear once per orientation; boundary
        // faces appear once, oriented as from inside).
        let mut open: rustc_hash::FxHashMap<[u32; 3], [u32; 3]> = rustc_hash::FxHashMap::default();
        for t in new_tets {
            let ti: [u32; 4] = std::array::from_fn(|k| (t[k] + 4) as u32);
            assert_eq!(
                orient(&self.pts, &self.exact, ti[0], ti[1], ti[2], ti[3]),
                Sign::Positive,
                "replacement tet {t:?} is not positively oriented",
            );
            for i in 0..4 {
                let f = face(ti, i);
                let mut key = f;
                key.sort_unstable();
                match open.entry(key) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        // Two new tets share this face: orientations must be
                        // opposite (an even permutation of the reversal).
                        let g = *o.get();
                        assert!(
                            is_reversed(f, g),
                            "replacement tets agree on face orientation {f:?}",
                        );
                        o.remove();
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(f);
                    }
                }
            }
        }
        assert_eq!(
            open.len(),
            boundary.len(),
            "replacement boundary face count mismatch",
        );
        for (key, f) in &open {
            let (bf, _) = boundary
                .get(key)
                .unwrap_or_else(|| panic!("replacement face {f:?} not on the cavity boundary"));
            assert!(
                is_same_cycle(*f, *bf),
                "replacement boundary face {f:?} has wrong orientation",
            );
        }

        // Apply: retire removed slots, allocate the new tets, wire neighbors.
        for &s in removed {
            self.alive[s as usize] = false;
            self.free.push(s);
        }
        let slots: Vec<u32> = new_tets
            .iter()
            .map(|t| self.alloc(std::array::from_fn(|k| (t[k] + 4) as u32)))
            .collect();
        let mut face_map: rustc_hash::FxHashMap<[u32; 3], (u32, u8)> =
            rustc_hash::FxHashMap::default();
        for &nt in &slots {
            let t = self.tets[nt as usize];
            for i in 0..4 {
                let f = face(t, i);
                let mut key = f;
                key.sort_unstable();
                match face_map.entry(key) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        let (ot, oi) = *o.get();
                        self.neighbors[nt as usize][i] = ot;
                        self.neighbors[ot as usize][oi as usize] = nt;
                        o.remove();
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert((nt, i as u8));
                    }
                }
            }
        }
        // Remaining unmatched faces are the region boundary: wire to the
        // surviving outside tets (matching by face, not by stale pointer:
        // an outside tet can border the region on several faces).
        for (key, (nt, i)) in face_map {
            let (_, outside) = boundary[&key];
            self.neighbors[nt as usize][i as usize] = outside;
            if outside != NONE {
                let ot = self.tets[outside as usize];
                let slot = (0..4)
                    .find(|&k| {
                        let mut f = face(ot, k);
                        f.sort_unstable();
                        f == key
                    })
                    .expect("outside tet must own the boundary face");
                self.neighbors[outside as usize][slot] = nt;
            }
        }
        // Refresh hints and the walk start.
        self.vert_hint.resize(self.pts.len(), NONE);
        for &nt in &slots {
            for &v in &self.tets[nt as usize] {
                self.vert_hint[v as usize] = nt;
            }
        }
        self.last = *slots.first().expect("replacement must not be empty");
    }

    /// The all-real faces of super-corner tets: the convex-hull boundary of
    /// the inserted points, each paired with the position of the super corner
    /// on its far side. A real face can border at most one super tet (the
    /// neighbor's fourth vertex), so faces here have exactly one real owner
    /// in [`DelaunayBuilder::tets`]; the super-corner position lets callers
    /// resolve which side of a constraint plane the outside lies on.
    pub fn hull_faces(&self) -> Vec<([usize; 3], [f64; 3])> {
        let mut out = Vec::new();
        for (ti, t) in self.tets.iter().enumerate() {
            if !self.alive[ti] {
                continue;
            }
            let supers: Vec<usize> = (0..4).filter(|&i| t[i] < 4).collect();
            if supers.len() != 1 {
                continue;
            }
            let f = face(*t, supers[0]);
            out.push((
                std::array::from_fn(|k| (f[k] - 4) as usize),
                self.pts[t[supers[0]] as usize],
            ));
        }
        out
    }
}

/// Exact Delaunay tetrahedralization of `points` (duplicates not allowed;
/// fully coplanar input yields no tets).
pub fn tetrahedralize(points: &[[f64; 3]]) -> DelaunayTets {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut b = DelaunayBuilder::enclosing(lo, hi);
    for &p in points {
        b.insert(p);
    }
    DelaunayTets {
        points: points.to_vec(),
        tets: b.tets(),
    }
}
