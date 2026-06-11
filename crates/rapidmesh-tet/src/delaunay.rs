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

use rapidmesh_exact::Sign;
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

fn orient(pts: &[[f64; 3]], a: u32, b: u32, c: u32, d: u32) -> Sign {
    Sign::of_f64(geometry_predicates::orient3d(
        pts[a as usize],
        pts[b as usize],
        pts[c as usize],
        pts[d as usize],
    ))
}

fn insphere(pts: &[[f64; 3]], t: [u32; 4], p: u32) -> Sign {
    Sign::of_f64(geometry_predicates::insphere(
        pts[t[0] as usize],
        pts[t[1] as usize],
        pts[t[2] as usize],
        pts[t[3] as usize],
        pts[p as usize],
    ))
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
        if orient(&pts, seed[0], seed[1], seed[2], seed[3]) == Sign::Negative {
            seed.swap(2, 3);
        }
        debug_assert_eq!(orient(&pts, seed[0], seed[1], seed[2], seed[3]), Sign::Positive);
        DelaunayBuilder {
            domain: (
                std::array::from_fn(|k| c[k] - big),
                c[0] + c[1] + c[2] + big,
            ),
            pts,
            tets: vec![seed],
            neighbors: vec![[NONE; 4]],
            alive: vec![true],
            free: Vec::new(),
            last: 0,
            mark: vec![0],
            epoch: 0,
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
                orient(&self.pts, f[0], f[1], f[2], p) != Sign::Negative
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
                if orient(&self.pts, f[0], f[1], f[2], p) == Sign::Negative {
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
        let ok = self.compute_cavity(p, -1.0, usize::MAX);
        debug_assert!(ok);
        self.refill(p)
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
                if insphere(&self.pts, self.tets[nb as usize], p) == Sign::Positive {
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
                    if orient(&self.pts, f[0], f[1], f[2], p) != Sign::Positive {
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
            debug_assert_eq!(orient(&self.pts, f[0], f[1], f[2], p), Sign::Positive);
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
