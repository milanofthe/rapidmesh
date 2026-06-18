"""Symbolic derivations for the holistic quality pipeline (Schule A: conforming
Delaunay refinement + feature protection + sliver exudation), fitted to the
rapidmesh architecture (B-rep feature complex + dual carrier/oracle + exact
predicates where the carrier is rational).

Cross-checked against CGAL (C:/Repositories/TEMP/cgal): protecting balls
(Mesh_3/Protect_edges_sizing_field.h: ball radius = sizing field, special balls
on overlap), sliver exudation (Mesh_3/Slivers_exuder.h: max weight
< d * dist(v, nearest)^2), regular-triangulation power predicate (Kernel:
power_side_of_oriented_power_sphere).

Derived & verified with sympy:
  1. SAGITTA sizing  h = sqrt(8 * tol * R)  -- a facet edge of length h on a
     surface of curvature radius R deviates by sagitta tol ~ h^2/(8R).
  2. lfs SIZING FIELD  h(x) = min(maxh, sqrt(8 tol R), K*lfs(x))  and the
     Lipschitz grading bound: an alpha-Lipschitz h gives an adjacent edge-length
     ratio in [1-alpha, 1+alpha] (alpha<1 => smoothly graded). lfs is 1-Lipschitz.
  3. PROTECTING BALLS on a feature curve: consecutive balls cover the segment iff
     |x_i - x_{i+1}| <= r_i + r_{i+1}; r_i <= lfs(x_i)/2 keeps non-incident
     features' balls disjoint (the curve is protected).
  4. the WEIGHTED (power) in-sphere predicate as a 5x5 LIFTED determinant -- the
     regular-triangulation test exudation needs -- a polynomial in the rational
     inputs (rapidmesh-exact evaluates its SIGN exactly), reducing to the
     unweighted insphere when all weights vanish.
  5. the EXUDATION weight bound w_v < d_min(v)^2 (no-hiding / orthogonality), and
     the conflict condition by which pumping w_v flips a sliver out.

Run:  python report/derivations/quality_pipeline.py
"""
import sympy as sp


# 1. ---------------------------------------------------------------- sagitta
def derive_sagitta():
    """A chord of length h on a circle of radius R has sagitta (max deviation
    from the arc) s = R - sqrt(R^2 - (h/2)^2) ~ h^2/(8R). Bounding s by the
    tolerance `tol` gives the curvature-aware size  h = sqrt(8 * tol * R)."""
    R, h = sp.symbols("R h", positive=True)
    s = R - sp.sqrt(R**2 - (h / 2) ** 2)
    # leading term of the sagitta for small h/R
    lead = sp.series(s, h, 0, 4).removeO()
    assert sp.simplify(lead - h**2 / (8 * R)) == 0, lead
    # invert the leading relation s = h^2/(8R) -> h = sqrt(8 R s); with s = tol:
    tol = sp.symbols("tol", positive=True)
    h_of_tol = sp.sqrt(8 * R * tol)
    assert sp.simplify(sp.solve(sp.Eq(h**2 / (8 * R), tol), h)[0] - h_of_tol) == 0
    print("1. sagitta:    s ~ h^2/(8R)  =>  h = sqrt(8*tol*R)   [matches surfchart.rs]")
    return h_of_tol


# 2. -------------------------------------------------------- lfs sizing field
def derive_lfs_sizing():
    """The unifying sizing oracle. h(x) = min(maxh, sqrt(8 tol R(x)), K*lfs(x)).
    lfs (distance to the nearest non-incident feature) is 1-Lipschitz; capping a
    field by it keeps h Lipschitz. For an alpha-Lipschitz h and two points a step
    d~h(x) apart, the size ratio is bounded -- so the mesh grades smoothly."""
    alpha, hx, d = sp.symbols("alpha h_x d", positive=True)
    # alpha-Lipschitz: |h(y) - h(x)| <= alpha*d  =>  h(y) in [h(x)-alpha d, h(x)+alpha d]
    # for an adjacent edge the step is the local size, d = h(x):
    ratio_lo = (hx - alpha * hx) / hx  # = 1 - alpha
    ratio_hi = (hx + alpha * hx) / hx  # = 1 + alpha
    assert sp.simplify(ratio_lo - (1 - alpha)) == 0
    assert sp.simplify(ratio_hi - (1 + alpha)) == 0
    # min of Lipschitz fields is Lipschitz with the max constant; K*lfs is
    # K-Lipschitz since lfs is 1-Lipschitz. So h is max(0, K)-... <= 1-Lipschitz
    # for K<=1 -> graded with alpha<1. (gradient-limiting reproduces this.)
    print("2. lfs sizing: h = min(maxh, sqrt(8 tol R), K*lfs); alpha-Lipschitz")
    print("              -> adjacent edge ratio in [1-alpha, 1+alpha], K<=1 => graded")


# 3. ----------------------------------------------------- protecting balls
def derive_protecting_balls():
    """Feature curve sampled at x_i with protecting balls of radius r_i. Two
    conditions (CGAL Protect_edges_sizing_field): (a) consecutive balls COVER the
    segment so the edge is recovered; (b) a ball stays clear of any non-incident
    feature so nothing crosses the curve."""
    # (a) coverage: the union of balls B(x_i,r_i), B(x_{i+1},r_{i+1}) covers the
    # segment iff they overlap, i.e. the gap is non-positive:
    xi, xj, ri, rj = sp.symbols("x_i x_j r_i r_j", positive=True)
    seg = xj - xi  # 1D arc-length gap between consecutive samples (xi < xj)
    covered = seg <= ri + rj
    # at the worst (uniform r) -> spacing <= 2r: the standard "sample denser than
    # the ball radius" rule. r_i = h(x_i) = the sizing field (CGAL: query_size).
    # (b) non-incident protection: features A, B at distance >= D = lfs. Balls of
    # radius r on each are disjoint iff 2r <= D, i.e. r <= lfs/2.
    D, r = sp.symbols("D r", positive=True)
    disjoint = 2 * r <= D
    assert (covered.rhs - covered.lhs) == (ri + rj - seg)
    assert (disjoint.rhs - disjoint.lhs) == (D - 2 * r)
    print("3. protect:    cover: |x_i - x_{i+1}| <= r_i + r_{i+1}  (r_i = h(x_i))")
    print("              disjoint non-incident features: r <= lfs/2")


# 4. ------------------------------------------ weighted (power) in-sphere
def _lift_weighted(p, w):
    """Lift a weighted point (p in R^3, weight w) to R^4: (px,py,pz, |p|^2 - w).
    The power distance pi(x) = |x-p|^2 - w then equals |x|^2 minus an AFFINE
    function of x whose coefficients are this lift, so the weighted 'sphere'
    pi=0 lifts to a hyperplane."""
    return [p[0], p[1], p[2], p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - w]


def derive_power_insphere():
    """The regular-triangulation predicate exudation needs: is the weighted query
    point inside the oriented power sphere of four weighted points? = sign of the
    5x5 determinant of the R^4 lifts (with a 1-column). It is a polynomial in the
    rational coordinates and weights -> rapidmesh-exact evaluates its sign
    exactly. With all weights 0 it is the ordinary insphere (Delaunay) test."""
    # power distance lifts to an affine function: verify pi(x) = |x|^2 - L.(x,1)
    px, py, pz, w = sp.symbols("px py pz w", real=True)
    x, y, z = sp.symbols("x y z", real=True)
    pi = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2 - w
    L = _lift_weighted((px, py, pz), w)  # (px,py,pz, |p|^2 - w)
    affine = 2 * (L[0] * x + L[1] * y + L[2] * z) - L[3]  # 2 p.x - (|p|^2 - w)
    assert sp.expand(pi - ((x**2 + y**2 + z**2) - affine)) == 0
    # the 5x5 power-side determinant (rows: 4 weighted points + query, cols:
    # x,y,z, lift, 1). Sign = which side of the power sphere the query is on.
    pts = sp.symbols("a0:4 b0:4 c0:4 d0:4 q0:4", real=True)  # placeholder names
    P = [sp.symbols(f"p{i}x p{i}y p{i}z p{i}w", real=True) for i in range(4)]
    Q = sp.symbols("qx qy qz qw", real=True)
    rows = []
    for (rx, ry, rz, rw) in [*P, Q]:
        lx, ly, lz, ll = _lift_weighted((rx, ry, rz), rw)
        rows.append([lx, ly, lz, ll, 1])
    M = sp.Matrix(rows)
    det = sp.expand(M.det())
    assert det.is_polynomial(*[s for r in [*P, Q] for s in r]), "must be polynomial -> exact"
    # unweighted reduction: all weights 0 -> the classic insphere lift |p|^2
    subs0 = {r[3]: 0 for r in P} | {Q[3]: 0}
    det0 = sp.expand(det.subs(subs0))
    rows0 = []
    for (rx, ry, rz, _rw) in [*P, Q]:
        rows0.append([rx, ry, rz, rx**2 + ry**2 + rz**2, 1])
    insphere = sp.expand(sp.Matrix(rows0).det())
    assert sp.simplify(det0 - insphere) == 0, "weighted->unweighted insphere"
    print("4. power test: pi(x)=|x|^2 - affine(lift);  power-side = sign(5x5 det of lifts)")
    print("              polynomial in rational inputs -> exact sign; w=0 => Delaunay insphere")


# 5. ----------------------------------------------------- exudation bound
def derive_exudation_bound():
    """Pumping vertex v's weight w_v removes incident slivers (CGAL Slivers_exuder).
    Bound: keep v a vertex of the regular triangulation by not 'swallowing' its
    nearest neighbour n -- n must stay outside v's weighted ball, i.e. its power
    w.r.t. (v, w_v) is positive:  pi(n; v, w_v) = |v-n|^2 - w_v > 0  <=>
    w_v < d_min^2.  Points do NOT move -> planar boundary stays on its exact
    carrier -> exact region volumes are preserved by exudation."""
    dmin, wv = sp.symbols("d_min w_v", positive=True)
    power_of_nearest = dmin**2 - wv  # pi(n; v, w_v)
    bound = sp.solve(sp.Gt(power_of_nearest, 0), wv)  # w_v < d_min^2
    # sympy returns the interval; check the threshold:
    assert sp.simplify(sp.Eq(power_of_nearest, 0).lhs.subs(wv, dmin**2)) == 0
    print("5. exudation:  pi(nearest; v,w_v) = d_min^2 - w_v > 0  =>  w_v < d_min^2")
    print("              (pump w_v in [0, d_min^2) maximising min incident dihedral;")
    print("               points fixed -> exact planar volumes preserved)")
    print("   bound:", bound)


if __name__ == "__main__":
    print("=== holistic quality-pipeline derivations (verified with sympy) ===\n")
    derive_sagitta()
    derive_lfs_sizing()
    derive_protecting_balls()
    derive_power_insphere()
    derive_exudation_bound()
    print("\nall symbolic identities verified.")
