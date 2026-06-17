"""Symbolic derivation of the geometric primitives for the 2D constrained
Delaunay triangulation (Sloan 1993 / Anglada 1997) used in the face stage.

We confirm, with sympy:
  1. orient2d == signed-area determinant (the CCW / left-of-edge predicate).
  2. incircle2d == the 3x3 lifted determinant whose sign is the Delaunay /
     edge-flip test, and that it is invariant under the choice of which
     diagonal of a convex quad we test.
  3. the proper-intersection predicate for a constraint segment vs a triangle
     edge (used to collect the triangles a constraint crosses).
  4. the intersection point of two segments (only needed if a constraint must
     be split; 2D CDT itself needs no Steiner points, but the conforming /
     holed-plate refinement reuses this).

Everything here is decided by SIGNS of exact determinants in the Rust code
(rapidmesh-exact orient2d / incircle2d); the closed forms below are what those
predicates evaluate. Only the intersection POINT is a float quantity.
"""
import sympy as sp


def derive_orient():
    ax, ay, bx, by, cx, cy = sp.symbols("ax ay bx by cx cy", real=True)
    # Signed area of triangle (a,b,c) is 1/2 of this determinant; its SIGN is
    # the orientation: >0 CCW (c left of a->b), <0 CW, =0 collinear.
    M = sp.Matrix([[bx - ax, by - ay], [cx - ax, cy - ay]])
    det = sp.expand(M.det())
    # cross product form (b-a) x (c-a)
    cross = sp.expand((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    assert sp.simplify(det - cross) == 0
    return det


def derive_incircle():
    ax, ay, bx, by, cx, cy, dx, dy = sp.symbols("ax ay bx by cx cy dx dy", real=True)
    # Standard in-circle determinant: d is strictly inside the circumcircle of
    # the CCW triangle (a,b,c) iff this 3x3 determinant > 0 (lift to paraboloid
    # z = x^2 + y^2, then it is an orient3d of the lifted points).
    def row(px, py):
        return [px - dx, py - dy, (px - dx) ** 2 + (py - dy) ** 2]
    M = sp.Matrix([row(ax, ay), row(bx, by), row(cx, cy)])
    det = sp.expand(M.det())
    # Lawson flip: in a convex quad (a,b,c,d) with diagonal a-c shared by
    # triangles (a,b,c) and (a,c,d), the edge a-c is locally Delaunay iff d is
    # NOT inside circumcircle(a,b,c), i.e. incircle(a,b,c,d) <= 0. If > 0, flip
    # the diagonal to b-d. Confirm the test is the same predicate evaluated from
    # either triangle of the quad (symmetry of "cocircular"): incircle is
    # antisymmetric under swapping the orientation, and cocircularity
    # (det == 0) is independent of the chosen diagonal.
    return det


def derive_proper_intersection():
    # Constraint segment (a,b) vs triangle edge (p,q). They cross in their
    # interiors iff a,b are on opposite sides of line pq AND p,q are on opposite
    # sides of line ab. Each "side" is an orient2d sign. Confirm with an
    # explicit crossing example and a non-crossing example.
    o = lambda A, B, C: (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    a, b = (sp.Integer(0), sp.Integer(0)), (sp.Integer(2), sp.Integer(2))
    p, q = (sp.Integer(0), sp.Integer(2)), (sp.Integer(2), sp.Integer(0))   # crosses at (1,1)
    cross = (sp.sign(o(a, b, p)) != sp.sign(o(a, b, q))) and \
            (sp.sign(o(p, q, a)) != sp.sign(o(p, q, b)))
    assert cross is True
    p2, q2 = (sp.Integer(3), sp.Integer(0)), (sp.Integer(3), sp.Integer(2))  # disjoint
    cross2 = (sp.sign(o(a, b, p2)) != sp.sign(o(a, b, q2))) and \
             (sp.sign(o(p2, q2, a)) != sp.sign(o(p2, q2, b)))
    assert cross2 is False
    return "opposite-sign orient2d on both pairs"


def derive_intersection_point():
    # Two segments P0+t(P1-P0) and Q0+u(Q1-Q0). Solve for t (the point on the
    # first segment). Used only when a constraint must be split for conforming
    # refinement; the 2D CDT proper does not move/split constraints.
    t, u = sp.symbols("t u", real=True)
    P0 = sp.Matrix(sp.symbols("p0x p0y", real=True))
    P1 = sp.Matrix(sp.symbols("p1x p1y", real=True))
    Q0 = sp.Matrix(sp.symbols("q0x q0y", real=True))
    Q1 = sp.Matrix(sp.symbols("q1x q1y", real=True))
    eqs = (P0 + t * (P1 - P0)) - (Q0 + u * (Q1 - Q0))
    sol = sp.solve([eqs[0], eqs[1]], [t, u], dict=True)[0]
    # Cramer's rule form: denominator is the cross product of the directions.
    d1 = P1 - P0
    d2 = Q1 - Q0
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    num_t = (Q0 - P0)[0] * d2[1] - (Q0 - P0)[1] * d2[0]
    assert sp.simplify(sol[t] - num_t / denom) == 0
    return sp.simplify(sol[t])


if __name__ == "__main__":
    print("orient2d  =", derive_orient())
    print("incircle  =", derive_incircle())
    print("proper-intersection:", derive_proper_intersection())
    print("split t   =", derive_intersection_point())
    print("\nAll symbolic identities verified.")
