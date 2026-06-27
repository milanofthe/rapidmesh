//! The canonical orderings and sign conventions, in one place. These are the
//! payload of the crate: they let a solver consume incidence without redoing
//! orientation work. Changing any of them is a breaking change to consumers.

/// The "no neighbour" sentinel (a boundary face/edge has only one incident
/// cell). Matches the `usize::MAX` convention rapidfem already uses.
pub const NONE: u32 = u32::MAX;

/// Local edges of a triangle, as local-vertex index pairs.
pub const TRI_EDGE_LOCAL: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Local edges of a tetrahedron, as local-vertex index pairs.
pub const TET_EDGE_LOCAL: [[usize; 2]; 6] =
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Local faces of a tetrahedron, as local-vertex index triples. Face `i`
/// excludes local vertex `i`, ordered so the triangle normal points **outward**
/// of a positively oriented tet (verified by [`tests::tet_faces_point_outward`]).
pub const TET_FACE_LOCAL: [[usize; 3]; 4] =
    [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// Canonical (min, max) form of an edge plus the sign of the supplied direction:
/// `+1` if `(a, b)` already runs min→max, `-1` if it is reversed.
#[inline]
pub fn canonical_edge(a: u32, b: u32) -> ([u32; 2], i8) {
    if a <= b {
        ([a, b], 1)
    } else {
        ([b, a], -1)
    }
}

/// Sort a triangle's three vertex ids ascending and return the parity of the
/// permutation that did it: `+1` for an even number of swaps, `-1` for odd.
/// Used to relate a face's local (oriented) order to its canonical (sorted) id.
#[inline]
pub fn sort3_sign(t: [u32; 3]) -> ([u32; 3], i8) {
    let mut a = t;
    let mut s: i8 = 1;
    if a[0] > a[1] {
        a.swap(0, 1);
        s = -s;
    }
    if a[1] > a[2] {
        a.swap(1, 2);
        s = -s;
    }
    if a[0] > a[1] {
        a.swap(0, 1);
        s = -s;
    }
    (a, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_edge_sign() {
        assert_eq!(canonical_edge(2, 5), ([2, 5], 1));
        assert_eq!(canonical_edge(5, 2), ([2, 5], -1));
    }

    #[test]
    fn sort3_parity() {
        assert_eq!(sort3_sign([1, 2, 3]), ([1, 2, 3], 1)); // identity: even
        assert_eq!(sort3_sign([2, 1, 3]), ([1, 2, 3], -1)); // one swap: odd
        assert_eq!(sort3_sign([2, 3, 1]), ([1, 2, 3], 1)); // two swaps: even
        assert_eq!(sort3_sign([1, 3, 2]), ([1, 2, 3], -1)); // one swap: odd
    }

    #[test]
    fn tet_faces_point_outward() {
        // Reference positively oriented tet (orient3d > 0).
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let centroid = [0.25, 0.25, 0.25];
        for (i, f) in TET_FACE_LOCAL.iter().enumerate() {
            let (a, b, c) = (v[f[0]], v[f[1]], v[f[2]]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            // Vector from the tet centroid to the face centroid must agree with
            // the normal -> the normal points away from the body (outward).
            let fc = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
            let out = [fc[0] - centroid[0], fc[1] - centroid[1], fc[2] - centroid[2]];
            let dot = n[0] * out[0] + n[1] * out[1] + n[2] * out[2];
            assert!(dot > 0.0, "face {i} normal is not outward (dot={dot})");
        }
    }
}
