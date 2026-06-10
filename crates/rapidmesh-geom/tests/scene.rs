//! Scene assembly: region resolution, per-region closure, sheet embedding
//! and coincident-facet merging on an EM-style air box + dielectric + PEC
//! scene.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_geom::{sheet_rect, solid_box, FaceTag, RegionTag, Scene, TaggedPlc};
use rapidmesh_testutil::rat;

/// Exact 6x volume of a region from its interface facets (normals point into
/// `front`, so facets with `back == r` bound r positively).
fn region_volume6(plc: &TaggedPlc, r: RegionTag) -> BigRational {
    let mut acc = BigRational::zero();
    for (t, tags) in plc.triangles.iter().zip(&plc.region_tags) {
        // Facets with the same region on both sides (embedded sheets) do not
        // bound the region.
        let sign = if tags[0] == tags[1] {
            continue;
        } else if tags[1] == r {
            1
        } else if tags[0] == r {
            -1
        } else {
            continue;
        };
        let (a, b, c) = (
            plc.vertices[t[0] as usize].map(rat),
            plc.vertices[t[1] as usize].map(rat),
            plc.vertices[t[2] as usize].map(rat),
        );
        let det = &a[0] * (&b[1] * &c[2] - &b[2] * &c[1])
            - &a[1] * (&b[0] * &c[2] - &b[2] * &c[0])
            + &a[2] * (&b[0] * &c[1] - &b[1] * &c[0]);
        acc += if sign > 0 { det } else { -det };
    }
    acc
}

/// Closure check: the boundary facets of a region, oriented out of it, form
/// a closed surface.
fn assert_region_closed(plc: &TaggedPlc, r: RegionTag) {
    let mut directed: std::collections::HashMap<(u32, u32), i64> =
        std::collections::HashMap::new();
    for (t, tags) in plc.triangles.iter().zip(&plc.region_tags) {
        // Outward orientation for r: keep as-is if back == r, flip if
        // front == r. Same-region facets (embedded sheets) are interior.
        let tri: [u32; 3] = if tags[0] == tags[1] {
            continue;
        } else if tags[1] == r {
            *t
        } else if tags[0] == r {
            [t[0], t[2], t[1]]
        } else {
            continue;
        };
        for e in 0..3 {
            *directed.entry((tri[e], tri[(e + 1) % 3])).or_default() += 1;
        }
    }
    assert!(!directed.is_empty(), "region {r:?} has no boundary facets");
    for (&(u, v), &n) in &directed {
        assert_eq!(n, 1, "directed edge ({u},{v}) used {n} times for {r:?}");
        assert!(
            directed.contains_key(&(v, u)),
            "edge ({u},{v}) unmatched: region {r:?} boundary not closed"
        );
    }
}

#[test]
fn air_dielectric_pec_scene() {
    let pec = FaceTag(7);
    let mut scene = Scene::new();
    let air = scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    // PEC patch lying exactly on the dielectric top face (coincident with
    // the air/dielectric interface): must merge, not duplicate.
    scene.add_sheet(
        sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        pec,
    );
    // Floating PEC sheet embedded in air.
    scene.add_sheet(
        sheet_rect([0.5, 0.5, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        pec,
    );
    let plc = scene.assemble();

    // Exact region volumes: dielectric 2*2*1 = 4, air 64 - 4 = 60.
    assert_eq!(region_volume6(&plc, diel), rat(24.0));
    assert_eq!(region_volume6(&plc, air), rat(360.0));
    assert_region_closed(&plc, air);
    assert_region_closed(&plc, diel);

    // No duplicate facets (unordered vertex triples unique).
    let mut seen = std::collections::HashSet::new();
    for t in &plc.triangles {
        let mut k = *t;
        k.sort_unstable();
        assert!(seen.insert(k), "duplicate facet {t:?}");
    }

    // The coincident PEC patch: facets tagged PEC *and* on the air/diel
    // interface.
    let merged = plc
        .face_tags
        .iter()
        .zip(&plc.region_tags)
        .filter(|(tag, tags)| **tag == pec && tags.contains(&air) && tags.contains(&diel))
        .count();
    assert!(merged >= 2, "coincident PEC patch must be merged into the interface");

    // The floating sheet: PEC facets fully inside air ([air, air]).
    let floating = plc
        .face_tags
        .iter()
        .zip(&plc.region_tags)
        .filter(|(tag, tags)| **tag == pec && **tags == [air, air])
        .count();
    assert!(floating >= 2, "floating PEC sheet must be present in air");

    // Sanity: total interface area of the dielectric box is fully present —
    // count facets touching diel.
    assert!(
        plc.region_tags.iter().filter(|t| t.contains(&diel)).count() >= 12,
        "dielectric interface unexpectedly small"
    );
}

#[test]
fn overlapping_solids_resolve_by_priority() {
    let mut scene = Scene::new();
    let first = scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]));
    let second = scene.add_solid(solid_box([1.0, 0.0, 0.0], [3.0, 2.0, 2.0]));
    let plc = scene.assemble();
    // Later solid wins the overlap [1,2]: first keeps 1*2*2 = 4, second 8.
    assert_eq!(region_volume6(&plc, first), rat(24.0));
    assert_eq!(region_volume6(&plc, second), rat(48.0));
    assert_region_closed(&plc, first);
    assert_region_closed(&plc, second);
    // Background boundary is the union hull: volume 3*2*2 = 12.
    let total = region_volume6(&plc, first) + region_volume6(&plc, second);
    assert_eq!(total, rat(72.0));
    // The hidden wall of `first` inside `second` must be gone: no facet may
    // separate `second` from `second`.
    for tags in &plc.region_tags {
        assert_ne!(tags[0], tags[1], "interior facet survived region resolution");
    }
}
