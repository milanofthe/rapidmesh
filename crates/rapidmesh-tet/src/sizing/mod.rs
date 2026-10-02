//! The size field of a mesh ([`tree::DomainTree`], built with every
//! override by [`build_sizing_domain`]) and the element budgets that scale it
//! ([`budget`], [`budgeted`]).

pub(crate) mod tree;

use crate::mesh::TetMesh;

use crate::params::MeshParams;
use crate::sizing::tree::DomainTree;
use rapidmesh_geom::vec3::dist;

/// The mesh of `model` by `mesher`, under a tet budget where one is given:
/// the sizes scale over a few meshes until the count lands near it.
pub fn budgeted<E>(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    target_elements: Option<usize>,
    mesher: &dyn Fn(&MeshParams) -> Result<TetMesh, E>,
) -> Result<(TetMesh, MeshParams), E> {
    // the thickness bounds join the per-region sizes
    let params = &params.with_thickness_caps(&model.plc);
    let target = target_elements.unwrap_or(0);
    budget(params, target, Budget::Near(3.0), mesher, |m| m.tets.len())
}

/// How a mesh meets an element budget.
#[derive(Clone, Copy)]
pub(crate) enum Budget {
    /// The count lands near the budget; the count grows with the size to
    /// this power (3 for tets).
    Near(f64),
    /// The count is at most a little over the budget (a cap); the power as
    /// for `Near` (2 for triangles).
    Cap(f64),
}

/// Meshes at most this often to meet a budget.
const BUDGET_ROUNDS: usize = 6;
/// A count this fraction off the budget meets it.
const BUDGET_SLACK: f64 = 0.06;

/// The mesh by `mesh` under a budget of `target` elements (`count` of
/// them; 0 none): the sizes scale by one global factor over a few meshes
/// until the count meets it. Returns the mesh and the parameters it was
/// made with.
pub(crate) fn budget<M, E>(
    params: &MeshParams,
    target: usize,
    how: Budget,
    mesh: impl Fn(&MeshParams) -> Result<M, E>,
    count: impl Fn(&M) -> usize,
) -> Result<(M, MeshParams), E> {
    let mut p = params.clone();
    let mut m = mesh(&p)?;
    if target == 0 {
        return Ok((m, p));
    }
    let mut s = 1.0_f64;
    for _ in 1..BUDGET_ROUNDS {
        let n = count(&m).max(1) as f64;
        let ratio = n / target as f64;
        let (met, power) = match how {
            Budget::Near(d) => ((ratio - 1.0).abs() < BUDGET_SLACK, d),
            Budget::Cap(d) => (ratio <= 1.0 + BUDGET_SLACK, d),
        };
        if met {
            break;
        }
        s *= ratio.powf(1.0 / power);
        p = params.scaled(s);
        m = mesh(&p)?;
    }
    Ok((m, p))
}

/// Builds the domain sizing octree with the per-entity overrides applied: per-face
/// `surf_maxh` -> per-facet volume target (`facet_surf`), and per-edge `edge_maxh`
/// -> point sources sampled along the brep edge chain (so the field stays fine
/// along a refined edge). Shared by the volume meshes and the surface-only
/// export ([`crate::mesher::surface_mesh`]), so BOTH honor the same sizing knobs
/// (per-entity AND global caps, which `DomainTree::build` composes).
pub(crate) fn build_sizing_domain(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
) -> DomainTree {
    let (plc, brep) = (&model.plc, &model.brep);
    // Per-face `surf_maxh` -> per-facet volume target, per-face `surf_tol`
    // -> per-facet chord tolerance.
    let mut facet_surf = vec![f64::INFINITY; plc.triangles.len()];
    let mut facet_tol = vec![params.tol_surf; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        let h = params.surf_maxh_for(fid);
        let tol = params.surf_tol_for(fid);
        for &ti in &f.facets {
            facet_surf[ti as usize] = facet_surf[ti as usize].min(h);
            facet_tol[ti as usize] = facet_tol[ti as usize].min(tol);
        }
    }
    if params.edge_maxh.is_empty() {
        return DomainTree::build(plc, model.index(), params, &facet_surf, &facet_tol);
    }
    // Per-edge `edge_maxh` -> point sources along the brep edge chain. Only clones
    // the params when an edge override is actually present.
    let mut pa = params.clone();
    for (eid, e) in brep.edges.iter().enumerate() {
        let Some(&(_, h)) = params.edge_maxh.iter().find(|&&(i, _)| i as usize == eid) else {
            continue;
        };
        for w in e.chain.windows(2) {
            let n = ((dist(w[0], w[1]) / h).ceil() as usize).max(1);
            for k in 0..n {
                let t = k as f64 / n as f64;
                pa.size_points.push((
                    std::array::from_fn(|c| w[0][c] + t * (w[1][c] - w[0][c])),
                    h,
                ));
            }
        }
        if let Some(&last) = e.chain.last() {
            pa.size_points.push((last, h));
        }
    }
    DomainTree::build(plc, model.index(), &pa, &facet_surf, &facet_tol)
}
