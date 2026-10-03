//! Reads STEP files and tells how: the time, per body its facets and the
//! surfaces with the most of them.
//!
//! cargo run --release -p rapidmesh-step --example read -- part.step [more.step ...]

use rapidmesh_step::{read, Tolerance};

fn main() {
    if std::env::var_os("RAPIDMESH_LOG").is_none() {
        rapidmesh_exact::log::set_level(Some(rapidmesh_exact::log::Level::Debug));
    }
    for path in std::env::args().skip(1) {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                println!("{path}: {e}");
                continue;
            }
        };
        let t = std::time::Instant::now();
        let step = match read(&text, Tolerance::default()) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}: {e}");
                continue;
            }
        };
        println!(
            "{path}: {} bodies in {:.0} ms",
            step.bodies.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        for b in &step.bodies {
            let mut per: Vec<(usize, usize)> =
                (0..b.solid.surfaces.len()).map(|s| (0, s)).collect();
            for &s in &b.solid.face_surface {
                per[s as usize].0 += 1;
            }
            per.sort_unstable_by(|x, y| y.cmp(x));
            let top: Vec<String> = per
                .iter()
                .take(3)
                .map(|&(n, s)| {
                    format!("{n} on {:?}", b.solid.surfaces[s])
                        .chars()
                        .take(60)
                        .collect()
                })
                .collect();
            // Closed: each edge of the facets used once each way.
            let mut edges: std::collections::HashMap<[[u64; 3]; 2], i32> = Default::default();
            for t in &b.solid.tris {
                for k in 0..3 {
                    let (p, q) = (t.v[k].map(f64::to_bits), t.v[(k + 1) % 3].map(f64::to_bits));
                    *edges.entry([p.min(q), p.max(q)]).or_default() += if p < q { 1 } else { -1 };
                }
            }
            let open: Vec<_> = edges.iter().filter(|(_, &c)| c != 0).collect();
            println!(
                "  {:<24} {:>7} facets, {} open edges, most: {}",
                b.name,
                b.solid.tris.len(),
                open.len(),
                top.join("; ")
            );
            // Where the first of them are.
            for (e, _) in open.iter().take(6) {
                let [p, q] = e.map(|v| v.map(f64::from_bits));
                let on: Vec<String> = (0..b.solid.tris.len())
                    .filter(|&t| {
                        let v = b.solid.tris[t].v.map(|x| x.map(f64::to_bits));
                        v.contains(&e[0]) && v.contains(&e[1])
                    })
                    .map(|t| {
                        format!("{:?}", b.solid.surfaces[b.solid.face_surface[t] as usize])
                            .chars()
                            .take(50)
                            .collect()
                    })
                    .collect();
                println!("    open {p:.4?} to {q:.4?} on {}", on.join(" | "));
            }
        }
    }
}
