//! 2D-Pfad-Benchmark: ein RFIC-artiges Layout (Spiral-Windung aus abuttenden
//! Segmenten + Ground-Ring auf zweiter Lage) durch `mesh_layers` unter Budget.
//! Misst die Wand-Zeit der komplette 2D-Pipeline (Union, Resample, CVT,
//! Ruppert, Smoothing) über wachsende Budgets, damit Skalierungsknicke der
//! Kern-Stufen (Constraint-Forcing, smooth_mesh) sichtbar werden.
//!
//!     cargo run --release -p rapidmesh-topo --features mesher --example bench2d
//!
//! Ausgabe: budget, tris, points, min_angle, seconds. Keine Asserts — ein
//! Mess-, kein Testwerkzeug (die Qualität sichern die Unit-Tests).

use rapidmesh_topo::{mesh_layers, Mesh2DOptions, Region2D};

/// Quadratische Spirale aus `turns` Windungen abuttender Rechtecke der Breite
/// `w` mit Pitch `p` (Segment-Rechtecke, wie ein Inductor-Generator sie legt).
fn spiral(turns: usize, w: f64, p: f64, tag: i64) -> Vec<Region2D> {
    let mut out = Vec::new();
    let mut half = p * turns as f64; // halbe Kantenlänge, schrumpft je Windung
    let mut rect = |x0: f64, y0: f64, x1: f64, y1: f64| {
        out.push(Region2D::new(
            vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            tag,
        ));
    };
    for _ in 0..turns {
        // vier Schenkel einer Windung (im Uhrzeigersinn, überlappend an den Ecken)
        rect(-half, half - w, half, half); // oben
        rect(half - w, -half, half, half); // rechts
        rect(-half, -half, half, -half + w); // unten
        rect(-half, -half + p, -half + w, half); // links (eine Stufe kürzer -> Spirale)
        half -= p;
    }
    out
}

fn main() {
    let turns = 6;
    let (w, p) = (0.6, 2.0);
    let coil = spiral(turns, w, p, 1);
    // Ground-Ring auf einer zweiten Lage, überlappt die Spule in der Ebene.
    let g = p * (turns + 2) as f64;
    let ring: Vec<Region2D> = vec![
        Region2D::new(vec![[-g, g - 1.5], [g, g - 1.5], [g, g], [-g, g]], 2),
        Region2D::new(vec![[g - 1.5, -g], [g, -g], [g, g], [g - 1.5, g]], 2),
        Region2D::new(vec![[-g, -g], [g, -g], [g, -g + 1.5], [-g, -g + 1.5]], 2),
        Region2D::new(vec![[-g, -g], [-g + 1.5, -g], [-g + 1.5, g], [-g, g]], 2),
    ];
    let groups = [coil, ring];

    println!("{:>8} {:>9} {:>9} {:>10} {:>9}", "budget", "tris", "points", "min_angle", "secs");
    for &budget in &[5_000usize, 20_000, 60_000] {
        let opts = Mesh2DOptions { target_count: budget, ..Default::default() };
        let t0 = std::time::Instant::now();
        let ms = mesh_layers(&groups, |_p| 0.5, &opts);
        let dt = t0.elapsed().as_secs_f64();
        let tris: usize = ms.iter().map(|m| m.tris.len()).sum();
        let pts: usize = ms.iter().map(|m| m.points.len()).sum();
        let min_angle = ms
            .iter()
            .flat_map(|m| m.geom.min_angle.iter().copied())
            .fold(f64::MAX, f64::min);
        println!("{budget:>8} {tris:>9} {pts:>9} {min_angle:>10.2} {dt:>9.2}");
    }
}
