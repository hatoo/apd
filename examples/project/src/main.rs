use apd::d2::{advect, MacGrid};
use ndarray::Array;
use noise::{NoiseFn, Perlin};

fn main() {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut soot = Array::from_elem((N, N), 0.5);

    let perlin = Perlin::new();
    let freq = 4.0;
    let u = Array::from_shape_fn((N + 1, N), |(i, j)| {
        let u = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]);
        u
    });
    let v = Array::from_shape_fn((N, N + 1), |(i, j)| {
        let v = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]);
        v
    });

    let dt = 1.0 / 24.0;
    let dx = 1.0 / N as f64;

    let divergence = Array::zeros((N, N));
    let mut mac_grid = MacGrid::new(u, v);

    for f in 1..=N_FRAME {
        image_util::save_monochrome("project", f, &soot).unwrap();

        mac_grid.self_advect(dt / dx);
        mac_grid.project(dt, dx, &divergence);
        let uv = mac_grid.create_uv();
        soot = advect(&soot, &uv, dt / dx, 0.0);

        eprint!("\r {} / {}", f, N_FRAME);
    }
}
