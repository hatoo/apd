use apd::d2::advect;
use cgmath::vec2;
use ndarray::Array;
use noise::{NoiseFn, Perlin};

fn main() {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut soot = Array::zeros((N, N));

    for i in N / 2 - 100..=N / 2 + 100 {
        for j in N / 2 - 100..=N / 2 + 100 {
            soot[[i, j]] = 0.5;
        }
    }

    let perlin = Perlin::new();
    let freq = 4.0;
    let uv = Array::from_shape_fn((N, N), |(i, j)| {
        let u = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]);
        let v = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]);

        vec2(u, v) * 1.0
    });

    let dt = 1.0 / 24.0;
    let dx = 1.0 / N as f64;

    for f in 1..=N_FRAME {
        image_util::save_monochrome("diffuse", f, &soot).unwrap();

        soot = advect(&soot, &uv, dt / dx);

        eprint!("\r {} / {}", f, N_FRAME);
    }
}
