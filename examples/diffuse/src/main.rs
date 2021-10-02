use apd::d2::diffuse;
use ndarray::Array;

fn main() {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut soot = Array::zeros((N, N));

    let dt = 1.0 / 24.0;
    let diff = 0.001;
    let sigma2 = dt * diff;

    soot[[N / 2, N / 2]] = 1000.0;

    for f in 1..=N_FRAME {
        image_util::save_monochrome("diffuse", f, &soot).unwrap();
        soot = diffuse(&soot, sigma2, 1.0 / N as f64, 0.0);

        eprint!("\r {} / {}", f, N_FRAME);
    }
}
