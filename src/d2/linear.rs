use ndarray::{Array, Array2, Zip};

fn apply_a(out: &mut Array2<f64>, v: &Array2<f64>, diag: f64, others: f64) {
    let (w, h) = v.dim();

    out.indexed_iter_mut().for_each(|((i, j), e)| {
        *e = diag * v[[i, j]];

        if i > 0 {
            *e += others * v[[i - 1, j]];
        }

        if i + 1 < w {
            *e += others * v[[i + 1, j]];
        }

        if j > 0 {
            *e += others * v[[i, j - 1]];
        }

        if j + 1 < h {
            *e += others * v[[i, j + 1]];
        }
    })
}

fn pre_compute(diag: f64, others: f64, (w, h): (usize, usize)) -> Array2<f64> {
    let tuning = 0.97;
    let sigma = 0.25;

    let mut precon = Array::from_elem((w, h), 0.0);

    for i in 0..w {
        for j in 0..h {
            let e = diag
                - if i > 0 {
                    others * precon[[i - 1, j]]
                } else {
                    0.0
                }
                .powi(2)
                - if j > 0 {
                    others * precon[[i, j - 1]]
                } else {
                    0.0
                }
                .powi(2)
                - tuning
                    * (if i > 0 {
                        others * others * precon[[i - 1, j]].powi(2)
                    } else {
                        0.0
                    } + if j > 0 {
                        others * others * precon[[i, j - 1]].powi(2)
                    } else {
                        0.0
                    });

            let e = if e < sigma * diag { diag } else { e };
            precon[[i, j]] = 1.0 / e.sqrt();
        }
    }
    precon
}

#[allow(clippy::many_single_char_names)]
fn apply_precon(z: &mut Array2<f64>, r: &Array2<f64>, others: f64, precon: &Array2<f64>) {
    let (w, h) = z.dim();

    let mut q = Array::zeros(z.dim());

    for i in 0..w {
        for j in 0..h {
            let t = r[[i, j]]
                - if i > 0 {
                    others * precon[[i - 1, j]] * q[[i - 1, j]]
                } else {
                    0.0
                }
                - if j > 0 {
                    others * precon[[i, j - 1]] * q[[i, j - 1]]
                } else {
                    0.0
                };
            q[[i, j]] = t * precon[[i, j]];
        }
    }

    for i in (0..w).rev() {
        for j in (0..h).rev() {
            let t = q[[i, j]]
                - if i + 1 < w {
                    others * precon[[i, j]] * z[[i + 1, j]]
                } else {
                    0.0
                }
                - if j + 1 < h {
                    others * precon[[i, j]] * z[[i, j + 1]]
                } else {
                    0.0
                };

            z[[i, j]] = t * precon[[i, j]];
        }
    }
}

fn dot_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.indexed_iter()
        .map(|((i, j), e)| *e * b[[i, j]])
        .sum::<f64>()
}

/// Modified Incomplete Cholesky Conjugate Gradient, Level Zero
#[allow(clippy::many_single_char_names)]
pub fn lin_solve_pcg(p: &mut Array2<f64>, b: &Array2<f64>, diag: f64, others: f64) -> (usize, f64) {
    assert_eq!(p.dim(), b.dim());

    const MAX_ITER: usize = 200;

    if b.iter().all(|&d| d < 1e-6) {
        return (0, 0.0);
    }

    let tol = 1e-6 * b.iter().fold(0.0f64, |a, &b| a.max(b));

    let precon = pre_compute(diag, others, p.dim());
    let mut r = b.clone();
    let mut z = Array::zeros(p.dim());
    apply_precon(&mut z, &r, others, &precon);
    let mut s = z.clone();

    let mut sigma = dot_product(&z, &r);
    let mut err = 0.0;

    for i in 0..MAX_ITER {
        apply_a(&mut z, &s, diag, others);
        let alpha = sigma / dot_product(&z, &s);

        Zip::from(&mut *p).and(&s).for_each(|a, &b| {
            *a += alpha * b;
        });

        Zip::from(&mut r).and(&z).for_each(|a, &b| {
            *a -= alpha * b;
        });

        err = r.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if err < tol {
            return (i, err);
        }

        apply_precon(&mut z, &r, others, &precon);

        let sigma_new = dot_product(&z, &r);
        let beta = sigma_new / sigma;

        Zip::from(&mut s).and(&z).for_each(|a, &b| {
            *a = b + beta * *a;
        });

        sigma = sigma_new;
    }

    (MAX_ITER, err)
}
