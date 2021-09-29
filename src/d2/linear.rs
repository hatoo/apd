use ndarray::{Array, Array2, Zip};

pub fn apply_a2(out: &mut Array2<f64>, ans: &Array2<f64>, a: &Array2<f64>, c: &Array2<f64>) {
    let (w, h) = ans.dim();

    out.indexed_iter_mut().for_each(|((i, j), e)| {
        *e = c[[i, j]] * ans[[i, j]];

        if i > 0 {
            *e += a[[i - 1, j]] * ans[[i - 1, j]];
        }

        if i + 1 < w {
            *e += a[[i + 1, j]] * ans[[i + 1, j]];
        }

        if j > 0 {
            *e += a[[i, j - 1]] * ans[[i, j - 1]];
        }

        if j + 1 < h {
            *e += a[[i, j + 1]] * ans[[i, j + 1]];
        }
    })
}

fn pre_compute(a: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    let tuning = 0.97;
    let sigma = 0.25;

    let (w, h) = a.dim();
    let mut precon = Array::from_elem(a.dim(), 0.0);

    for i in 0..w {
        for j in 0..h {
            let e = c[[i, j]]
                - if i > 0 {
                    a[[i - 1, j]] * precon[[i - 1, j]]
                } else {
                    0.0
                }
                .powi(2)
                - if j > 0 {
                    a[[i, j - 1]] * precon[[i, j - 1]]
                } else {
                    0.0
                }
                .powi(2)
                - tuning
                    * (if i > 0 {
                        a[[i - 1, j]] * a[[i - 1, j]] * precon[[i - 1, j]].powi(2)
                    } else {
                        0.0
                    } + if j > 0 {
                        a[[i, j - 1]] * a[[i, j - 1]] * precon[[i, j - 1]].powi(2)
                    } else {
                        0.0
                    });

            let e = if e < sigma * c[[i, j]] { c[[i, j]] } else { e };
            precon[[i, j]] = 1.0 / e.sqrt();
        }
    }
    precon
}

fn apply_precon2(z: &mut Array2<f64>, r: &Array2<f64>, a: &Array2<f64>, precon: &Array2<f64>) {
    let (w, h) = z.dim();

    let mut q = Array::zeros(z.dim());

    for i in 0..w {
        for j in 0..h {
            let t = r[[i, j]]
                - if i > 0 {
                    a[[i - 1, j]] * precon[[i - 1, j]] * q[[i - 1, j]]
                } else {
                    0.0
                }
                - if j > 0 {
                    a[[i, j - 1]] * precon[[i, j - 1]] * q[[i, j - 1]]
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
                    a[[i, j]] * precon[[i, j]] * z[[i + 1, j]]
                } else {
                    0.0
                }
                - if j + 1 < h {
                    a[[i, j]] * precon[[i, j]] * z[[i, j + 1]]
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

pub fn lin_solve_pcg2(
    p: &mut Array2<f64>,
    d: &Array2<f64>,
    a: &Array2<f64>,
    c: &Array2<f64>,
) -> (usize, f64) {
    assert_eq!(p.dim(), d.dim());

    if d.iter().all(|&d| d < 1e-6) {
        return (0, 0.0);
    }

    let tol = 1e-6 * d.iter().fold(0.0f64, |a, &b| a.max(b));

    let precon = pre_compute(a, c);
    let mut r = d.clone();
    let mut z = Array::zeros(p.dim());
    apply_precon2(&mut z, &r, a, &precon);
    let mut s = z.clone();

    let mut sigma = dot_product(&z, &r);
    let mut err = 0.0;

    for i in 0..200 {
        apply_a2(&mut z, &s, a, c);
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

        apply_precon2(&mut z, &r, a, &precon);

        let sigma_new = dot_product(&z, &r);
        let beta = sigma_new / sigma;

        Zip::from(&mut s).and(&z).for_each(|a, &b| {
            *a = b + beta * *a;
        });

        sigma = sigma_new;
    }

    (200, err)
}
