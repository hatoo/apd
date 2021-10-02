use std::f64::consts::PI;

use cgmath::{vec2, Vector2};
use ndarray::{Array, Array2, ArrayViewMut2, Zip};

mod linear;

#[derive(Clone)]
pub struct MacGrid {
    u: Array2<f64>,
    v: Array2<f64>,
}

#[allow(clippy::many_single_char_names)]
fn interpolate_linear(q: &Array2<Vector2<f64>>, ij: Vector2<f64>) -> Vector2<f64> {
    let x = ij.x;
    let y = ij.y;

    let i0 = x as usize;
    let i1 = i0 + 1;

    let j0 = y as usize;
    let j1 = j0 + 1;

    let s1 = x.fract();
    let s0 = 1.0 - s1;

    let t1 = y.fract();
    let t0 = 1.0 - t1;

    let get = |i, j| q.get([i, j]).copied().unwrap_or(vec2(0.0, 0.0));

    (get(i0, j0) * t0 + get(i0, j1) * t1) * s0 + (get(i1, j0) * t0 + get(i1, j1) * t1) * s1
}

#[allow(clippy::many_single_char_names)]
fn interpolate_bicubic(q: &Array2<f64>, ij: Vector2<f64>, ambient_value: f64) -> f64 {
    let (w, h) = q.dim();

    let x = ij.x;
    let y = ij.y;

    let i = x as isize;
    let j = y as isize;

    let s = x.fract();
    let t = y.fract();

    let s_m1 = -1.0 / 3.0 * s + 0.5 * s * s - 1.0 / 6.0 * s * s * s;
    let s_0 = 1.0 - s * s + 0.5 * (s * s * s - s);
    let s_p1 = s + 0.5 * (s * s - s * s * s);
    let s_p2 = 1.0 / 6.0 * (s * s * s - s);

    let t_m1 = -1.0 / 3.0 * t + 0.5 * t * t - 1.0 / 6.0 * t * t * t;
    let t_0 = 1.0 - t * t + 0.5 * (t * t * t - t);
    let t_p1 = t + 0.5 * (t * t - t * t * t);
    let t_p2 = 1.0 / 6.0 * (t * t * t - t);

    let get = |i, j| {
        if (0..w as isize).contains(&i) && (0..h as isize).contains(&j) {
            q[[i as usize, j as usize]]
        } else {
            ambient_value
        }
    };

    let q_m1 = get(i - 1, j - 1) * s_m1
        + get(i, j - 1) * s_0
        + get(i + 1, j - 1) * s_p1
        + get(i + 2, j - 1) * s_p2;

    let q_0 = get(i - 1, j) * s_m1 + get(i, j) * s_0 + get(i + 1, j) * s_p1 + get(i + 2, j) * s_p2;

    let q_p1 = get(i - 1, j + 1) * s_m1
        + get(i, j + 1) * s_0
        + get(i + 1, j + 1) * s_p1
        + get(i + 2, j + 1) * s_p2;

    let q_p2 = get(i - 1, j + 2) * s_m1
        + get(i, j + 2) * s_0
        + get(i + 1, j + 2) * s_p1
        + get(i + 2, j + 2) * s_p2;

    q_m1 * t_m1 + q_0 * t_0 + q_p1 * t_p1 + q_p2 * t_p2
}

/// Advect `q` by `coeff` * `uv` vectors.
pub fn advect(
    q: &Array2<f64>,
    uv: &Array2<Vector2<f64>>,
    coeff: f64,
    ambient_value: f64,
) -> Array2<f64> {
    assert_eq!(q.dim(), uv.dim());

    Array::from_shape_fn(q.dim(), |(i, j)| {
        // Runge-Kutta(3)
        let x0 = vec2(i as f64, j as f64);
        let k1 = uv[[i, j]];
        let k2 = interpolate_linear(uv, x0 - 0.5 * coeff * k1);
        let k3 = interpolate_linear(uv, x0 - 0.75 * coeff * k2);

        let v = x0 - 2.0 / 9.0 * coeff * k1 - 3.0 / 9.0 * coeff * k2 - 4.0 / 9.0 * coeff * k3;

        interpolate_bicubic(q, v, ambient_value)
    })
}

pub fn diffuse(q: &Array2<f64>, sigma2: f64, dx: f64, ambient_value: f64) -> Array2<f64> {
    let cut_off = 0.1 * sigma2.sqrt();
    let left = 1.0 / (2.0 * PI * sigma2).sqrt();
    let coeff: Vec<f64> = (0..)
        .map(|i| left * (-(i as f64 * dx).powi(2) / (2.0 * sigma2)).exp())
        .take_while(|&f| f > cut_off)
        .take(q.dim().0.max(q.dim().1))
        .collect();

    let coeff_sum_inv = 1.0 / (coeff.iter().sum::<f64>() * 2.0 - coeff[0]);

    let x1 = Array::from_shape_fn(q.dim(), |(i, j)| {
        let mut sum = coeff[0] * q[[i, j]];
        for c in 1..coeff.len() {
            sum += coeff[c] * if i >= c { q[[i - c, j]] } else { ambient_value };

            sum += coeff[c]
                * if i + c < q.dim().0 {
                    q[[i + c, j]]
                } else {
                    ambient_value
                };
        }
        sum * coeff_sum_inv
    });

    Array::from_shape_fn(q.dim(), |(i, j)| {
        let mut sum = coeff[0] * x1[[i, j]];
        for c in 1..coeff.len() {
            sum += coeff[c]
                * if j >= c {
                    x1[[i, j - c]]
                } else {
                    ambient_value
                };

            sum += coeff[c]
                * if j + c < q.dim().1 {
                    x1[[i, j + c]]
                } else {
                    ambient_value
                };
        }
        sum * coeff_sum_inv
    })
}

impl MacGrid {
    pub fn new(u: Array2<f64>, v: Array2<f64>) -> Self {
        let u_dim = u.dim();
        let v_dim = v.dim();

        assert_eq!((u_dim.0 - 1, u_dim.1), (v_dim.0, v_dim.1 - 1));

        Self { u, v }
    }

    pub fn zeros((w, h): (usize, usize)) -> Self {
        Self {
            u: Array::zeros((w + 1, h)),
            v: Array::zeros((w, h + 1)),
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.v.dim().0, self.u.dim().1)
    }

    pub fn create_uv(&self) -> Array2<Vector2<f64>> {
        Array::from_shape_fn(self.dim(), |(i, j)| {
            let u = 0.5 * (self.u[[i, j]] + self.u[[i + 1, j]]);
            let v = 0.5 * (self.v[[i, j]] + self.v[[i, j + 1]]);

            vec2(u, v)
        })
    }

    pub fn add(&mut self, other: &Self, weight: f64) {
        assert_eq!(self.dim(), other.dim());

        Zip::from(&mut self.u).and(&other.u).for_each(|a, &b| {
            *a += weight * b;
        });

        Zip::from(&mut self.v).and(&other.v).for_each(|a, &b| {
            *a += weight * b;
        });
    }

    pub fn self_advect(&mut self, weight: f64) {
        let u_uv = Array::from_shape_fn(self.u.dim(), |(i, j)| {
            let u = self.u[[i, j]];
            let v = if i == 0 {
                0.5 * (self.v[[i, j]] + self.v[[i, j + 1]])
            } else if i == self.u.dim().0 - 1 {
                0.5 * (self.v[[i - 1, j]] + self.v[[i - 1, j + 1]])
            } else {
                0.25 * (self.v[[i, j]]
                    + self.v[[i, j + 1]]
                    + self.v[[i - 1, j]]
                    + self.v[[i - 1, j + 1]])
            };
            vec2(u, v)
        });

        let v_uv = Array::from_shape_fn(self.v.dim(), |(i, j)| {
            let v = self.v[[i, j]];
            let u = if j == 0 {
                0.5 * (self.u[[i, j]] + self.u[[i + 1, j]])
            } else if j == self.v.dim().1 - 1 {
                0.5 * (self.u[[i, j - 1]] + self.u[[i + 1, j - 1]])
            } else {
                0.25 * (self.u[[i, j]]
                    + self.u[[i, j - 1]]
                    + self.u[[i + 1, j]]
                    + self.u[[i + 1, j - 1]])
            };
            vec2(u, v)
        });

        self.u = advect(&self.u, &u_uv, weight, 0.0);
        self.v = advect(&self.v, &v_uv, weight, 0.0);
    }

    pub fn project(&mut self, dt: f64, dx: f64, divergence: &Array2<f64>) {
        let divergence = Array::from_shape_fn(self.dim(), |(i, j)| {
            -1.0 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]]) / dx
                + divergence[[i, j]] / dx
        });

        // TODO: Support variable density.
        // Currently the density is restricted to constant since the linear system solver
        // does not converge unless the matrix is symmetry.
        let density = Array::from_elem(self.dim(), 1.0);
        let mut pressure = Array::zeros(divergence.dim());

        let scale = dt / (dx * dx);

        let diag = 4.0 * scale / &density;
        let others = -1.0 * scale / &density;

        linear::lin_solve_pcg(&mut pressure, &divergence, &diag, &others);

        let l = dt / dx;

        let (w, h) = self.dim();
        for i in 1..w {
            for j in 0..h {
                self.u[[i, j]] -= l * (pressure[[i, j]] - pressure[[i - 1, j]])
                    / (0.5 * (density[[i - 1, j]] + density[[i, j]]));
            }
        }

        for i in 0..w {
            for j in 1..h {
                self.v[[i, j]] -= l * (pressure[[i, j]] - pressure[[i, j - 1]])
                    / (0.5 * (density[[i, j - 1]] + density[[i, j]]));
            }
        }
    }

    pub fn diffuse(&mut self, sigma2: f64, dx: f64) {
        self.u = diffuse(&self.u, sigma2, dx, 0.0);
        self.v = diffuse(&self.v, sigma2, dx, 0.0);
    }

    pub fn u(&self) -> &Array2<f64> {
        &self.u
    }

    pub fn v(&self) -> &Array2<f64> {
        &self.v
    }

    pub fn u_mut(&mut self) -> ArrayViewMut2<f64> {
        self.u.view_mut()
    }

    pub fn v_mut(&mut self) -> ArrayViewMut2<f64> {
        self.v.view_mut()
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn close_v2(v1: Vector2<f64>, v2: Vector2<f64>) {
        assert_abs_diff_eq!(v1.x, v2.x);
        assert_abs_diff_eq!(v1.y, v2.y);
    }

    #[test]
    fn test_interpolate_linear() {
        let q = array![[vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(2.0, 2.0)]];

        close_v2(interpolate_linear(&q, vec2(0.0, 0.0)), vec2(0.0, 0.0));
        close_v2(interpolate_linear(&q, vec2(0.0, 0.5)), vec2(0.5, 0.5));
        close_v2(interpolate_linear(&q, vec2(0.0, 1.0)), vec2(1.0, 1.0));
    }

    #[test]
    fn test_interpolate_bicubic() {
        let q = array![
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ];

        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 1.0), 0.0), 0.0);
        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 1.5), 0.0), 0.5);
        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 2.0), 0.0), 1.0);

        for i in 0..4 {
            for j in 0..4 {
                // Ensure no index out of bounds
                interpolate_bicubic(&q, vec2(i as f64, j as f64), 0.0);
            }
        }
    }
}
