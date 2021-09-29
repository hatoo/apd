use cgmath::{vec2, Vector2};
use ndarray::{Array, Array2};

fn interpolate_linear(q: &Array2<Vector2<f64>>, ij: Vector2<f64>) -> Vector2<f64> {
    let (w, h) = q.dim();

    assert!(w >= 1);
    assert!(h >= 1);

    let x = ij.x.max(0.0).min((w - 1) as f64);
    let y = ij.y.max(0.0).min((h - 1) as f64);

    let i0 = x as usize;
    let i1 = (i0 + 1).min(w - 1);

    let j0 = y as usize;
    let j1 = (j0 + 1).min(h - 1);

    let s1 = x - i0 as f64;
    let s0 = 1.0 - s1;

    let t1 = y - j0 as f64;
    let t0 = 1.0 - t1;

    (q[[i0, j0]] * t0 + q[[i0, j1]] * t1) * s0 + (q[[i1, j0]] * t0 + q[[i1, j1]] * t1) * s1
}

fn interpolate_bicubic(q: &Array2<f64>, ij: Vector2<f64>) -> f64 {
    let (w, h) = q.dim();

    assert!(w >= 4);
    assert!(h >= 4);

    let x = ij.x.max(1.0).min((w - 3) as f64);
    let y = ij.y.max(1.0).min((h - 3) as f64);

    let i = x as usize;
    let j = y as usize;

    let s = x - i as f64;
    let t = y - j as f64;

    let s_m1 = -1.0 / 3.0 * s + 0.5 * s * s - 1.0 / 6.0 * s * s * s;
    let s_0 = 1.0 - s * s + 0.5 * (s * s * s - s);
    let s_p1 = s + 0.5 * (s * s - s * s * s);
    let s_p2 = 1.0 / 6.0 * (s * s * s - s);

    let t_m1 = -1.0 / 3.0 * t + 0.5 * t * t - 1.0 / 6.0 * t * t * t;
    let t_0 = 1.0 - t * t + 0.5 * (t * t * t - t);
    let t_p1 = t + 0.5 * (t * t - t * t * t);
    let t_p2 = 1.0 / 6.0 * (t * t * t - t);

    let q_m1 = q[[i - 1, j - 1]] * s_m1
        + q[[i, j - 1]] * s_0
        + q[[i + 1, j - 1]] * s_p1
        + q[[i + 2, j - 1]] * s_p2;

    let q_0 = q[[i - 1, j]] * s_m1 + q[[i, j]] * s_0 + q[[i + 1, j]] * s_p1 + q[[i + 2, j]] * s_p2;

    let q_p1 = q[[i - 1, j + 1]] * s_m1
        + q[[i, j + 1]] * s_0
        + q[[i + 1, j + 1]] * s_p1
        + q[[i + 2, j + 1]] * s_p2;

    let q_p2 = q[[i - 1, j + 2]] * s_m1
        + q[[i, j + 2]] * s_0
        + q[[i + 1, j + 2]] * s_p1
        + q[[i + 2, j + 2]] * s_p2;

    q_m1 * t_m1 + q_0 * t_0 + q_p1 * t_p1 + q_p2 * t_p2
}

/// Advect `q` by `coeff` * `uv` vectors.
pub fn advect(q: &Array2<f64>, uv: &Array2<Vector2<f64>>, coeff: f64) -> Array2<f64> {
    assert_eq!(q.dim(), uv.dim());

    Array::from_shape_fn(q.dim(), |(i, j)| {
        // Runge-Kutta(3)
        let x0 = vec2(i as f64, j as f64);
        let k1 = uv[[i, j]];
        let k2 = interpolate_linear(uv, x0 - 0.5 * coeff * k1);
        let k3 = interpolate_linear(uv, x0 - 0.75 * coeff * k2);

        let v = x0 - 2.0 / 9.0 * coeff * k1 - 3.0 / 9.0 * coeff * k2 - 4.0 / 9.0 * coeff * k3;

        interpolate_bicubic(q, v)
    })
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

        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 1.0)), 0.0);
        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 1.5)), 0.5);
        assert_abs_diff_eq!(interpolate_bicubic(&q, vec2(1.0, 2.0)), 1.0);

        for i in 0..4 {
            for j in 0..4 {
                // Ensure no index out of bounds
                interpolate_bicubic(&q, vec2(i as f64, j as f64));
            }
        }
    }
}
