use ndarray::Array2;

pub fn interpolate_linear(q: &Array2<f64>, (i, j): (f64, f64)) -> f64 {
    let (w, h) = q.dim();

    assert!(w >= 1);
    assert!(h >= 1);

    let x = i.max(0.0).min((w - 1) as f64);
    let y = j.max(0.0).min((h - 1) as f64);

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

pub fn interpolate_bicubic(q: &Array2<f64>, (i, j): (f64, f64)) -> f64 {
    let (w, h) = q.dim();

    assert!(w >= 4);
    assert!(h >= 4);

    let x = i.max(1.0).min((w - 3) as f64);
    let y = j.max(1.0).min((h - 3) as f64);

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

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_interpolate_linear() {
        let q = array![[0.0, 1.0, 2.0]];

        assert_abs_diff_eq!(interpolate_linear(&q, (0.0, 0.0)), 0.0);
        assert_abs_diff_eq!(interpolate_linear(&q, (0.0, 0.5)), 0.5);
        assert_abs_diff_eq!(interpolate_linear(&q, (0.0, 1.0)), 1.0);
    }

    #[test]
    fn test_interpolate_bicubic() {
        let q = array![
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ];

        assert_abs_diff_eq!(interpolate_bicubic(&q, (1.0, 1.0)), 0.0);
        assert_abs_diff_eq!(interpolate_bicubic(&q, (1.0, 1.5)), 0.5);
        assert_abs_diff_eq!(interpolate_bicubic(&q, (1.0, 2.0)), 1.0);

        for i in 0..4 {
            for j in 0..4 {
                // Ensure no index out of bounds
                interpolate_bicubic(&q, (i as f64, j as f64));
            }
        }
    }
}
