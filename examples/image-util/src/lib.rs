use cgmath::{vec2, Vector2};
use image::{Rgb, RgbImage};
use ndarray::Array2;
use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform};

pub fn save_monochrome(prefix: &str, index: usize, x: &Array2<f64>) -> anyhow::Result<()> {
    let shape = x.dim();

    let mut img = RgbImage::new(shape.0 as u32, shape.1 as u32);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let l = (x[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            img.put_pixel(i as u32, j as u32, Rgb([l, l, l]));
        }
    }

    img.save(format!("out/{}_{:06}.png", prefix, index))?;

    Ok(())
}

pub fn save_uv(
    prefix: &str,
    index: usize,
    uv: &Array2<Vector2<f64>>,
    interval: usize,
    length: f64,
) -> anyhow::Result<()> {
    let (w, h) = uv.dim();

    let mut pixmap = Pixmap::new(w as u32, h as u32).unwrap();

    let paint = {
        let mut p = Paint::default();
        p.anti_alias = true;
        p.set_color(Color::WHITE);
        p
    };

    let stroke = {
        let mut s = Stroke::default();
        s.width = 1.0;
        s
    };

    let mut pb = PathBuilder::new();
    for i in 0..w / interval {
        for j in 0..h / interval {
            let i = i * interval;
            let j = j * interval;

            let x = i + interval / 2;
            let y = j + interval / 2;

            pb.move_to(x as f32, y as f32);

            let to = vec2(x as f64, y as f64) + length * uv[[i, j]];
            pb.line_to(to.x as f32, to.y as f32);
        }
    }

    let path = pb.finish().unwrap();
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    pixmap.save_png(format!("out/{}_{:06}.png", prefix, index))?;

    Ok(())
}
