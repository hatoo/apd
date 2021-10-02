use image::{Rgb, RgbImage};
use ndarray::Array2;

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
