pub fn mse(errors: &[f64]) -> f64 {
    errors.iter().map(|&x| x * x).sum::<f64>() / errors.len() as f64
}