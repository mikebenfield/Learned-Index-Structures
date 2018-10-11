//! Generating synthetic data

use rand::distributions::{Distribution, LogNormal};
use rand::{FromEntropy, XorShiftRng};

pub fn gen_numbers<F>(mut f: F, count: usize) -> Box<[f32]>
where
    F: FnMut() -> f32,
{
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(f());
    }
    result.sort_by(|a, b| a.partial_cmp(b).unwrap());
    result.into_boxed_slice()
}

/// generate `count` samples drawn from a Log-Normal distribution with mean 2.0
/// and std deviation 3.0, sorted.
pub fn gen_lognormal(count: usize) -> Box<[f32]> {
    let mut rng = XorShiftRng::from_entropy();
    let lognormal = LogNormal::new(0.0, 0.25);
    gen_numbers(|| lognormal.sample(&mut rng) as f32, count)
}
