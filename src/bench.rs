//! Benchmarking models

use std::time::{Duration, Instant};

use rand::distributions::Uniform;
use rand::{FromEntropy, Rng, XorShiftRng};

use model::Model;

pub fn duration_to_secs(dur: Duration) -> f64 {
    let secs = dur.as_secs() as f64;
    let frac = dur.subsec_millis() as f64 / 1000.0;
    secs + frac
}

/// Randomly sample `count` keys from `data`, call `eval_many` on `model`, and
/// return how long `eval_many` took.
pub fn bench<M>(model: &M, data: &[f32], count: usize) -> Duration
where
    M: Model<f32, u32>,
{
    let mut rng = XorShiftRng::from_entropy();

    let keys: Vec<f32> = {
        let dist = Uniform::new(0, count);
        let mut vec = Vec::with_capacity(count);
        for _ in 0..count {
            vec.push(data[rng.sample(dist)]);
        }
        vec
    };

    let mut indices = vec![None; count];

    let t1 = Instant::now();
    model.eval_many(&keys, &mut indices);
    let t2 = Instant::now();

    t2.duration_since(t1)
}
