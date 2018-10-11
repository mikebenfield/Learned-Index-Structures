//! A fully connected neural network with ReLU activations.
//!
//! Little thought given to performance optimization at the moment. Lots of
//! unnecessary allocations, no explicit SIMD.

use std::cmp::min;

use model::Model;

pub struct Stage {
    /// Width of each layer (other than the last)
    pub layers: Box<[usize]>,

    /// How many models in this stage?
    pub models: usize,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Layer {
    // row major order
    pub(crate) data: Box<[f32]>,
    pub(crate) bias: Box<[f32]>,
}

#[derive(Debug, PartialEq)]
pub struct Vector {
    pub(crate) data: Box<[f32]>,
}

impl Layer {
    fn apply_relu(&self, v: &Vector) -> Vector {
        let mut result = self.apply(v);
        for i in 0..result.data.len() {
            if result.data[i] < 0.0 {
                result.data[i] *= 0.3;
            }
        }
        result
    }

    fn apply(&self, v: &Vector) -> Vector {
        debug_assert!(self.data.len() % v.data.len() == 0);
        let out_dim = self.data.len() / v.data.len();
        debug_assert!(self.bias.len() == out_dim);
        let mut result = Vec::with_capacity(out_dim);
        for row in 0..out_dim {
            let mut val = 0.0;
            for column in 0..v.data.len() {
                val += v.data[column] * self.data[row * v.data.len() + column];
                val += self.bias[row];
            }
            result.push(val);
        }
        Vector {
            data: result.into_boxed_slice(),
        }
    }
}

pub struct Network {
    pub(crate) layers: Box<[Layer]>,
}

impl Network {
    fn apply0(&self, v: &Vector, layer: usize) -> Vector {
        if layer == self.layers.len() - 1 {
            self.layers[layer].apply_relu(v)
        } else {
            let result = self.layers[layer].apply_relu(v);
            self.apply0(&result, layer + 1)
        }
    }

    pub fn apply(&self, v: &Vector) -> Vector {
        self.apply0(v, 0)
    }
}

pub struct NeuralModel {
    network: Network,
    data: Box<[f32]>,
    min_offset: u32,
    max_offset: u32,
}

impl Model<f32, u32> for NeuralModel {
    fn eval(&self, key: f32) -> Option<u32> {
        let vec = Vector {
            data: vec![key].into_boxed_slice(),
        };
        let result_vec = self.network.apply(&vec);
        let result = result_vec.data[0] as u32;
        let start_index = if self.min_offset >= result {
            0
        } else {
            result - self.min_offset
        };
        let end_index = min(result + self.max_offset, self.data.len() as u32 - 1);
        for i in start_index..=end_index {
            if self.data[i as usize] == key {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(v: &Vector, w: &Vector) -> bool {
        let eps = 0.00001;
        if v.data.len() != w.data.len() {
            return false;
        }
        for i in 0..v.data.len() {
            if (v.data[i] - w.data[i]).abs() > eps {
                return false;
            }
        }
        true
    }

    #[test]
    fn apply_layer() {
        let vec = Vector {
            data: vec![1.0, 2.0, 3.0].into_boxed_slice(),
        };
        let layer = Layer {
            data: vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0].into_boxed_slice(),
        };
        let gold = Vector {
            data: vec![9.0, 20.0, 10.0].into_boxed_slice(),
        };
        assert!(approx_eq(&gold, &layer.apply(&vec)));
    }

    #[test]
    fn apply_network() {
        let vec = Vector {
            data: vec![1.0, 2.0].into_boxed_slice(),
        };
        let layer0 = Layer {
            data: vec![1.0, 2.0, 3.0, -4.0].into_boxed_slice(),
        };
        let layer1 = Layer {
            data: vec![4.0, 3.0, 2.0, 1.0].into_boxed_slice(),
        };
        let network = Network {
            layers: vec![layer0, layer1].into_boxed_slice(),
        };
        let golden = Vector {
            data: vec![20.0, 10.0].into_boxed_slice(),
        };
        let result = network.apply(&vec);
        assert!(approx_eq(&golden, &result));
    }

    #[test]
    fn model() {
        // test with made-up weights, but huge offsets, so the algorithm
        // degrades into a linear search
        use synthetic;

        let data = synthetic::gen_lognormal(100);
        let layer0 = Layer {
            data: vec![1.0, 2.0, 3.0].into_boxed_slice(),
        };
        let layer1 = Layer {
            data: vec![1.0, -2.0, -3.0].into_boxed_slice(),
        };
        let network = Network {
            layers: vec![layer0, layer1].into_boxed_slice(),
        };

        let model = NeuralModel {
            network,
            data,
            min_offset: 0x8FFFFFF,
            max_offset: 0x8FFFFFF,
        };

        for &v in model.data.iter() {
            // let result = model.data.eval(v).unwrap();
            assert_eq!(model.data[model.eval(v).unwrap() as usize], v);
        }
    }
}
