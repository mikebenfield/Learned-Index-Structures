//! A fully connected neural network with ReLU activations.
//!
//! Little thought given to performance optimization at the moment. Lots of
//! unnecessary allocations, no explicit SIMD.

#[derive(Debug, PartialEq)]
struct Layer {
    // row major order
    data: Box<[f32]>,
}

#[derive(Debug, PartialEq)]
pub struct Vector {
    data: Box<[f32]>,
}

impl Layer {
    fn apply_relu(&self, v: &Vector) -> Vector {
        let mut result = self.apply(v);
        for i in 0..result.data.len() {
            if result.data[i] < 0.0 {
                result.data[i] = 0.0;
            }
        }
        result
    }

    fn apply(&self, v: &Vector) -> Vector {
        debug_assert!(self.data.len() % v.data.len() == 0);
        let out_dim = self.data.len() / v.data.len();
        let mut result = Vec::with_capacity(out_dim);
        for row in 0..out_dim {
            let mut val = 0.0;
            for column in 0..v.data.len() {
                val += v.data[column] * self.data[row * v.data.len() + column];
            }
            result.push(val);
        }
        Vector {
            data: result.into_boxed_slice(),
        }
    }
}

pub struct Network {
    layers: Box<[Layer]>,
}

impl Network {
    fn apply0(&self, v: &Vector, layer: usize) -> Vector {
        if layer == self.layers.len() - 1 {
            self.layers[layer].apply(v)
        } else {
            let result = self.layers[layer].apply_relu(v);
            self.apply0(&result, layer + 1)
        }
    }

    pub fn apply(&self, v: &Vector) -> Vector {
        self.apply0(v, 0)
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
}
