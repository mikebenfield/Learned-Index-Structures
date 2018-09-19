//! A fully connected neural network with ReLU activations.
//!
//! Little thought given to performance optimization at the moment.

#[derive(Debug, PartialEq)]
struct Layer {
    // row major order
    data: Box<[f32]>,
}

#[derive(Debug, PartialEq)]
struct Vector {
    data: Box<[f32]>,
}

impl Layer {
    fn apply_relu(&self, v: &Vector) -> Vector {
        let mut result = self.apply(v);
        for i in 0 .. result.data.len() {
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
    fn f() {
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
}
