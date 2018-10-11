//! A fully connected neural network with Leaky ReLU activations.
//!
//! Not optimized at the moment, but at least there are fewer superfluous
//! allocations now.

use std::mem;
use std::ops::{Index, IndexMut};
use std::slice;

use toml::Value;

// since we will eventually need 32-byte aligned memory for AVX instructions, we
// have to jump through some hoops to allocate and deallocate

#[repr(align(32))]
struct Aligned(u8);

const LEAKY_SLOPE: f32 = 0.3;

fn ceil_div(dividend: usize, divisor: usize) -> usize {
    (dividend + divisor - 1) / divisor
}

fn allocate_aligned_f32(len: usize) -> *mut f32 {
    let aligned_len = ceil_div(4 * len, mem::align_of::<Aligned>());
    let mut v: Vec<Aligned> = Vec::with_capacity(aligned_len);
    let ptr = v.as_mut_ptr();
    mem::forget(v);
    unsafe { mem::transmute(ptr) }
}

fn deallocate_aligned_f32(ptr: *mut f32, len: usize) {
    let aligned_len = ceil_div(len, mem::size_of::<Aligned>());
    unsafe {
        let _: Vec<Aligned> = Vec::from_raw_parts(mem::transmute(ptr), 0, aligned_len);
    }
}

fn value_array_arrays_float(v: &Value) -> Box<[Box<[f32]>]> {
    use self::Value::*;

    if let Array(a) = v {
        let mut arrays: Vec<Box<[f32]>> = Vec::new();
        for value in a.iter() {
            if let Array(immediate_array) = value {
                let mut array: Vec<f32> = Vec::new();
                for integer in immediate_array.iter() {
                    if let Float(i) = integer {
                        array.push(*i as f32);
                    } else {
                        panic!("Invalid TOML format");
                    }
                }
                arrays.push(array.into_boxed_slice());
            } else {
                panic!("Invalid TOML format");
            }
        }
        return arrays.into_boxed_slice();
    } else {
        panic!("Invalid TOML format");
    }
}

#[repr(C)]
pub struct FirstLayer {
    data: *mut f32,
    bias: *mut f32,
    size: usize,
}

impl Index<usize> for FirstLayer {
    type Output = f32;

    fn index(&self, i: usize) -> &f32 {
        if self.size <= i {
            panic!("FirstLayer: index out of bounds");
        } else {
            unsafe { &*self.data.offset(i as isize) }
        }
    }
}

impl IndexMut<usize> for FirstLayer {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        if self.size <= i {
            panic!("FirstLayer: index out of bounds");
        } else {
            unsafe { &mut *self.data.offset(i as isize) }
        }
    }
}

impl FirstLayer {
    pub fn new(size: usize) -> Self {
        FirstLayer {
            data: allocate_aligned_f32(size),
            bias: allocate_aligned_f32(size),
            size,
        }
    }

    fn bias(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.bias, self.size) }
    }

    fn bias_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.bias, self.size) }
    }
}

impl Drop for FirstLayer {
    fn drop(&mut self) {
        deallocate_aligned_f32(self.data, self.size);
        deallocate_aligned_f32(self.bias, self.size);
    }
}

#[repr(C)]
pub struct LastLayer {
    data: *mut f32,
    size: usize,
    bias: f32,
}

impl Index<usize> for LastLayer {
    type Output = f32;

    fn index(&self, i: usize) -> &f32 {
        if self.size <= i {
            panic!("FirstLayer: index out of bounds");
        } else {
            unsafe { &*self.data.offset(i as isize) }
        }
    }
}

impl IndexMut<usize> for LastLayer {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        if self.size <= i {
            panic!("FirstLayer: index too small");
        } else {
            unsafe { &mut *self.data.offset(i as isize) }
        }
    }
}

impl LastLayer {
    pub fn new(size: usize) -> Self {
        LastLayer {
            data: allocate_aligned_f32(size),
            bias: 0.0,
            size,
        }
    }

    fn bias(&self) -> &f32 {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut f32 {
        &mut self.bias
    }
}

impl Drop for LastLayer {
    fn drop(&mut self) {
        deallocate_aligned_f32(self.data, self.size);
    }
}

#[repr(C)]
pub struct InteriorLayer {
    data: *mut f32,
    bias: *mut f32,
    rows: usize,
    columns: usize,
}

impl InteriorLayer {
    fn new(rows: usize, columns: usize) -> Self {
        InteriorLayer {
            data: allocate_aligned_f32(rows * columns),
            bias: allocate_aligned_f32(rows),
            rows,
            columns,
        }
    }

    fn bias(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.bias, self.rows) }
    }

    fn bias_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.bias, self.rows) }
    }
}

impl Drop for InteriorLayer {
    fn drop(&mut self) {
        deallocate_aligned_f32(self.data, self.rows * self.columns);
        deallocate_aligned_f32(self.data, self.rows);
    }
}

impl Index<(usize, usize)> for InteriorLayer {
    type Output = f32;
    fn index(&self, i: (usize, usize)) -> &f32 {
        if i.0 >= self.rows || i.1 >= self.columns {
            panic!("InteriorLayer: index out of bounds")
        } else {
            unsafe { &*self.data.offset((self.columns * i.0 + i.1) as isize) }
        }
    }
}

impl IndexMut<(usize, usize)> for InteriorLayer {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut f32 {
        if i.0 >= self.rows || i.1 >= self.columns {
            panic!("InteriorLayer: index out of bounds")
        } else {
            unsafe { &mut *self.data.offset((self.columns * i.0 + i.1) as isize) }
        }
    }
}

pub struct Network {
    first_layer: FirstLayer,
    last_layer: LastLayer,
    interior_layers: Box<[InteriorLayer]>,
}

impl Network {
    pub fn apply_buffer(&self, x: f32, buf1: &mut [f32], buf2: &mut [f32]) -> f32 {
        // first layer
        debug_assert!(buf1.len() >= self.first_layer.size);
        for i in 0..self.first_layer.size {
            buf1[i] = x * self.first_layer[i] + self.first_layer.bias()[i];
            if buf1[i] < 0.0 {
                buf1[i] *= LEAKY_SLOPE;
            }
        }

        // interior layers
        fn write_layer(
            layers: &[InteriorLayer],
            layer_index: usize,
            read: &mut [f32],
            write: &mut [f32],
        ) {
            if layer_index >= layers.len() {
                return;
            }
            let layer = &layers[layer_index];
            debug_assert!(read.len() >= layer.columns);
            debug_assert!(write.len() >= layer.rows);
            for row in 0..layer.rows {
                write[row] = 0.0;
                for col in 0..layer.columns {
                    write[row] += layer[(row, col)] * read[col];
                }
                write[row] += layer.bias()[row];
                if write[row] < 0.0 {
                    write[row] *= LEAKY_SLOPE;
                }
            }
            write_layer(layers, layer_index + 1, write, read);
        }

        write_layer(&self.interior_layers, 0, buf1, buf2);

        let mut result = 0.0f32;

        // last layer
        let read = if self.interior_layers.len() % 2 == 0 {
            buf1
        } else {
            buf2
        };

        debug_assert!(read.len() >= self.last_layer.size);

        for row in 0..self.last_layer.size {
            result += self.last_layer[row] * read[row];
        }

        result += *self.last_layer.bias();

        if result < 0.0 {
            result *= LEAKY_SLOPE;
        }

        result
    }

    /// What size of buffer is necessary to pass to `apply_buffer`?
    pub fn buf_size(&self) -> usize {
        use std::cmp::max;

        let mut bufsize = 0usize;
        bufsize = max(bufsize, self.first_layer.size);
        for layer in self.interior_layers.iter() {
            bufsize = max(bufsize, layer.rows);
        }

        bufsize
    }

    /// Create a Network from a TOML value in my custom format.
    ///
    /// This is the only way to create a Network outside this module at the
    /// moment.
    pub fn from_toml(v: &Value) -> Self {
        use self::Value::*;

        let table = if let Table(table) = v {
            table
        } else {
            panic!("Bad TOML format");
        };

        let mut last_layer_index = 0usize;
        for i in 0.. {
            let layer_var = format!("layer{}", i);
            if let Some(_layer) = table.get(&layer_var) {
                last_layer_index = i;
            } else {
                break;
            }
        }

        if last_layer_index < 2 {
            panic!("Need at least two layers")
        }

        // first layer

        let first_layer_toml = if let Some(layer) = table.get("layer0") {
            layer
        } else {
            panic!("Bad TOML format");
        };

        let arrays = value_array_arrays_float(first_layer_toml);

        let mut first_layer = FirstLayer::new(arrays[0].len());

        unsafe {
            slice::from_raw_parts_mut(first_layer.data, first_layer.size)
                .copy_from_slice(&arrays[0]);
        }
        first_layer.bias_mut().copy_from_slice(&arrays[1]);

        // interior layers

        let mut interior_layers = Vec::new();

        let mut previous_layer_rows = first_layer.size;

        for layer_index in 1..last_layer_index {
            let layer_toml = if let Some(layer) = table.get(&format!("layer{}", layer_index)) {
                layer
            } else {
                unreachable!();
            };

            let arrays = value_array_arrays_float(layer_toml);

            if arrays[0].len() % previous_layer_rows != 0 {
                panic!("Invalid layer sizes: layer {}", layer_index);
            }

            let columns = previous_layer_rows;
            let rows = arrays[0].len() / previous_layer_rows;

            let mut layer = InteriorLayer::new(rows, columns);

            unsafe {
                slice::from_raw_parts_mut(layer.data, rows * columns).copy_from_slice(&arrays[0]);
            }
            layer.bias_mut().copy_from_slice(&arrays[1]);

            interior_layers.push(layer);

            previous_layer_rows = rows;
        }

        // last layer

        let last_layer_toml = if let Some(layer) = table.get(&format!("layer{}", last_layer_index))
        {
            layer
        } else {
            panic!("Bad TOML format");
        };

        let arrays = value_array_arrays_float(last_layer_toml);

        let mut last_layer = LastLayer::new(arrays[0].len());
        unsafe {
            slice::from_raw_parts_mut(last_layer.data, last_layer.size).copy_from_slice(&arrays[0]);
        }
        *last_layer.bias_mut() = arrays[1][0];

        Network {
            first_layer,
            last_layer,
            interior_layers: interior_layers.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f() {
        let mut first = FirstLayer::new(2);
        first[0] = 1.0;
        first[1] = 2.0;
        first.bias_mut()[0] = -3.0;
        first.bias_mut()[1] = 4.0;

        let mut interior = InteriorLayer::new(2, 2);
        interior[(0, 0)] = 1.0;
        interior[(0, 1)] = 2.0;
        interior[(1, 0)] = 3.0;
        interior[(1, 1)] = 4.0;
        interior.bias_mut()[0] = 5.0;
        interior.bias_mut()[1] = 5.0;

        let mut last = LastLayer::new(2);
        last[0] = -1.0;
        last[1] = 1.0;
        *last.bias_mut() = 2.0;

        let network = Network {
            first_layer: first,
            interior_layers: vec![interior].into_boxed_slice(),
            last_layer: last,
        };

        let mut buf1 = vec![0.0, 0.0];
        let mut buf2 = vec![0.0, 0.0];

        let result = network.apply_buffer(1.0, &mut buf1, &mut buf2);

        const GOLDEN: f32 = 12.8;

        assert!((result - GOLDEN).abs() < 0.0001);
    }
}
