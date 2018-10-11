//! A model consisting of a top level neural net that selects one of several B
//! Trees

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;

use toml::{self, Value};

use btree::BTree;
use model::Model;
use neural::Network;

use self::Value::*;

pub struct ForwardingModel {
    net: Network,
    btrees: Box<[BTree<f32, u32>]>,
    max_prediction: u32,
}

impl Model<f32, u32> for ForwardingModel {
    fn eval(&self, key: f32) -> Option<u32> {
        let buf_size = self.net.buf_size();
        let mut buf1 = vec![0.0f32; buf_size];
        let mut buf2 = vec![0.0f32; buf_size];
        let predicted_label = self.net.apply_buffer(key, &mut buf1, &mut buf2);
        let model =
            ((predicted_label / self.max_prediction as f32) * self.btrees.len() as f32) as usize;
        self.btrees[model].eval(key)
    }

    fn eval_many(&self, keys: &[f32], indices: &mut [Option<u32>]) {
        let buf_size = self.net.buf_size();
        let mut buf1 = vec![0.0f32; buf_size];
        let mut buf2 = vec![0.0f32; buf_size];
        for (i, &key) in keys.iter().enumerate() {
            let predicted_label = self.net.apply_buffer(key, &mut buf1, &mut buf2);
            let model = ((predicted_label / self.max_prediction as f32) * self.btrees.len() as f32)
                as usize;
            indices[i] = self.btrees[model].eval(key)
        }
    }
}

fn value_array_arrays(v: &Value) -> Box<[Box<[u32]>]> {
    if let Array(a) = v {
        let mut arrays: Vec<Box<[u32]>> = Vec::new();
        for value in a.iter() {
            if let Array(immediate_array) = value {
                let mut array: Vec<u32> = Vec::new();
                for integer in immediate_array.iter() {
                    // println!("OK");
                    // println!("it's {:?}", integer);
                    if let Integer(i) = integer {
                        array.push(*i as u32);
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

pub fn read_data<P>(data_path: &P) -> Box<[f32]>
where
    P: AsRef<Path>,
{
    read_data0(data_path.as_ref())
}

fn read_data0(data_path: &Path) -> Box<[f32]> {
    use std::string::String;
    let mut result = Vec::new();
    let mut buf = String::new();
    let mut file = BufReader::new(File::open(data_path).expect("Unable to open data file"));

    loop {
        if let Ok(_) = file.read_line(&mut buf) {
            if buf.len() == 0 {
                break;
            } else if buf.len() == 1 {
                continue;
            }
            buf.pop(); // drop the newline
            let value = f32::from_str(&buf).expect("Invalid data format");
            result.push(value);
            buf.clear();
        } else {
            panic!("file read error");
        }
    }
    result.into_boxed_slice()
}

impl ForwardingModel {
    pub fn read_toml<P>(toml_path: &P, data: &Box<[f32]>) -> Self
    where
        P: AsRef<Path>,
    {
        Self::read_toml0(toml_path.as_ref(), data)
    }

    fn read_toml0(toml_path: &Path, data: &Box<[f32]>) -> Self {
        use std::cmp::max;

        let s = {
            use std::string::String;
            let mut buf = String::new();
            let mut file = File::open(toml_path).expect("Unable to open TOML file");
            file.read_to_string(&mut buf)
                .expect("Unable to read TOML file");
            buf
        };

        let value: Value = toml::from_str(&s).expect("Unable to parse TOML file");

        let table = if let &Table(ref table) = &value {
            table
        } else {
            panic!("Bad TOML format");
        };

        let indices = if let Some(indices) = table.get("btree_indices") {
            indices
        } else {
            panic!("Invalid TOML format");
        };

        let arrays = value_array_arrays(&indices);

        let mut max_prediction: u32 = 0;

        let btrees: Vec<BTree<f32, u32>> = arrays
            .iter()
            .map(|array| {
                let mut btree = BTree::new();
                for &index in array.iter() {
                    max_prediction = max(index, max_prediction);
                    btree.insert(data[index as usize], index);
                }
                btree
            })
            .collect();

        let network = Network::from_toml(&value);

        return ForwardingModel {
            net: network,
            btrees: btrees.into_boxed_slice(),
            max_prediction,
        };

    }
}
