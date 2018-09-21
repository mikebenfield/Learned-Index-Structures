extern crate learned_index_structures;

use std::env;
use std::time::Duration;

use learned_index_structures::bench;
use learned_index_structures::btree::BTree;
use learned_index_structures::forwarding_model::{self, ForwardingModel};
use learned_index_structures::model::Model;

fn duration_to_secs(dur: Duration) -> f64 {
    let mut secs = dur.as_secs() as f64;
    secs += dur.subsec_nanos() as f64 / 1000000000.0;
    secs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let data = forwarding_model::read_data(&args[2]);
    let model = ForwardingModel::read_toml(&args[1], &data);
    println!(
        "{:.4}",
        duration_to_secs(bench::bench(&model, &data, 100000))
    );
    let mut btree = BTree::new();
    for i in 0..data.len() {
        btree.insert(data[i], i as u32);
    }
    println!(
        "{:.4}",
        duration_to_secs(bench::bench(&btree, &data, 100000))
    );
}
