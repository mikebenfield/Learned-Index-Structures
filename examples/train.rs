extern crate learned_index_structures;

use learned_index_structures::train;
use learned_index_structures::synthetic;

fn main() {
    train::read_toml("/Users/mike/trash.toml");
    // let data = synthetic::gen_lognormal(1000000);
    // train::train(&data, 4, 256, 30, &"py/train.py");
}
