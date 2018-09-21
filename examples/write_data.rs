extern crate learned_index_structures;

use learned_index_structures::synthetic;

use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();
    let count: usize = args[2]
        .parse()
        .expect("Couldn't parse command line arguments");
    let data = synthetic::gen_lognormal(count);

    let mut file = File::create(&args[1]).expect("Unable to open file");
    for &datum in data.iter() {
        writeln!(file, "{}", datum).expect("Unable to write to file");
    }
}
