# Learned Index Structures

This will be an implementation of Learned Index Structures as described in
[this](https://arxiv.org/abs/1712.01208) 2017 paper.

Currently there's a B Tree implementation, some tests, and the beginning of the
actual neural net model from the paper.

## Usage

Training and inference is done using a combination of Rust programs in
'example/' and the Python script in 'py/'.

To generate 100000 data points in the file `data_filename`:
`cargo run --example data_filename 100000`

Then, to train a model on those data points, saving the result
to `out.toml`:
`python py/train.py --config examples/config.toml --index data_filename --save out.toml`
You'll need Keras and the Python `toml` library installed.

Finally, to benchmark both a B Tree and the learned index structure on
looking up 10,000 data points:
`cargo run --release --example read_saved out.toml data_filename`
