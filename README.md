# Learned Index Structures

This is an implementation of Learned Index Structures as described in
[this](https://arxiv.org/abs/1712.01208) 2017 paper.

Normally in a database the mapping from key to index is done using a traditional
data structure, often a B Tree. In this paper, the authors proposed instead to
use function approximation to implement that mapping. Namely, they used neural
networks. To elaborate, they used a hierarchy of neural networks, with each
stage of the hierarchy doing model selection for the next stage. The hierarchy
ends either in a B Tree, which maps key to index for a much smaller selection of
records, or in a neural network, which provides an approximation to the correct
index. Since this is only an approximation, a linear or binary search must be
performed among nearby records to find the correct record.

My model is a simplified version of that in the paper: I use only one top level
neural net, which feeds into one of many B Trees.

## Usage

Training and inference is done using a combination of Rust programs in
'example/' and the Python script in 'py/'.

To generate 100000 data points (sampled from a log normal distribution) in the
file `data_filename`:
`cargo run --example data_filename 100000`
(You'll need Rust and related tools installed; the easiest way to do that is
using [rustup](https://rustup.rs).)

Then, to train a model on those data points, saving the result
to `out.toml`:
`python py/train.py --config examples/config.toml --index data_filename --save out.toml`
You'll need Keras and the Python `toml` library installed.

If you want to modify the type of model used, you can modify the
`examples/config.toml` file. The format is simple: the lines of the form "0 =
32" indicate the width of each layer. `btree_count` indicates how many btrees
are used.

Finally, to benchmark both a B Tree and the learned index structure on
looking up 10,000 records
`cargo run --release --example read_saved out.toml data_filename`

## Implementation notes

The authors of the paper above implemented inference for their models in native
code on the CPU. They gave two reasons for this choice:

1. Latency. A roundtrip to the GPU takes on the order of two microseconds.
   Python adds additional overhead, as does Tensorflow, which is optimized for
   larger models than used here.

2. Comparing apples to apples. Given that the new model is being compared
against a sequential data structure implemented on the CPU, the authors appear
to feel that it would not be justified to use different hardware for the new
model.

The first concern will hopefully be alleviated in the future as GPUs become more
closely integrated with CPUs and main memory.

In any case, I have followed the authors in this respect. Training in this
project happens in a Python script using Tensorflow. Inference happens in native
code, written in Rust.

For the reader unfamiliar with Rust, it's a newer systems programming language
suitable for the same domains as C or C++, but with many modern features,
notably a lifetime system that provides memory safety without garbage
collection.

## Benchmarks

Currently, on my system, executing the commands under `Usage` gives
these performance results:

| Model         | Runtime (sec) |
| ------------- | ------------- |
| B Tree        | 0.001         |
| Learned Model | 0.052         |

That is, the learned model is slow.

This is entirely predictable: the current implementation simply does matrix
multiplication scalar by scalar. I looked at the assembly output by the
compiler, and the code is not auto-vectorized at all.

I'm working on (and will hopefully finish very shortly) a more optimized version
using vectorized AVX instructions. This should be dramatically faster than the
current code. I'm interested to see whether it beats the B Tree.

Note that the authors of the paper used custom code generation techniques to
achieve substantially greater performance than an optimized B Tree.
Unfortunately their code is not publicly available.

## Discussion

I think the idea of treating the mapping from key to index as a function
approximation problem is really interesting.

Having worked with and thought about this idea for some time though, I am
skeptical that neural networks are the right tool. Note that, in the case of the
randomly generated log normal data, we are essentially just learning the
cumulative distribution function of the log normal distribution. This is a
relatively simple function, and a hierarchy of thousands of neural nets is an
extraordinarily heavyweight tool for this purpose.

There have been several interesting discussions online of this paper, notably
[this blog
post](http://databasearchitects.blogspot.com/2017/12/the-case-for-b-tree-index-structures.html).
The blogger mentions the possibility of using splines for function
approximation, and points out that B Trees in some sense already perform the
same partition/interpolate process as splines. See also comments by Tim Kraska,
one of the authors of of the original paper, where he clarifies that the paper
was intended to introduce the idea of machine learning for this application, not
necessarily to suggest that neural networks are the best tool for the purpose.

Finally, I'll mention one more point of view to illustrate why I am skeptical of
neural nets for this application. The authors talk about their model in terms of
precision gain. A given neural network in their hierarchy may reduce the
potential error in predicted index from 100M to 10k, so this is a precision gain
of 10,000. But assuming for simplicity that the B Tree we're replacing is
binary, even with that large precision gain we're only doing the work of about
lg(10,000) = 13 levels of the B Tree.

Traversing each layer of a B Tree requires very little computation. In contrast,
even a small neural network as used in the paper requires many thousands of
operations. Due to modern vectorized CPUs, with substantial engineering effort
the authors are able to make much of that massive amount of computation happen
in parallel and beat the performance of a B Tree, which cannot be parallelized.
Nevertheless, I believe in practice a more work-efficient approach would be
preferable, especially considering total throughput.

In particular, if I had an infinite amount of time, I would investigate
the following approaches:

1. GPU-based B Trees. Although a single B Tree search cannot be parallelized, it
   would be possible to batch indexes and perform many thousands in parallel on
   the GPU. This would not solve the latency issue, but for applications where throughput rather than latency is the concern, I'd be interested to see the results.

2. Other function approximation techniques. There are so many methods it's hard
to know where to start. I mentioned splines earlier. There are also Chebyshev
polynomials, Remez's algorithm, and many more.
