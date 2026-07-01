# neuro

A feed-forward neural network written from scratch in Rust, using only the
standard library — no `rand`, `ndarray`, BLAS, or ML crates. It learns XOR,
classifies Iris, and handles the 30-feature Wisconsin breast cancer dataset,
all loaded from CSV.

## Quick start

```bash
cargo run --release   # trains the three demos and prints a report
cargo test            # unit + integration + doc tests
```

## Features

SGD and Adam optimisers, MSE and cross-entropy loss, softmax outputs,
mini-batch gradient descent, L2 regularisation, dropout, early stopping,
Xavier/He weight init, and a gradient check that verifies back-prop against
finite differences.

## Results

From `cargo run --release`:

```
XOR           : gradient check 2.68e-5, all rows correct
Iris          : test accuracy 100%
Breast cancer : test accuracy ~99%
```

On the breast cancer run the validation loss starts rising while training loss
keeps dropping (overfitting), and early stopping rewinds to the best epoch.

## Modules

| Module        | Responsibility                                        |
|---------------|-------------------------------------------------------|
| `rng`         | xorshift32 random numbers                             |
| `activation`  | sigmoid / tanh / ReLU / softmax and derivatives       |
| `loss`        | MSE and cross-entropy                                  |
| `layer`       | one dense layer: forward + gradients                  |
| `optimizer`   | SGD and Adam                                          |
| `network`     | layer stack: forward, back-prop, gradient check       |
| `trainer`     | training loop, regularisation, early stopping         |
| `metrics`     | accuracy, argmax, confusion matrix                    |
| `data`        | XOR, CSV loader, normalisation, train/test split      |

`Layer::backward` computes gradients but does not change the weights; the
optimiser applies them. That split is what allows mini-batching, Adam, and the
gradient check. `network` holds the maths, `trainer` holds the training policy.

## How it works

Each layer computes `a = f(W·x + b)`. The forward pass runs left to right; the
backward pass sends the error `δ` right to left, leaving a gradient at every
weight.

```
forward:   x ──► W₁·x+b₁ ──f──► h ──► W₂·h+b₂ ──f──► prediction

backward:  ∂L/∂W₁ ◄── Wᵀ·δ·f'(z) ◄── δ = pred − target ◄── loss
```

## Using the library

```rust
use neuro::{Activation, Network, Rng};
use neuro::loss::Loss;
use neuro::optimizer::Optimizer;
use neuro::trainer::Trainer;

let mut rng = Rng::new(7);
let mut net = Network::new(4)
    .add(10, Activation::Relu, &mut rng)
    .add(3, Activation::Softmax, &mut rng);

Trainer::new(Optimizer::adam(0.02), Loss::CrossEntropy)
    .epochs(1500)
    .batch_size(16)
    .dropout(0.2)
    .l2(1e-4)
    .early_stopping(40)
    .train(&mut net, &train, Some(&validation), &mut rng);

let probabilities = net.predict(&features);
```

## Data

Bundled in `data/`: `iris.csv` (150×4, 3 classes) and `breast_cancer.csv`
(569×30, 2 classes). `data::load_csv` works on any CSV with a header, numeric
feature columns, and a categorical label in the last column.
