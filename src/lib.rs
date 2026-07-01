//! # neuro
//!
//! A feed-forward neural network with back-propagation, written using only the
//! Rust standard library (no `rand`, `ndarray`, or BLAS).
//!
//! Features: SGD and Adam optimisers, MSE and cross-entropy losses, softmax
//! outputs, mini-batching, L2 regularisation, dropout, early stopping, and a
//! gradient check ([`Network::gradient_check`]).
//!
//! ## Modules
//!
//! | Module        | Responsibility                                        |
//! |---------------|-------------------------------------------------------|
//! | [`rng`]       | xorshift random numbers                               |
//! | [`activation`]| sigmoid / tanh / ReLU / softmax and derivatives       |
//! | [`loss`]      | MSE and cross-entropy                                  |
//! | [`layer`]     | one dense layer: forward + gradients                  |
//! | [`optimizer`] | SGD and Adam                                          |
//! | [`network`]   | layer stack: forward, back-prop, gradient check       |
//! | [`trainer`]   | training loop, regularisation, early stopping         |
//! | [`metrics`]   | accuracy, argmax, confusion matrix                    |
//! | [`data`]      | XOR, CSV loader, normalisation, train/test split      |
//!
//! ## Example
//!
//! ```
//! use neuro::{Activation, Network, Rng};
//! use neuro::loss::Loss;
//! use neuro::optimizer::Optimizer;
//! use neuro::trainer::Trainer;
//!
//! let mut rng = Rng::new(7);
//! let mut net = Network::new(2)
//!     .add(4, Activation::Tanh, &mut rng)
//!     .add(1, Activation::Sigmoid, &mut rng);
//!
//! Trainer::new(Optimizer::adam(0.05), Loss::Mse)
//!     .epochs(400)
//!     .batch_size(4)
//!     .train(&mut net, &neuro::data::xor(), None, &mut rng);
//!
//! assert!(net.predict(&[1.0, 0.0])[0] > 0.5); // XOR(1, 0) ≈ 1
//! ```

pub mod activation;
pub mod data;
pub mod layer;
pub mod loss;
pub mod metrics;
pub mod network;
pub mod optimizer;
pub mod rng;
pub mod trainer;

// Re-export the most-used types at the crate root.
pub use activation::Activation;
pub use network::{Network, Samples};
pub use rng::Rng;
