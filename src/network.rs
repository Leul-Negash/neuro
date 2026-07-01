//! The feed-forward network: a stack of [`Layer`]s plus the forward pass,
//! back-prop (with dropout) and gradient checking. Training policy lives in
//! [`crate::trainer`].
//!
//! Built with a fluent builder:
//!
//! ```
//! use neuro::{Activation, Network, Rng};
//!
//! let mut rng = Rng::new(42);
//! let net = Network::new(2)
//!     .add(4, Activation::Tanh, &mut rng)
//!     .add(1, Activation::Sigmoid, &mut rng);
//! assert_eq!(net.layer_sizes(), vec![2, 4, 1]);
//! ```

use crate::activation::Activation;
use crate::layer::{Forward, Gradients, Layer};
use crate::loss::Loss;
use crate::metrics::argmax;
use crate::rng::Rng;

/// A labelled training/evaluation set: each entry is `(features, target)`.
pub type Samples = Vec<(Vec<f32>, Vec<f32>)>;

/// Cached values from a training forward pass, consumed by back-prop.
pub(crate) struct ForwardTrace {
    /// Input fed to each layer (post-dropout from the layer below).
    pub inputs: Vec<Vec<f32>>,
    /// Each layer's pre-/post-activation values.
    pub forwards: Vec<Forward>,
    /// Dropout masks per layer (all-ones when disabled).
    pub masks: Vec<Vec<f32>>,
}

/// A multi-layer feed-forward network.
#[derive(Debug, Clone)]
pub struct Network {
    pub(crate) layers: Vec<Layer>,
    input_size: usize,
}

impl Network {
    /// Starts a new network that expects `input_size` features per sample.
    pub fn new(input_size: usize) -> Self {
        Network {
            layers: Vec::new(),
            input_size,
        }
    }

    /// Appends a dense layer; its input width is inferred from the previous
    /// layer (or `input_size` for the first).
    pub fn add(mut self, neurons: usize, activation: Activation, rng: &mut Rng) -> Self {
        let inputs = self.layers.last().map_or(self.input_size, Layer::neurons);
        self.layers
            .push(Layer::random(inputs, neurons, activation, rng));
        self
    }

    /// Returns the width of every stage: input size followed by each layer.
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.input_size];
        sizes.extend(self.layers.iter().map(Layer::neurons));
        sizes
    }

    /// Total number of trainable parameters (weights + biases).
    pub fn parameter_count(&self) -> usize {
        self.layer_sizes()
            .windows(2)
            .map(|w| w[0] * w[1] + w[1])
            .sum()
    }

    /// Forward pass for inference. No dropout.
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_size, "wrong number of features");
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current).output;
        }
        current
    }

    /// Forward pass for training: caches per-layer values and, when
    /// `dropout > 0`, applies inverted dropout to hidden layers.
    pub(crate) fn forward_train(&self, input: &[f32], dropout: f32, rng: &mut Rng) -> ForwardTrace {
        let last = self.layers.len() - 1;
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut forwards = Vec::with_capacity(self.layers.len());
        let mut masks = Vec::with_capacity(self.layers.len());

        let mut current = input.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            let forward = layer.forward(&current);
            inputs.push(current);

            // Dropout on hidden layers only.
            let mut output = forward.output.clone();
            let mask = if dropout > 0.0 && idx != last {
                dropout_mask(output.len(), dropout, rng)
            } else {
                vec![1.0; output.len()]
            };
            for (o, m) in output.iter_mut().zip(&mask) {
                *o *= m;
            }

            masks.push(mask);
            forwards.push(forward);
            current = output;
        }

        ForwardTrace {
            inputs,
            forwards,
            masks,
        }
    }

    /// Back-prop for one sample. Returns per-layer gradients.
    pub(crate) fn backprop(
        &self,
        trace: &ForwardTrace,
        target: &[f32],
        loss: Loss,
    ) -> Vec<Gradients> {
        let mut grads: Vec<Gradients> = Vec::with_capacity(self.layers.len());
        let last = self.layers.len() - 1;

        // Output layer: delta comes from the loss.
        let out_layer = &self.layers[last];
        let out_forward = &trace.forwards[last];
        let delta = loss.output_delta(&out_forward.output, target, |i| {
            out_layer
                .activation()
                .derivative(out_forward.pre_activation[i])
        });
        let (grad, mut grad_input) = out_layer.backward(&trace.inputs[last], &delta);
        grads.push(grad);

        // Hidden layers: chain through the dropout mask, then the activation.
        for idx in (0..last).rev() {
            let layer = &self.layers[idx];
            let forward = &trace.forwards[idx];
            let mask = &trace.masks[idx];

            let delta: Vec<f32> = grad_input
                .iter()
                .enumerate()
                .map(|(j, &g)| {
                    g * mask[j] * layer.activation().derivative(forward.pre_activation[j])
                })
                .collect();

            let (grad, next_grad_input) = layer.backward(&trace.inputs[idx], &delta);
            grads.push(grad);
            grad_input = next_grad_input;
        }

        grads.reverse(); // built output-first; restore input-first order
        grads
    }

    /// Evaluates on `data`, returning `(mean_loss, accuracy)`.
    pub fn evaluate(&self, data: &Samples, loss: Loss) -> (f32, f32) {
        if data.is_empty() {
            return (0.0, 0.0);
        }
        let mut total_loss = 0.0;
        let mut correct = 0;
        for (input, target) in data {
            let prediction = self.predict(input);
            total_loss += loss.value(&prediction, target);
            if argmax(&prediction) == argmax(target) {
                correct += 1;
            }
        }
        let n = data.len() as f32;
        (total_loss / n, correct as f32 / n)
    }

    /// Checks back-prop against central finite differences for one sample,
    /// returning the largest relative error (a correct impl gives ~1e-4 or less).
    pub fn gradient_check(&self, input: &[f32], target: &[f32], loss: Loss) -> f32 {
        // Larger step trades truncation for less f32 cancellation in f(x±h).
        const EPS: f32 = 1e-2;
        // Floor the denominator so near-zero gradients use absolute error.
        const DENOM_FLOOR: f32 = 1e-2;

        // Analytic gradients (dropout disabled for determinism).
        let mut rng = Rng::new(1);
        let trace = self.forward_train(input, 0.0, &mut rng);
        let analytic = self.backprop(&trace, target, loss);

        let mut max_rel_error: f32 = 0.0;
        let mut probe = self.clone();

        for (l, layer_grad) in analytic.iter().enumerate() {
            for (o, row) in layer_grad.weights.iter().enumerate() {
                for (i, &a) in row.iter().enumerate() {
                    let original = probe.layers[l].weights[o][i];

                    probe.layers[l].weights[o][i] = original + EPS;
                    let loss_plus = loss.value(&probe.predict(input), target);
                    probe.layers[l].weights[o][i] = original - EPS;
                    let loss_minus = loss.value(&probe.predict(input), target);
                    probe.layers[l].weights[o][i] = original; // restore

                    let numeric = (loss_plus - loss_minus) / (2.0 * EPS);
                    let denom = (numeric.abs() + a.abs()).max(DENOM_FLOOR);
                    max_rel_error = max_rel_error.max((numeric - a).abs() / denom);
                }
            }
        }
        max_rel_error
    }
}

/// Builds an inverted-dropout mask: kept units carry `1/(1-rate)`, the rest `0`.
fn dropout_mask(len: usize, rate: f32, rng: &mut Rng) -> Vec<f32> {
    let keep = 1.0 - rate;
    let scale = 1.0 / keep;
    (0..len)
        .map(|_| if rng.next_f32() < keep { scale } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_infers_dimensions() {
        let mut rng = Rng::new(1);
        let net = Network::new(3).add(5, Activation::Relu, &mut rng).add(
            2,
            Activation::Sigmoid,
            &mut rng,
        );
        assert_eq!(net.layer_sizes(), vec![3, 5, 2]);
        // (3*5 + 5) + (5*2 + 2) = 20 + 12 = 32
        assert_eq!(net.parameter_count(), 32);
    }

    #[test]
    fn predict_has_correct_output_width() {
        let mut rng = Rng::new(2);
        let net = Network::new(2).add(3, Activation::Tanh, &mut rng);
        assert_eq!(net.predict(&[0.0, 1.0]).len(), 3);
    }

    /// Back-prop must agree with finite differences for every loss/activation.
    #[test]
    fn gradient_check_passes() {
        let mut rng = Rng::new(7);

        // MSE with sigmoid output.
        let net = Network::new(3).add(4, Activation::Tanh, &mut rng).add(
            2,
            Activation::Sigmoid,
            &mut rng,
        );
        let err = net.gradient_check(&[0.5, -0.3, 0.8], &[1.0, 0.0], Loss::Mse);
        assert!(err < 1e-2, "MSE gradient error too large: {err}");

        // Cross-entropy with softmax output.
        let net = Network::new(3).add(5, Activation::Relu, &mut rng).add(
            3,
            Activation::Softmax,
            &mut rng,
        );
        let err = net.gradient_check(&[0.2, 0.9, -0.4], &[0.0, 1.0, 0.0], Loss::CrossEntropy);
        assert!(err < 1e-2, "cross-entropy gradient error too large: {err}");
    }
}
