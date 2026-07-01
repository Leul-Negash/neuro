//! A dense (fully-connected) layer.
//!
//! `forward` and `backward` are both pure: `backward` returns the gradients but
//! does not update the weights. Applying the update is the optimiser's job (see
//! [`crate::optimizer`]), which is what allows mini-batching and Adam.

use crate::activation::{Activation, softmax};
use crate::rng::Rng;

/// The intermediate values produced by [`Layer::forward`], kept so that
/// back-propagation can reuse them instead of recomputing.
#[derive(Debug, Clone)]
pub struct Forward {
    /// Pre-activations `z = W·x + b`, one per neuron.
    pub pre_activation: Vec<f32>,
    /// Post-activations `a = f(z)`, the layer's actual output.
    pub output: Vec<f32>,
}

/// Gradients of the loss w.r.t. a layer's parameters, matching their shapes.
#[derive(Debug, Clone)]
pub struct Gradients {
    /// `∂L/∂W`, same shape as the weight matrix.
    pub weights: Vec<Vec<f32>>,
    /// `∂L/∂b`, one per neuron.
    pub biases: Vec<f32>,
}

impl Gradients {
    /// All-zero gradients shaped like a layer with `neurons × inputs` weights.
    pub fn zeros(neurons: usize, inputs: usize) -> Self {
        Gradients {
            weights: vec![vec![0.0; inputs]; neurons],
            biases: vec![0.0; neurons],
        }
    }

    /// Adds another gradient into this one.
    pub fn add(&mut self, other: &Gradients) {
        for (row, orow) in self.weights.iter_mut().zip(&other.weights) {
            for (w, ow) in row.iter_mut().zip(orow) {
                *w += ow;
            }
        }
        for (b, ob) in self.biases.iter_mut().zip(&other.biases) {
            *b += ob;
        }
    }

    /// Scales every gradient by `factor`.
    pub fn scale(&mut self, factor: f32) {
        for row in &mut self.weights {
            for w in row {
                *w *= factor;
            }
        }
        for b in &mut self.biases {
            *b *= factor;
        }
    }
}

/// A dense layer: `output = activation(weights · input + bias)`.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Row `o` holds the incoming weights for neuron `o`. Shape: `[neurons][inputs]`.
    pub(crate) weights: Vec<Vec<f32>>,
    /// One bias per neuron. Shape: `[neurons]`.
    pub(crate) biases: Vec<f32>,
    /// The non-linearity applied to every neuron in this layer.
    pub(crate) activation: Activation,
}

impl Layer {
    /// Random layer using He init for ReLU and Xavier/Glorot otherwise.
    pub fn random(inputs: usize, neurons: usize, activation: Activation, rng: &mut Rng) -> Self {
        let limit = match activation {
            // He uniform: ±sqrt(6 / fan_in).
            Activation::Relu => (6.0 / inputs as f32).sqrt(),
            // Glorot uniform: ±sqrt(6 / (fan_in + fan_out)).
            _ => (6.0 / (inputs + neurons) as f32).sqrt(),
        };

        let weights = (0..neurons)
            .map(|_| (0..inputs).map(|_| rng.uniform(-limit, limit)).collect())
            .collect();
        let biases = vec![0.0; neurons];

        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// Number of neurons (i.e. the size of this layer's output vector).
    pub fn neurons(&self) -> usize {
        self.biases.len()
    }

    /// Number of inputs each neuron expects.
    pub fn inputs(&self) -> usize {
        self.weights.first().map_or(0, Vec::len)
    }

    /// The layer's activation function.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Forward pass for one input vector.
    pub fn forward(&self, input: &[f32]) -> Forward {
        let pre_activation: Vec<f32> = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(weights, &bias)| dot(weights, input) + bias)
            .collect();

        let output = if self.activation.is_vector_wise() {
            softmax(&pre_activation)
        } else {
            pre_activation
                .iter()
                .map(|&z| self.activation.apply(z))
                .collect()
        };

        Forward {
            pre_activation,
            output,
        }
    }

    /// Back-prop given `delta = ∂L/∂z`. Returns the parameter gradients and
    /// `∂L/∂input` for the previous layer; `delta` already includes the
    /// activation derivative, so this is just `∂L/∂W = delta ⊗ input` and
    /// `∂L/∂input = Wᵀ·delta`.
    pub fn backward(&self, input: &[f32], delta: &[f32]) -> (Gradients, Vec<f32>) {
        let mut grad = Gradients::zeros(self.neurons(), input.len());
        let mut grad_input = vec![0.0; input.len()];

        for (o, row) in self.weights.iter().enumerate() {
            grad.biases[o] = delta[o];
            for (i, &w) in row.iter().enumerate() {
                grad.weights[o][i] = delta[o] * input[i];
                grad_input[i] += delta[o] * w;
            }
        }

        (grad, grad_input)
    }
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product needs equal lengths");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shapes_match_neuron_count() {
        let mut rng = Rng::new(1);
        let layer = Layer::random(3, 4, Activation::Sigmoid, &mut rng);
        let out = layer.forward(&[0.1, 0.2, 0.3]);
        assert_eq!(out.output.len(), 4);
        assert_eq!(out.pre_activation.len(), 4);
    }

    #[test]
    fn softmax_layer_outputs_a_distribution() {
        let mut rng = Rng::new(1);
        let layer = Layer::random(3, 4, Activation::Softmax, &mut rng);
        let out = layer.forward(&[0.5, -0.5, 1.0]);
        let sum: f32 = out.output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gradient_shapes_are_correct() {
        let mut rng = Rng::new(2);
        let layer = Layer::random(3, 2, Activation::Tanh, &mut rng);
        let (grad, grad_input) = layer.backward(&[1.0, 2.0, 3.0], &[0.1, -0.2]);
        assert_eq!(grad.weights.len(), 2);
        assert_eq!(grad.weights[0].len(), 3);
        assert_eq!(grad.biases.len(), 2);
        assert_eq!(grad_input.len(), 3);
    }

    #[test]
    fn gradients_accumulate_and_scale() {
        let mut a = Gradients::zeros(1, 2);
        let b = Gradients {
            weights: vec![vec![2.0, 4.0]],
            biases: vec![6.0],
        };
        a.add(&b);
        a.add(&b);
        a.scale(0.5);
        assert_eq!(a.weights[0], vec![2.0, 4.0]);
        assert_eq!(a.biases[0], 6.0);
    }
}
