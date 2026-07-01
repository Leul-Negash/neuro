//! Optimisers: SGD and Adam.
//!
//! Config lives in [`Optimizer`]; Adam's moment buffers live in
//! [`OptimizerState`] so each training run starts with fresh state.

use crate::layer::{Gradients, Layer};

/// Optimisation algorithm and its hyper-parameters.
#[derive(Debug, Clone, Copy)]
pub enum Optimizer {
    /// `θ -= lr · g`.
    Sgd { learning_rate: f32 },
    /// Adam (Kingma & Ba, 2015).
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    /// Plain SGD with the given learning rate.
    pub fn sgd(learning_rate: f32) -> Self {
        Optimizer::Sgd { learning_rate }
    }

    /// Adam with defaults `β1=0.9, β2=0.999, ε=1e-8`.
    pub fn adam(learning_rate: f32) -> Self {
        Optimizer::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam's per-layer moment estimates: mean (`m`) and variance (`v`).
#[derive(Debug, Clone)]
struct Moments {
    mean: Gradients,
    variance: Gradients,
}

/// Mutable optimiser state for a given network shape.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    optimizer: Optimizer,
    moments: Vec<Moments>,
    /// Update count, used for Adam bias correction.
    step_count: i32,
}

impl OptimizerState {
    /// Creates state matching `layers`.
    pub fn new(optimizer: Optimizer, layers: &[Layer]) -> Self {
        let moments = layers
            .iter()
            .map(|l| Moments {
                mean: Gradients::zeros(l.neurons(), l.inputs()),
                variance: Gradients::zeros(l.neurons(), l.inputs()),
            })
            .collect();
        OptimizerState {
            optimizer,
            moments,
            step_count: 0,
        }
    }

    /// Updates every layer from its gradient.
    pub fn step(&mut self, layers: &mut [Layer], grads: &[Gradients]) {
        self.step_count += 1;
        for (idx, (layer, grad)) in layers.iter_mut().zip(grads).enumerate() {
            match self.optimizer {
                Optimizer::Sgd { learning_rate } => {
                    update_each(layer, grad, |w, g| w - learning_rate * g);
                }
                Optimizer::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    let t = self.step_count;
                    let m = &mut self.moments[idx];
                    adam_update(layer, grad, m, learning_rate, beta1, beta2, epsilon, t);
                }
            }
        }
    }
}

/// Applies `new = f(old, grad)` to every parameter.
fn update_each(layer: &mut Layer, grad: &Gradients, f: impl Fn(f32, f32) -> f32) {
    for (row, grow) in layer.weights.iter_mut().zip(&grad.weights) {
        for (w, g) in row.iter_mut().zip(grow) {
            *w = f(*w, *g);
        }
    }
    for (b, g) in layer.biases.iter_mut().zip(&grad.biases) {
        *b = f(*b, *g);
    }
}

/// Adam update for one layer; mutates parameters and moments in place.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    layer: &mut Layer,
    grad: &Gradients,
    moments: &mut Moments,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: i32,
) {
    // Bias correction for moments starting at zero.
    let bias1 = 1.0 - beta1.powi(t);
    let bias2 = 1.0 - beta2.powi(t);

    let step = |param: &mut f32, g: f32, m: &mut f32, v: &mut f32| {
        *m = beta1 * *m + (1.0 - beta1) * g;
        *v = beta2 * *v + (1.0 - beta2) * g * g;
        let m_hat = *m / bias1;
        let v_hat = *v / bias2;
        *param -= lr * m_hat / (v_hat.sqrt() + epsilon);
    };

    for o in 0..layer.weights.len() {
        for i in 0..layer.weights[o].len() {
            step(
                &mut layer.weights[o][i],
                grad.weights[o][i],
                &mut moments.mean.weights[o][i],
                &mut moments.variance.weights[o][i],
            );
        }
        step(
            &mut layer.biases[o],
            grad.biases[o],
            &mut moments.mean.biases[o],
            &mut moments.variance.biases[o],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::rng::Rng;

    /// Adam should reduce a single weight's contribution to a simple objective.
    #[test]
    fn adam_moves_parameters_downhill() {
        let mut rng = Rng::new(1);
        let mut layers = vec![Layer::random(2, 1, Activation::Sigmoid, &mut rng)];
        let mut state = OptimizerState::new(Optimizer::adam(0.1), &layers);

        // Constant positive gradient should steadily decrease the parameters.
        let before = layers[0].biases[0];
        for _ in 0..10 {
            let grad = Gradients {
                weights: vec![vec![1.0, 1.0]],
                biases: vec![1.0],
            };
            state.step(&mut layers, &[grad]);
        }
        assert!(layers[0].biases[0] < before);
    }
}
