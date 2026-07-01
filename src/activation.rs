//! Activation functions and their derivatives.
//!
//! Sigmoid, Tanh and ReLU are element-wise (`apply`/`derivative`). Softmax is
//! the exception: it spans the whole layer (see [`softmax`]) and its gradient is
//! handled together with cross-entropy, where it reduces to `output - target`.

/// Non-linearity applied after a neuron's weighted sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `1 / (1 + e^-z)`, range `(0, 1)`.
    Sigmoid,
    /// `tanh(z)`, range `(-1, 1)`.
    Tanh,
    /// `max(0, z)`.
    Relu,
    /// Normalised exponential over the layer; use with cross-entropy.
    Softmax,
}

impl Activation {
    /// True only for [`Activation::Softmax`], which is applied layer-wide.
    pub fn is_vector_wise(self) -> bool {
        matches!(self, Activation::Softmax)
    }

    /// Element-wise `f(z)`. Panics for softmax (use [`softmax`]).
    pub fn apply(self, z: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Activation::Tanh => z.tanh(),
            Activation::Relu => z.max(0.0),
            Activation::Softmax => panic!("softmax is vector-wise; use softmax()"),
        }
    }

    /// Element-wise `f'(z)`. Panics for softmax (handled with cross-entropy).
    pub fn derivative(self, z: f32) -> f32 {
        match self {
            Activation::Sigmoid => {
                let s = self.apply(z);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = z.tanh();
                1.0 - t * t
            }
            // f'(0) is undefined; use 0 by convention.
            Activation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Softmax => panic!("softmax derivative is handled with cross-entropy"),
        }
    }
}

/// Numerically stable softmax (subtract the max before exp).
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_midpoint_and_derivative() {
        assert_eq!(Activation::Sigmoid.apply(0.0), 0.5);
        assert!((Activation::Sigmoid.derivative(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn tanh_is_zero_centred() {
        assert!(Activation::Tanh.apply(0.0).abs() < 1e-6);
        assert!((Activation::Tanh.derivative(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn relu_clips_negatives() {
        assert_eq!(Activation::Relu.apply(-3.0), 0.0);
        assert_eq!(Activation::Relu.apply(2.5), 2.5);
        assert_eq!(Activation::Relu.derivative(-1.0), 0.0);
        assert_eq!(Activation::Relu.derivative(1.0), 1.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Larger score -> larger probability.
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn softmax_is_overflow_safe() {
        let p = softmax(&[1000.0, 1001.0, 1002.0]);
        assert!(p.iter().all(|x| x.is_finite()));
    }

    /// Cross-check every element-wise derivative against a numerical estimate.
    #[test]
    fn derivatives_match_finite_differences() {
        let h = 1e-3;
        for act in [Activation::Sigmoid, Activation::Tanh, Activation::Relu] {
            for &z in &[-2.0, -0.5, 0.5, 2.0] {
                let numeric = (act.apply(z + h) - act.apply(z - h)) / (2.0 * h);
                let analytic = act.derivative(z);
                assert!(
                    (numeric - analytic).abs() < 1e-2,
                    "{act:?} at z={z}: numeric={numeric}, analytic={analytic}"
                );
            }
        }
    }
}
