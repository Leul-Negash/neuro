//! Loss functions and the output-layer error they produce for back-prop.
//!
//! MSE is general-purpose; cross-entropy is for classification and, with a
//! softmax/sigmoid output, its gradient reduces to `output - target`.

/// A differentiable training objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// `(1/n) · Σ (pred − target)²`.
    Mse,
    /// `−Σ target · ln(pred)`.
    CrossEntropy,
}

impl Loss {
    /// Scalar loss for one prediction/target pair.
    pub fn value(self, prediction: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(prediction.len(), target.len());
        match self {
            Loss::Mse => {
                let sum: f32 = prediction
                    .iter()
                    .zip(target)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();
                sum / prediction.len() as f32
            }
            Loss::CrossEntropy => {
                // Clamp to avoid ln(0).
                -prediction
                    .iter()
                    .zip(target)
                    .map(|(p, t)| t * p.clamp(1e-7, 1.0).ln())
                    .sum::<f32>()
            }
        }
    }

    /// Output-layer error `∂L/∂z` that seeds back-prop.
    ///
    /// MSE needs the output activation derivative (passed in by index);
    /// cross-entropy reduces to `output - target` and ignores it.
    pub fn output_delta(
        self,
        output: &[f32],
        target: &[f32],
        activation_derivative: impl Fn(usize) -> f32,
    ) -> Vec<f32> {
        match self {
            Loss::Mse => {
                let n = output.len() as f32;
                output
                    .iter()
                    .zip(target)
                    .enumerate()
                    .map(|(i, (a, t))| (2.0 / n) * (a - t) * activation_derivative(i))
                    .collect()
            }
            Loss::CrossEntropy => output.iter().zip(target).map(|(a, t)| a - t).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_zero_for_perfect_prediction() {
        assert_eq!(Loss::Mse.value(&[1.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn cross_entropy_lower_when_confident_correct() {
        let confident = Loss::CrossEntropy.value(&[0.9, 0.1], &[1.0, 0.0]);
        let unsure = Loss::CrossEntropy.value(&[0.5, 0.5], &[1.0, 0.0]);
        assert!(confident < unsure);
    }

    #[test]
    fn cross_entropy_delta_is_output_minus_target() {
        let delta = Loss::CrossEntropy.output_delta(&[0.7, 0.3], &[1.0, 0.0], |_| 1.0);
        assert!((delta[0] - (-0.3)).abs() < 1e-6);
        assert!((delta[1] - 0.3).abs() < 1e-6);
    }
}
