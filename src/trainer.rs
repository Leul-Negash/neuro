//! Training loop: mini-batch gradient descent with a pluggable optimiser, L2
//! weight decay, dropout, and early stopping. Configured via chained setters.
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
//! let data = neuro::data::xor();
//! let report = Trainer::new(Optimizer::adam(0.05), Loss::Mse)
//!     .epochs(400)
//!     .batch_size(4)
//!     .train(&mut net, &data, None, &mut rng);
//!
//! assert!(*report.loss_history.last().unwrap() < 0.05);
//! ```

use crate::layer::Gradients;
use crate::loss::Loss;
use crate::network::{Network, Samples};
use crate::optimizer::{Optimizer, OptimizerState};
use crate::rng::Rng;

/// Outcome of a training run.
#[derive(Debug, Clone)]
pub struct TrainingReport {
    /// Training loss per epoch.
    pub loss_history: Vec<f32>,
    /// Validation loss per epoch (empty if no validation set).
    pub val_loss_history: Vec<f32>,
    /// Epoch with the best validation loss.
    pub best_epoch: usize,
    /// Whether early stopping triggered.
    pub stopped_early: bool,
}

/// Configurable training procedure.
#[derive(Debug, Clone)]
pub struct Trainer {
    optimizer: Optimizer,
    loss: Loss,
    epochs: usize,
    batch_size: usize,
    l2: f32,
    dropout: f32,
    patience: Option<usize>,
}

impl Trainer {
    /// Defaults: 100 epochs, batch 32, no regularisation/dropout/early stopping.
    pub fn new(optimizer: Optimizer, loss: Loss) -> Self {
        Trainer {
            optimizer,
            loss,
            epochs: 100,
            batch_size: 32,
            l2: 0.0,
            dropout: 0.0,
            patience: None,
        }
    }

    /// Number of passes over the training set.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Mini-batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// L2 weight-decay coefficient (`0.0` = off).
    pub fn l2(mut self, lambda: f32) -> Self {
        self.l2 = lambda;
        self
    }

    /// Hidden-layer dropout rate (`0.0` = off).
    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout = rate.clamp(0.0, 0.9);
        self
    }

    /// Stop after `patience` epochs without validation improvement and restore
    /// the best weights.
    pub fn early_stopping(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }

    /// Trains `net`, optionally using `validation` for early stopping.
    pub fn train(
        &self,
        net: &mut Network,
        train: &Samples,
        validation: Option<&Samples>,
        rng: &mut Rng,
    ) -> TrainingReport {
        let mut state = OptimizerState::new(self.optimizer, &net.layers);
        let mut order: Vec<usize> = (0..train.len()).collect();

        let mut report = TrainingReport {
            loss_history: Vec::with_capacity(self.epochs),
            val_loss_history: Vec::new(),
            best_epoch: 0,
            stopped_early: false,
        };

        let mut best_val = f32::INFINITY;
        let mut best_net = net.clone();
        let mut stale_epochs = 0;

        for epoch in 0..self.epochs {
            rng.shuffle(&mut order);

            for batch in order.chunks(self.batch_size) {
                let grads = self.batch_gradients(net, train, batch, rng);
                state.step(&mut net.layers, &grads);
            }

            report.loss_history.push(net.evaluate(train, self.loss).0);

            // Validation tracking + early stopping.
            if let Some(val) = validation {
                let val_loss = net.evaluate(val, self.loss).0;
                report.val_loss_history.push(val_loss);

                if val_loss < best_val - 1e-6 {
                    best_val = val_loss;
                    best_net = net.clone();
                    report.best_epoch = epoch;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                    if self.patience.is_some_and(|p| stale_epochs >= p) {
                        report.stopped_early = true;
                        break;
                    }
                }
            }
        }

        // Restore the best validation weights when early stopping is in play.
        if self.patience.is_some() && validation.is_some() {
            *net = best_net;
        }

        report
    }

    /// Computes the averaged, regularised gradient for one mini-batch.
    fn batch_gradients(
        &self,
        net: &Network,
        train: &Samples,
        batch: &[usize],
        rng: &mut Rng,
    ) -> Vec<Gradients> {
        let mut accumulated: Vec<Gradients> = net
            .layers
            .iter()
            .map(|l| Gradients::zeros(l.neurons(), l.inputs()))
            .collect();

        for &i in batch {
            let (input, target) = &train[i];
            let trace = net.forward_train(input, self.dropout, rng);
            let sample_grads = net.backprop(&trace, target, self.loss);
            for (acc, g) in accumulated.iter_mut().zip(&sample_grads) {
                acc.add(g);
            }
        }

        let scale = 1.0 / batch.len() as f32;
        for acc in &mut accumulated {
            acc.scale(scale);
        }

        // L2 weight decay: add λ·w to each weight gradient (not to biases).
        if self.l2 > 0.0 {
            for (acc, layer) in accumulated.iter_mut().zip(&net.layers) {
                for (grow, wrow) in acc.weights.iter_mut().zip(&layer.weights) {
                    for (g, w) in grow.iter_mut().zip(wrow) {
                        *g += self.l2 * *w;
                    }
                }
            }
        }

        accumulated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::data;

    #[test]
    fn adam_learns_xor_fast() {
        let mut rng = Rng::new(7);
        let mut net = Network::new(2).add(4, Activation::Tanh, &mut rng).add(
            1,
            Activation::Sigmoid,
            &mut rng,
        );

        let report = Trainer::new(Optimizer::adam(0.05), Loss::Mse)
            .epochs(500)
            .batch_size(4)
            .train(&mut net, &data::xor(), None, &mut rng);

        assert!(*report.loss_history.last().unwrap() < 0.02);
        for (input, target) in &data::xor() {
            assert_eq!((net.predict(input)[0] >= 0.5) as u8, target[0] as u8);
        }
    }

    #[test]
    fn early_stopping_records_best_epoch() {
        let mut rng = Rng::new(3);
        let data = data::xor();
        let mut net = Network::new(2).add(4, Activation::Tanh, &mut rng).add(
            1,
            Activation::Sigmoid,
            &mut rng,
        );

        let report = Trainer::new(Optimizer::adam(0.05), Loss::Mse)
            .epochs(1000)
            .batch_size(4)
            .early_stopping(20)
            .train(&mut net, &data, Some(&data), &mut rng);

        assert!(!report.val_loss_history.is_empty());
        assert!(report.best_epoch < report.loss_history.len());
    }
}
