//! End-to-end tests through the public API.

use neuro::data;
use neuro::loss::Loss;
use neuro::optimizer::Optimizer;
use neuro::trainer::Trainer;
use neuro::{Activation, Network, Rng};

/// A 2-4-1 network trained with Adam must master XOR.
#[test]
fn network_learns_xor_end_to_end() {
    let mut rng = Rng::new(7);
    let data = data::xor();

    let mut net =
        Network::new(2)
            .add(4, Activation::Tanh, &mut rng)
            .add(1, Activation::Sigmoid, &mut rng);

    let report = Trainer::new(Optimizer::adam(0.05), Loss::Mse)
        .epochs(800)
        .batch_size(4)
        .train(&mut net, &data, None, &mut rng);

    let first = report.loss_history.first().copied().unwrap();
    let last = report.loss_history.last().copied().unwrap();
    assert!(
        last < first * 0.1,
        "loss barely improved: {first} -> {last}"
    );

    for (input, target) in &data {
        let pred = net.predict(input)[0];
        assert_eq!(
            (pred >= 0.5) as u8,
            target[0] as u8,
            "wrong class for {input:?}"
        );
    }
}

/// Back-prop must match finite differences end-to-end through the public API.
#[test]
fn gradient_check_holds_through_public_api() {
    let mut rng = Rng::new(11);
    let net =
        Network::new(4)
            .add(6, Activation::Tanh, &mut rng)
            .add(3, Activation::Softmax, &mut rng);

    let err = net.gradient_check(&[0.3, -0.7, 0.5, 0.1], &[0.0, 0.0, 1.0], Loss::CrossEntropy);
    assert!(err < 1e-2, "gradient check failed: {err}");
}

/// Softmax + cross-entropy + Adam + early stopping on real Iris data.
#[test]
fn classifies_iris_above_90_percent() {
    let dataset = data::load_csv("data/iris.csv").expect("iris.csv should load");

    let mut samples = dataset.samples.clone();
    data::normalize(&mut samples);

    let mut rng = Rng::new(2024);
    let (train, rest) = data::train_test_split(samples, 0.6, &mut rng);
    let (val, test) = data::train_test_split(rest, 0.5, &mut rng);

    let mut net = Network::new(dataset.num_features())
        .add(10, Activation::Relu, &mut rng)
        .add(dataset.num_classes(), Activation::Softmax, &mut rng);

    Trainer::new(Optimizer::adam(0.02), Loss::CrossEntropy)
        .epochs(1500)
        .batch_size(16)
        .early_stopping(40)
        .train(&mut net, &train, Some(&val), &mut rng);

    let (_, test_acc) = net.evaluate(&test, Loss::CrossEntropy);
    assert!(test_acc > 0.90, "iris test accuracy too low: {test_acc}");
}

/// The harder 30-feature breast-cancer dataset, regularised, must classify well.
#[test]
fn classifies_breast_cancer_above_90_percent() {
    let dataset = data::load_csv("data/breast_cancer.csv").expect("breast_cancer.csv should load");
    assert_eq!(dataset.num_features(), 30);
    assert_eq!(dataset.num_classes(), 2);

    let mut samples = dataset.samples.clone();
    data::normalize(&mut samples);

    let mut rng = Rng::new(99);
    let (train, rest) = data::train_test_split(samples, 0.6, &mut rng);
    let (val, test) = data::train_test_split(rest, 0.5, &mut rng);

    let mut net = Network::new(dataset.num_features())
        .add(16, Activation::Relu, &mut rng)
        .add(8, Activation::Relu, &mut rng)
        .add(dataset.num_classes(), Activation::Softmax, &mut rng);

    Trainer::new(Optimizer::adam(0.01), Loss::CrossEntropy)
        .epochs(2000)
        .batch_size(32)
        .dropout(0.2)
        .l2(1e-4)
        .early_stopping(30)
        .train(&mut net, &train, Some(&val), &mut rng);

    let (_, test_acc) = net.evaluate(&test, Loss::CrossEntropy);
    assert!(
        test_acc > 0.90,
        "breast-cancer test accuracy too low: {test_acc}"
    );
}
