//! Smoke test. `cargo run --release` trains three networks and prints a report
//! for each: XOR (with a gradient check), Iris (softmax + cross-entropy + early
//! stopping), and breast cancer (30 features, dropout + L2). Presentation only;
//! the learning lives in the library.

use neuro::data::{self, Dataset};
use neuro::loss::Loss;
use neuro::metrics::{argmax, confusion_matrix};
use neuro::optimizer::Optimizer;
use neuro::trainer::{Trainer, TrainingReport};
use neuro::{Activation, Network, Rng, Samples};

fn main() {
    banner("neuro  ·  a neural network built from scratch in pure Rust");
    run_xor();
    run_iris();
    run_breast_cancer();
    println!("\nAll demonstrations finished.\n");
}

fn run_xor() {
    section("DEMO 1 — XOR  ·  non-linearity + back-prop correctness");

    let mut rng = Rng::new(7);
    let data = data::xor();

    let mut net =
        Network::new(2)
            .add(4, Activation::Tanh, &mut rng)
            .add(1, Activation::Sigmoid, &mut rng);

    describe(&net, "hidden = Tanh, output = Sigmoid");
    println!("Optimiser    : Adam(lr=0.05), MSE loss, 800 epochs, batch 4\n");

    // Gradient check before training.
    let grad_error = net.gradient_check(&[1.0, 0.0], &[1.0], Loss::Mse);
    println!(
        "Gradient check : max relative error {:.2e}  →  {}",
        grad_error,
        if grad_error < 1e-3 {
            "back-prop is CORRECT ✓"
        } else {
            "MISMATCH ✗"
        }
    );

    let report = Trainer::new(Optimizer::adam(0.05), Loss::Mse)
        .epochs(800)
        .batch_size(4)
        .train(&mut net, &data, None, &mut rng);
    print_loss_curve(&report);

    println!("\nLearned truth table:");
    println!("  ┌─────┬─────┬──────────┬────────┬────────┐");
    println!("  │  A  │  B  │ expected │  pred  │ result │");
    println!("  ├─────┼─────┼──────────┼────────┼────────┤");
    let mut all_correct = true;
    for (input, target) in &data {
        let prediction = net.predict(input)[0];
        let correct = (prediction >= 0.5) as u8 == target[0] as u8;
        all_correct &= correct;
        println!(
            "  │  {}  │  {}  │    {}     │ {:>5.3}  │   {}   │",
            input[0] as u8,
            input[1] as u8,
            target[0] as u8,
            prediction,
            mark(correct)
        );
    }
    println!("  └─────┴─────┴──────────┴────────┴────────┘");
    println!("Verdict      : {}", verdict(all_correct));
}

fn run_iris() {
    section("DEMO 2 — Iris  ·  softmax + cross-entropy + early stopping");

    let Some(dataset) = load("data/iris.csv") else {
        return;
    };
    print_dataset_summary(&dataset, "data/iris.csv");

    let (train, val, test) = prepare_splits(&dataset, 2024);

    let mut rng = Rng::new(2024);
    let mut net = Network::new(dataset.num_features())
        .add(10, Activation::Relu, &mut rng)
        .add(dataset.num_classes(), Activation::Softmax, &mut rng);

    describe(&net, "hidden = ReLU, output = Softmax");
    println!("Optimiser    : Adam(lr=0.02), cross-entropy, ≤1500 epochs, batch 16, early-stop(40)");
    println!(
        "Split        : {} train / {} val / {} test\n",
        train.len(),
        val.len(),
        test.len()
    );

    let report = Trainer::new(Optimizer::adam(0.02), Loss::CrossEntropy)
        .epochs(1500)
        .batch_size(16)
        .early_stopping(40)
        .train(&mut net, &train, Some(&val), &mut rng);

    print_loss_curve(&report);
    print_early_stop(&report);
    print_classification_results(&net, &train, &test, &dataset, Loss::CrossEntropy);
}

fn run_breast_cancer() {
    section("DEMO 3 — Breast cancer  ·  30 features, dropout + L2 weight decay");

    let Some(dataset) = load("data/breast_cancer.csv") else {
        return;
    };
    print_dataset_summary(&dataset, "data/breast_cancer.csv");

    let (train, val, test) = prepare_splits(&dataset, 99);

    let mut rng = Rng::new(99);
    let mut net = Network::new(dataset.num_features())
        .add(16, Activation::Relu, &mut rng)
        .add(8, Activation::Relu, &mut rng)
        .add(dataset.num_classes(), Activation::Softmax, &mut rng);

    describe(&net, "hidden = ReLU×2, output = Softmax");
    println!("Regularised  : Adam(lr=0.01), cross-entropy, dropout 0.2, L2 1e-4, early-stop(30)");
    println!(
        "Split        : {} train / {} val / {} test\n",
        train.len(),
        val.len(),
        test.len()
    );

    let report = Trainer::new(Optimizer::adam(0.01), Loss::CrossEntropy)
        .epochs(2000)
        .batch_size(32)
        .dropout(0.2)
        .l2(1e-4)
        .early_stopping(30)
        .train(&mut net, &train, Some(&val), &mut rng);

    print_loss_curve(&report);
    print_early_stop(&report);
    print_classification_results(&net, &train, &test, &dataset, Loss::CrossEntropy);
}

/// Loads a CSV dataset, printing a message on failure.
fn load(path: &str) -> Option<Dataset> {
    match data::load_csv(path) {
        Ok(ds) => Some(ds),
        Err(e) => {
            println!("Could not load {path}: {e}");
            println!("(Run from the crate root so the data/ folder is found.)");
            None
        }
    }
}

/// Normalises, shuffles, and splits a dataset 60/20/20 into train/val/test.
fn prepare_splits(dataset: &Dataset, seed: u32) -> (Samples, Samples, Samples) {
    let mut samples = dataset.samples.clone();
    data::normalize(&mut samples);

    let mut rng = Rng::new(seed);
    let (train, rest) = data::train_test_split(samples, 0.6, &mut rng);
    let (val, test) = data::train_test_split(rest, 0.5, &mut rng);
    (train, val, test)
}

/// Prints architecture and parameter count.
fn describe(net: &Network, activations: &str) {
    println!(
        "Architecture : {}   ({} trainable parameters)",
        net.layer_sizes()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" → "),
        net.parameter_count()
    );
    println!("Activations  : {activations}");
}

/// Prints train/test accuracy, a confusion matrix, and sample predictions.
fn print_classification_results(
    net: &Network,
    train: &Samples,
    test: &Samples,
    dataset: &Dataset,
    loss: Loss,
) {
    let (train_loss, train_acc) = net.evaluate(train, loss);
    let (test_loss, test_acc) = net.evaluate(test, loss);
    println!("\nResults:");
    println!(
        "  train : accuracy {:>6.2}%   loss {:.4}",
        train_acc * 100.0,
        train_loss
    );
    println!(
        "  test  : accuracy {:>6.2}%   loss {:.4}",
        test_acc * 100.0,
        test_loss
    );

    print_confusion_matrix(net, test, dataset);
    print_sample_predictions(net, test, dataset, 5);
}

fn banner(title: &str) {
    let line = "═".repeat(title.chars().count() + 4);
    println!("\n╔{line}╗");
    println!("║  {title}  ║");
    println!("╚{line}╝");
}

fn section(title: &str) {
    println!("\n──────────────────────────────────────────────────────────────");
    println!("  {title}");
    println!("──────────────────────────────────────────────────────────────");
}

fn mark(correct: bool) -> &'static str {
    if correct { "✓" } else { "✗" }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS — all rows correct ✓"
    } else {
        "FAIL — some rows wrong ✗"
    }
}

/// Renders the training (and, if present, validation) loss as a sparkline.
fn print_loss_curve(report: &TrainingReport) {
    print_sparkline("Train loss   ", &report.loss_history);
    if !report.val_loss_history.is_empty() {
        print_sparkline("Val loss     ", &report.val_loss_history);
    }
}

/// Prints one labelled unicode sparkline on a log scale.
fn print_sparkline(label: &str, history: &[f32]) {
    if history.is_empty() {
        return;
    }
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    let step = (history.len() / 60).max(1);
    let sampled: Vec<f32> = history.iter().step_by(step).copied().collect();

    // Log scale keeps the slow tail of the descent visible.
    let logs: Vec<f32> = sampled.iter().map(|&v| (v.max(1e-9)).ln()).collect();
    let min = logs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-9);

    let spark: String = logs
        .iter()
        .map(|&v| BARS[(((v - min) / range) * 7.0).round() as usize])
        .collect();

    println!(
        "{label}: {spark}  ({:.4} → {:.4})",
        history[0],
        history[history.len() - 1]
    );
}

/// Reports the outcome of early stopping.
fn print_early_stop(report: &TrainingReport) {
    if report.val_loss_history.is_empty() {
        return;
    }
    let trained = report.loss_history.len();
    if report.stopped_early {
        println!(
            "Early stop   : after {trained} epochs; best weights from epoch {} restored",
            report.best_epoch + 1
        );
    } else {
        println!(
            "Early stop   : ran full schedule; best epoch {}",
            report.best_epoch + 1
        );
    }
}

fn print_dataset_summary(dataset: &Dataset, path: &str) {
    println!("Source       : {path}  ({} samples)", dataset.samples.len());
    println!("Features     : {} numeric columns", dataset.num_features());
    let parts: Vec<String> = dataset
        .class_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let count = dataset
                .samples
                .iter()
                .filter(|(_, t)| argmax(t) == i)
                .count();
            format!("{} ({count})", short_label(name))
        })
        .collect();
    println!("Classes      : {}", parts.join(", "));
}

fn print_confusion_matrix(net: &Network, test: &Samples, dataset: &Dataset) {
    let n = dataset.num_classes();
    let matrix = confusion_matrix(test, n, |input| net.predict(input));
    let labels: Vec<String> = dataset.class_names.iter().map(|c| short_label(c)).collect();

    let w = labels.iter().map(String::len).max().unwrap_or(6) + 2;
    let label_w = w - 2;

    println!("\nConfusion matrix (rows = actual, cols = predicted):");
    print!("  {:<label_w$}", "");
    for label in &labels {
        print!("{label:>w$}");
    }
    println!();
    for (i, row) in matrix.iter().enumerate() {
        print!("  {:<label_w$}", labels[i]);
        for &count in row {
            print!("{count:>w$}");
        }
        println!();
    }
}

fn print_sample_predictions(net: &Network, test: &Samples, dataset: &Dataset, count: usize) {
    println!("\nSample predictions (first {count} test rows):");
    for (input, target) in test.iter().take(count) {
        let output = net.predict(input);
        let predicted = argmax(&output);
        let actual = argmax(target);
        let confidence = output[predicted] * 100.0;
        println!(
            "  {}  predicted {:<12} (conf {:>5.1}%)   actual {}",
            mark(predicted == actual),
            short_label(&dataset.class_names[predicted]),
            confidence,
            short_label(&dataset.class_names[actual])
        );
    }
}

/// Trims a long label like `Iris-versicolor` down to `versicolor`.
fn short_label(name: &str) -> String {
    name.rsplit('-').next().unwrap_or(name).to_string()
}
