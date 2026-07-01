//! Datasets and preprocessing: the [`xor`] set, a [`load_csv`] reader, plus
//! z-score normalisation and train/test splitting.
//!
//! [`load_csv`] expects a header row, numeric feature columns, and a single
//! categorical label in the last column (one-hot encoded).

use crate::network::Samples;
use crate::rng::Rng;
use std::fs;
use std::io;

/// A prepared dataset ready for a [`crate::Network`].
#[derive(Debug, Clone)]
pub struct Dataset {
    /// `(features, one_hot_target)` pairs.
    pub samples: Samples,
    /// Feature column names, in order.
    pub feature_names: Vec<String>,
    /// Class names, indexed to match one-hot positions.
    pub class_names: Vec<String>,
}

impl Dataset {
    /// Number of input features per sample.
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Number of distinct output classes.
    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }
}

/// The XOR truth table as a labelled dataset.
pub fn xor() -> Samples {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

/// Loads a CSV classification dataset (header, numeric features, label last).
/// The label is one-hot encoded by order of first appearance.
pub fn load_csv(path: &str) -> io::Result<Dataset> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let header = lines.next().ok_or_else(|| invalid("CSV file is empty"))?;
    let columns: Vec<&str> = header.split(',').collect();
    if columns.len() < 2 {
        return Err(invalid(
            "CSV needs at least one feature and one label column",
        ));
    }
    let feature_names: Vec<String> = columns[..columns.len() - 1]
        .iter()
        .map(|s| s.trim().to_string())
        .collect();

    // First pass: parse features and collect raw string labels.
    let mut raw_features: Vec<Vec<f32>> = Vec::new();
    let mut raw_labels: Vec<String> = Vec::new();
    let mut class_names: Vec<String> = Vec::new();

    for (row, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != columns.len() {
            return Err(invalid(&format!(
                "row {} has {} columns, expected {}",
                row + 2,
                cells.len(),
                columns.len()
            )));
        }

        let mut features = Vec::with_capacity(feature_names.len());
        for cell in &cells[..cells.len() - 1] {
            let value: f32 = cell
                .trim()
                .parse()
                .map_err(|_| invalid(&format!("could not parse '{}' as a number", cell.trim())))?;
            features.push(value);
        }
        raw_features.push(features);

        let label = cells[cells.len() - 1].trim().to_string();
        if !class_names.contains(&label) {
            class_names.push(label.clone());
        }
        raw_labels.push(label);
    }

    // Second pass: one-hot encode labels now that all classes are known.
    let samples = raw_features
        .into_iter()
        .zip(raw_labels)
        .map(|(features, label)| {
            let mut target = vec![0.0; class_names.len()];
            let class = class_names.iter().position(|c| c == &label).unwrap();
            target[class] = 1.0;
            (features, target)
        })
        .collect();

    Ok(Dataset {
        samples,
        feature_names,
        class_names,
    })
}

/// Z-score normalises each feature column to zero mean and unit variance.
pub fn normalize(samples: &mut Samples) {
    if samples.is_empty() {
        return;
    }
    let num_features = samples[0].0.len();
    let count = samples.len() as f32;

    for f in 0..num_features {
        let mean: f32 = samples.iter().map(|(x, _)| x[f]).sum::<f32>() / count;
        let variance: f32 = samples
            .iter()
            .map(|(x, _)| (x[f] - mean).powi(2))
            .sum::<f32>()
            / count;
        let std = variance.sqrt().max(1e-8); // guard against constant columns

        for (x, _) in samples.iter_mut() {
            x[f] = (x[f] - mean) / std;
        }
    }
}

/// Shuffles, then splits into `(train, test)` by `train_ratio`.
pub fn train_test_split(
    mut samples: Samples,
    train_ratio: f32,
    rng: &mut Rng,
) -> (Samples, Samples) {
    rng.shuffle(&mut samples);
    let train_len = (samples.len() as f32 * train_ratio).round() as usize;
    let test = samples.split_off(train_len);
    (samples, test)
}

/// Builds an `InvalidData` IO error with a message.
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_has_four_rows() {
        assert_eq!(xor().len(), 4);
    }

    #[test]
    fn normalize_yields_zero_mean() {
        let mut samples = vec![
            (vec![1.0], vec![0.0]),
            (vec![3.0], vec![0.0]),
            (vec![5.0], vec![0.0]),
        ];
        normalize(&mut samples);
        let mean: f32 = samples.iter().map(|(x, _)| x[0]).sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn split_partitions_without_loss() {
        let mut rng = Rng::new(1);
        let samples: Samples = (0..10).map(|i| (vec![i as f32], vec![0.0])).collect();
        let (train, test) = train_test_split(samples, 0.7, &mut rng);
        assert_eq!(train.len(), 7);
        assert_eq!(test.len(), 3);
    }

    #[test]
    fn loads_bundled_iris() {
        // Runs from the crate root, where data/ lives.
        let ds = load_csv("data/iris.csv").expect("iris.csv should load");
        assert_eq!(ds.num_features(), 4);
        assert_eq!(ds.num_classes(), 3);
        assert_eq!(ds.samples.len(), 150);
    }
}
