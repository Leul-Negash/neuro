//! Evaluation metrics (losses live in [`crate::loss`]).

use crate::network::Samples;

/// Index of the largest element (the predicted class).
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > values[best] { i } else { best })
}

/// Accuracy in `[0, 1]`: fraction of pairs whose argmax matches.
pub fn accuracy(pairs: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let correct = pairs
        .iter()
        .filter(|(pred, target)| argmax(pred) == argmax(target))
        .count();
    correct as f32 / pairs.len() as f32
}

/// A `classes × classes` confusion matrix indexed `[actual][predicted]`.
pub fn confusion_matrix(
    data: &Samples,
    classes: usize,
    predict: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; classes]; classes];
    for (input, target) in data {
        let actual = argmax(target);
        let predicted = argmax(&predict(input));
        matrix[actual][predicted] += 1;
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.5]), 0);
    }

    #[test]
    fn accuracy_counts_argmax_matches() {
        let pairs = vec![
            (vec![0.9, 0.1], vec![1.0, 0.0]), // correct
            (vec![0.2, 0.8], vec![1.0, 0.0]), // wrong
        ];
        assert_eq!(accuracy(&pairs), 0.5);
    }

    #[test]
    fn confusion_matrix_counts_correctly() {
        let data = vec![(vec![0.0], vec![1.0, 0.0]), (vec![1.0], vec![0.0, 1.0])];
        // Predict class 0 for everything.
        let m = confusion_matrix(&data, 2, |_| vec![1.0, 0.0]);
        assert_eq!(m[0][0], 1); // class 0 predicted 0
        assert_eq!(m[1][0], 1); // class 1 mispredicted as 0
    }
}
