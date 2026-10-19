import numpy as np

from orthobench.sensitivity import paired_mask_metrics


def test_paired_mask_metrics_counts_directional_changes():
    y = np.array([0, 1, 0, 1])
    pred_none = np.array([0, 1, 1, 1])
    pred_mask = np.array([1, 1, 0, 1])
    prob_none = np.array([[.8, .2], [.1, .9], [.4, .6], [.2, .8]])
    prob_mask = np.array([[.3, .7], [.2, .8], [.7, .3], [.1, .9]])
    result = paired_mask_metrics(y, pred_none, prob_none, pred_mask, prob_mask)
    assert result["prediction_flip_rate"] == 0.5
    assert result["correct_to_incorrect_n"] == 1
    assert result["incorrect_to_correct_n"] == 1
    assert result["delta_accuracy_vs_none"] == 0.0
    assert result["mcnemar_exact_p"] == 1.0


def test_no_prediction_changes_has_mcnemar_p_one():
    y = np.array([0, 1])
    pred = np.array([0, 1])
    prob = np.array([[.9, .1], [.1, .9]])
    result = paired_mask_metrics(y, pred, prob, pred, prob)
    assert result["prediction_flip_rate"] == 0.0
    assert result["mcnemar_exact_p"] == 1.0
