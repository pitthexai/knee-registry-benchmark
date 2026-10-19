import numpy as np
import pytest
import pandas as pd

from orthobench.conformal import qhat, mondrian_qhats, class_conditional_set
from orthobench.statistics import grouped_bootstrap_ci, paired_grouped_bootstrap_delta
from orthobench.triage import triage_report


def test_blank_paths_are_not_duplicate_images():
    df = pd.DataFrame({"png_path": ["/valid/a.png", None, None], "png_ok": [1, 0, 0]})
    check_rows = df[df["png_ok"].eq(1)]
    paths = check_rows["png_path"].dropna().astype(str).str.strip()
    assert not paths[paths.ne("")].duplicated().any()


def test_finite_sample_quantile():
    assert qhat(np.arange(10), alpha=.2) == 8


def test_class_conditional_threshold_applies_to_candidate_label():
    p = np.array([[.55, .35, .10], [.20, .70, .10]])
    y = np.array([0, 1])
    q = mondrian_qhats(p, y, 3, .1)
    assert class_conditional_set(p[0], q) == [0]
    assert class_conditional_set(p[1], q) == [1]


def test_grouped_bootstrap_and_paired_delta():
    y = np.array([0, 0, 1, 1]); g = np.array(["a", "a", "b", "b"])
    a = y.copy(); b = 1 - y
    ci = grouped_bootstrap_ci(y, a, g, n_boot=50)
    d = paired_grouped_bootstrap_delta(y, a, b, g, n_boot=50)
    assert ci["estimate"] == 1.0 and ci["n_groups"] == 2
    assert d["delta"] == 1.0


def test_triage_threshold_order_and_safety_names():
    with pytest.raises(ValueError):
        triage_report(np.array([.2]), np.array([1]), .8, .7)
    out = triage_report(np.array([.1, .9]), np.array([1, 1]), .2, .8)
    assert out["knee_false_rejection_rate"] == .5
