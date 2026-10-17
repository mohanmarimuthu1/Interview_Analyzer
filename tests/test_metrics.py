import numpy as np

from interviewlens.eval.metrics import accuracy, confusion_matrix, macro_f1


def test_confusion_and_scores():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert accuracy(cm) == 4 / 6
    # per-class f1: 0.5, 0.8, 0.667
    assert abs(macro_f1(cm) - (0.5 + 0.8 + 2 / 3) / 3) < 1e-9


def test_perfect_and_missing_class():
    y = np.array([0, 1, 1])
    cm = confusion_matrix(y, y, 3)  # class 2 never appears
    assert accuracy(cm) == 1.0
    assert abs(macro_f1(cm) - 2 / 3) < 1e-9
