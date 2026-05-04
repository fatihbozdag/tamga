"""sklearn classifier wrappers + CV helper with stylometry-aware splits."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.svm import SVC

from bitig.features import FeatureMatrix

_ESTIMATORS = {
    "logreg": lambda **kw: LogisticRegression(max_iter=2000, **kw),
    "svm_linear": lambda **kw: SVC(kernel="linear", probability=True, **kw),
    "svm_rbf": lambda **kw: SVC(kernel="rbf", probability=True, **kw),
    "rf": lambda **kw: RandomForestClassifier(**kw),
    "hgbm": lambda **kw: HistGradientBoostingClassifier(**kw),
}


def build_classifier(name: str, **kwargs: Any) -> BaseEstimator:
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown classifier {name!r}; known: {sorted(_ESTIMATORS)}")
    return _ESTIMATORS[name](**kwargs)


def cross_validate_bitig(
    estimator: BaseEstimator,
    fm: FeatureMatrix,
    y: np.ndarray,
    *,
    cv_kind: str = "stratified",
    groups_from: np.ndarray | None = None,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run cross-validation with a stylometry-aware CV strategy.

    cv_kind:
      - "stratified": StratifiedKFold(folds) — uses `seed` for the shuffle
      - "loao":       LeaveOneGroupOut (requires groups_from; deterministic)
      - "leave_one_text_out": LeaveOneOut (deterministic)
    """
    if cv_kind == "stratified":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        groups = None
    elif cv_kind == "loao":
        if groups_from is None:
            raise ValueError("cv_kind='loao' requires groups_from")
        cv = LeaveOneGroupOut()
        groups = np.asarray(groups_from)
    elif cv_kind == "leave_one_text_out":
        cv = LeaveOneOut()
        groups = None
    else:
        raise ValueError(f"unknown cv_kind {cv_kind!r}")

    preds = cross_val_predict(estimator, fm.X, y, cv=cv, groups=groups)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    proba: np.ndarray | None = None
    classes: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        try:
            proba = cross_val_predict(
                estimator, fm.X, y, cv=cv, groups=groups, method="predict_proba"
            )
            # cross_val_predict reorders class columns to match np.unique(y); recover.
            classes = np.unique(y)
        except Exception:
            # Some estimators raise when predict_proba is unsupported under a given CV fold
            # configuration -- skip silently and let downstream calibration code handle the
            # missing-proba case.
            proba = None
            classes = None

    return {
        "accuracy": float((preds == y).mean()),
        "predictions": preds,
        "per_class": report,
        "proba": proba,
        "classes": classes,
    }
