"""Train a robust CatBoost ensemble for Kaggle Playground S6E9."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LOGGER = logging.getLogger(__name__)
TARGET = "Will_Buy_EV"
DEFAULT_SEEDS = (42, 2026, 31415)


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.glob(f"**/{filename}"), key=lambda path: (len(path.parts), str(path)))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    return matches[0]


def prepare_features(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    result = frame.loc[:, list(feature_columns)].copy()
    for column in result.columns:
        if pd.api.types.is_bool_dtype(result[column]):
            result[column] = result[column].astype("int8")
        elif pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
        elif isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].astype(str).replace("nan", "__MISSING__")
    return result


def positive_class_index(target: pd.Series, classes: np.ndarray) -> int:
    if target.dtype == bool:
        positive = True
    elif pd.api.types.is_numeric_dtype(target):
        positive = 1
    else:
        positive = "1" if "1" in {str(value) for value in classes} else sorted(classes.astype(str))[-1]
    for index, value in enumerate(classes):
        if value == positive or str(value) == str(positive):
            return index
    raise ValueError(f"Unable to identify positive class in {classes!r}")


def train_ensemble(train: pd.DataFrame, test: pd.DataFrame, target: str, id_column: str, seeds: tuple[int, ...], folds: int):
    feature_columns = [column for column in train.columns if column not in {target, id_column}]
    if not feature_columns:
        raise ValueError("No feature columns remain after removing target and identifier")
    x_train = prepare_features(train, feature_columns)
    x_test = prepare_features(test, feature_columns)
    categorical_columns = [index for index, column in enumerate(x_train.columns) if x_train[column].dtype == object]
    y = train[target]
    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_scores = []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (fit_indices, valid_indices) in enumerate(splitter.split(x_train, y), start=1):
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="AUC", iterations=4000,
                learning_rate=0.03, depth=8, l2_leaf_reg=8.0,
                random_strength=0.5, bagging_temperature=0.5, border_count=254,
                random_seed=seed + fold, thread_count=-1,
                allow_writing_files=False, verbose=False,
            )
            model.fit(
                x_train.iloc[fit_indices], y.iloc[fit_indices], cat_features=categorical_columns,
                eval_set=(x_train.iloc[valid_indices], y.iloc[valid_indices]),
                use_best_model=True, early_stopping_rounds=200, verbose=False,
            )
            positive_index = positive_class_index(y, np.asarray(model.classes_))
            valid_prediction = model.predict_proba(x_train.iloc[valid_indices])[:, positive_index]
            test_prediction = model.predict_proba(x_test)[:, positive_index]
            oof[valid_indices] += valid_prediction / len(seeds)
            test_predictions += test_prediction / (len(seeds) * folds)
            score = roc_auc_score(y.iloc[valid_indices], valid_prediction)
            fold_scores.append(score)
            LOGGER.info("seed=%d fold=%d best_iteration=%d valid_auc=%.6f", seed, fold, model.get_best_iteration(), score)
    overall_score = roc_auc_score(y, oof)
    LOGGER.info("OOF ROC AUC: %.6f | mean fold ROC AUC: %.6f", overall_score, float(np.mean(fold_scores)))
    return oof, test_predictions, overall_score


def build_submission(test: pd.DataFrame, sample: pd.DataFrame | None, predictions: np.ndarray, id_column: str) -> pd.DataFrame:
    if sample is not None:
        if len(sample) != len(predictions):
            raise ValueError("sample_submission.csv and test.csv have different row counts")
        submission = sample.copy()
        if TARGET not in submission.columns:
            raise ValueError(f"sample_submission.csv must contain {TARGET}")
        submission[TARGET] = predictions
        return submission
    return pd.DataFrame({id_column: test[id_column].to_numpy(), TARGET: predictions})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing train.csv and test.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    search_root = args.data_dir or (Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("data"))
    train_path = find_file(search_root, "train.csv")
    test_path = find_file(search_root, "test.csv")
    sample_path = next(iter(sorted(search_root.glob("**/sample_submission.csv"))), None)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if sample_path else None
    if TARGET not in train:
        raise ValueError(f"Training data must contain target column {TARGET}")
    id_column = "id" if "id" in test.columns else (sample.columns[0] if sample is not None else test.columns[0])
    if id_column not in train or id_column not in test:
        raise ValueError(f"Identifier column {id_column!r} must be present in train and test")
    if train[TARGET].nunique(dropna=False) != 2:
        raise ValueError(f"{TARGET} must be binary for ROC AUC, found {train[TARGET].unique()!r}")
    LOGGER.info("train=%s test=%s features=%d id=%s", train.shape, test.shape, train.shape[1] - 2, id_column)
    _, predictions, score = train_ensemble(train, test, TARGET, id_column, tuple(args.seeds), args.folds)
    submission = build_submission(test, sample, predictions, id_column)
    if list(submission.columns) != [id_column, TARGET]:
        submission = submission[[id_column, TARGET]]
    if submission[id_column].duplicated().any() or submission[TARGET].isna().any():
        raise ValueError("Submission contains duplicate IDs or missing predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    diagnostics = args.output.with_name(f"{args.output.stem}_diagnostics.json")
    diagnostics.write_text(json.dumps({"oof_roc_auc": score, "rows": len(submission), "features": train.shape[1] - 2}, indent=2))
    LOGGER.info("wrote %s (%d rows)", args.output.resolve(), len(submission))


if __name__ == "__main__":
    main()