from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "training"
OUT_DIR = ROOT / "data" / "out"
EVAL_DIR = ROOT / "eval"
MODEL_PATH = OUT_DIR / "question_selector.joblib"
TRAINING_OUTPUT_PATH = OUT_DIR / "training_data.csv"
OOF_OUTPUT_PATH = OUT_DIR / "training_oof_predictions.csv"
METRICS_PATH = OUT_DIR / "model_metrics.json"
RESULTS_PATH = EVAL_DIR / "results.md"

FEATURES = [
    "active",
    "knocked",
    "teams_remaining",
    "players_remaining",
    "recent_active_drop",
    "match_progress",
]

FORBIDDEN_FUTURE_FEATURE_TOKENS = {
    "future",
    "outcome",
    "winner",
    "correct",
    "survived",
    "window_end",
}


def confidence_weight(value: object) -> float:
    value = str(value).strip().lower()
    if value == "high":
        return 1.0
    if value == "medium":
        return 0.6
    if value == "low":
        return 0.3
    return 0.75


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data() -> pd.DataFrame:
    files = sorted(TRAINING_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data/training/")

    frames: list[pd.DataFrame] = []
    for path in files:
        print("Loading", path.name)
        frame = pd.read_csv(path)
        required = {"match_id", "time_s", "squad", "worth_asking", *FEATURES}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing training columns: {missing}")
        frame = frame[frame["worth_asking"].notna()].copy()
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if data["match_id"].astype(str).eq("ffws2025-g4").any():
        raise ValueError("Held-out Game 4 appears in the training directory")

    if any(any(token in feature.lower() for token in FORBIDDEN_FUTURE_FEATURE_TOKENS) for feature in FEATURES):
        raise ValueError("A configured model feature appears to contain future information")

    for column in ["time_s", *FEATURES, "worth_asking"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[[*FEATURES, "worth_asking"]].isna().any().any():
        raise ValueError("Training features and target cannot contain missing values")

    data["worth_asking"] = data["worth_asking"].astype(int)
    if not set(data["worth_asking"].unique()).issubset({0, 1}):
        raise ValueError("worth_asking must contain only 0/1 labels")
    if data["worth_asking"].nunique() != 2:
        raise ValueError("Training data requires both positive and negative examples")
    if (~data["active"].between(0, 4)).any() or (~data["knocked"].between(0, 4)).any():
        raise ValueError("Training player-state features must be between 0 and 4")

    duplicates = data.duplicated(["match_id", "time_s", "squad"], keep=False)
    if duplicates.any():
        duplicate_rows = data.loc[duplicates, ["match_id", "time_s", "squad"]].to_dict("records")
        raise ValueError(f"Duplicate historical squad states: {duplicate_rows}")

    return data.sort_values(["match_id", "time_s", "squad"], kind="stable").reset_index(drop=True)


def ranking_comparison(predictions: pd.DataFrame) -> dict:
    model_labels: list[int] = []
    naive_labels: list[int] = []

    for _, group in predictions.groupby(["match_id", "time_s"], sort=False):
        if group.empty:
            continue
        model_index = group["oof_worth_score"].astype(float).idxmax()
        naive_index = group.sort_values(
            ["active", "knocked", "squad"],
            ascending=[True, False, True],
            kind="stable",
        ).index[0]
        model_labels.append(int(predictions.loc[model_index, "worth_asking"]))
        naive_labels.append(int(predictions.loc[naive_index, "worth_asking"]))

    return {
        "timestamp_groups": len(model_labels),
        "model_top_positive_rate": round(float(np.mean(model_labels)), 4),
        "naive_fewest_active_positive_rate": round(float(np.mean(naive_labels)), 4),
        "model_lift_claimed": False,
        "important_note": (
            "Historical rows contain a sparse, human-preselected squad subset at each timestamp; "
            "this comparison is descriptive rather than a full 12-squad ranking evaluation, and "
            "does not establish model superiority over the naive rule."
        ),
    }


def render_results(metrics: dict) -> str:
    validation = metrics["validation"]
    ranking = metrics["ranking"]
    coefficients = metrics["standardized_coefficients"]
    coefficient_rows = "\n".join(
        f"| `{feature}` | {value:+.4f} |" for feature, value in coefficients.items()
    )
    fold_rows = "\n".join(
        f"| {fold['held_out_match']} | {fold['balanced_accuracy']:.3f} | "
        f"{fold['average_precision']:.3f} | {fold['brier_score']:.3f} |"
        for fold in validation["folds"]
    )

    return f"""# Model evaluation

The selector is an L2-regularised logistic regression trained to predict `worth_asking`. It does **not** predict survival probability. FFWS 2025 Game 4 is held out completely and is used only after the model is frozen.

## Dataset

- Training matches: {', '.join(metrics['training_matches'])}
- Training rows: {metrics['training_rows']}
- Positive question-worthy rows: {metrics['positive_rows']}
- Positive prevalence: {metrics['positive_prevalence']:.3f}
- Features: {', '.join(f'`{feature}`' for feature in metrics['features'])}

## Leave-one-match-out validation

| Held-out match | Balanced accuracy | Average precision | Brier score |
|---|---:|---:|---:|
{fold_rows}

- Mean balanced accuracy: **{validation['mean_balanced_accuracy']:.3f}**
- Mean average precision: **{validation['mean_average_precision']:.3f}**
- Pooled model Brier score: **{validation['pooled_brier_score']:.3f}**
- Fold-prevalence baseline Brier score: **{validation['baseline_brier_score']:.3f}**

## Ranking comparison

Across {ranking['timestamp_groups']} labelled timestamp groups:

- Top model-ranked row labelled worth asking: **{ranking['model_top_positive_rate']:.1%}**
- Naive fewest-active row labelled worth asking: **{ranking['naive_fewest_active_positive_rate']:.1%}**

This ranking comparison is descriptive because the historical sheets contain a sparse, human-preselected subset rather than all 12 squads at every timestamp. It must not be presented as conclusive ranking lift.

The two top-positive rates are identical, so this evaluation does not establish that the current model outperforms the naive rule. It demonstrates the reproducible model-training and held-out-scoring workflow only.

## Standardised model coefficients

| Feature | Coefficient |
|---|---:|
{coefficient_rows}

Positive coefficients increase similarity to historical states labelled worth asking; negative coefficients decrease it. These values are not causal explanations.

## Held-out Game 4 output

The frozen selector scores Game 4 states after `reconstruct_game4.py` derives past-only change and kill-feed context. `generate_game4.py` then applies observation confidence, recent-event, tournament-context, forward settleability and spacing rules. Outcomes are computed separately from later evidence and never enter the model features.
"""


def main() -> None:
    data = load_training_data()
    X = data[FEATURES].astype(float)
    y = data["worth_asking"]
    groups = data["match_id"].astype(str)
    weights = (
        data["label_confidence"].map(confidence_weight)
        if "label_confidence" in data.columns
        else pd.Series(np.ones(len(data)), index=data.index)
    )

    logo = LeaveOneGroupOut()
    oof_probabilities = pd.Series(np.nan, index=data.index, dtype=float)
    baseline_probabilities = pd.Series(np.nan, index=data.index, dtype=float)
    fold_metrics: list[dict] = []

    print("\nHistorical leave-one-match-out validation:")
    for train_idx, test_idx in logo.split(X, y, groups):
        if y.iloc[train_idx].nunique() < 2:
            raise ValueError("A validation training fold contains only one class")

        model = make_model()
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            model__sample_weight=weights.iloc[train_idx],
        )
        probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
        prevalence = float(y.iloc[train_idx].mean())
        predictions = (probabilities >= 0.5).astype(int)
        held_out = str(groups.iloc[test_idx].iloc[0])

        oof_probabilities.iloc[test_idx] = probabilities
        baseline_probabilities.iloc[test_idx] = prevalence
        fold = {
            "held_out_match": held_out,
            "test_rows": int(len(test_idx)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y.iloc[test_idx], predictions)
            ),
            "average_precision": float(
                average_precision_score(y.iloc[test_idx], probabilities)
            ),
            "brier_score": float(
                brier_score_loss(y.iloc[test_idx], probabilities)
            ),
            "training_prevalence": prevalence,
        }
        fold_metrics.append(fold)
        print(
            f"{held_out}: balanced_accuracy={fold['balanced_accuracy']:.3f}, "
            f"average_precision={fold['average_precision']:.3f}, "
            f"brier={fold['brier_score']:.3f}"
        )

    if oof_probabilities.isna().any():
        raise ValueError("Leave-one-match-out predictions are incomplete")

    oof = data.copy()
    oof["oof_worth_score"] = oof_probabilities
    oof["fold_prevalence_baseline"] = baseline_probabilities
    ranking = ranking_comparison(oof)

    final_model = make_model()
    final_model.fit(X, y, model__sample_weight=weights)
    coefficients = final_model.named_steps["model"].coef_[0]
    standardized_coefficients = {
        feature: round(float(value), 6) for feature, value in zip(FEATURES, coefficients)
    }

    metrics = {
        "model_type": "L2-regularised logistic regression",
        "target": "worth_asking",
        "features": FEATURES,
        "training_matches": sorted(groups.unique().tolist()),
        "training_rows": int(len(data)),
        "positive_rows": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "validation": {
            "method": "LeaveOneGroupOut grouped by match_id",
            "folds": fold_metrics,
            "mean_balanced_accuracy": float(
                np.mean([fold["balanced_accuracy"] for fold in fold_metrics])
            ),
            "mean_average_precision": float(
                np.mean([fold["average_precision"] for fold in fold_metrics])
            ),
            "pooled_brier_score": float(brier_score_loss(y, oof_probabilities)),
            "baseline_brier_score": float(
                brier_score_loss(y, baseline_probabilities)
            ),
        },
        "ranking": ranking,
        "standardized_coefficients": standardized_coefficients,
        "future_information_used_in_features": False,
    }

    package = {
        "model": final_model,
        "features": FEATURES,
        "target": "worth_asking",
        "training_matches": metrics["training_matches"],
        "training_rows": metrics["training_rows"],
        "positive_rows": metrics["positive_rows"],
        "metrics": metrics,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRAINING_OUTPUT_PATH, index=False)
    oof.to_csv(OOF_OUTPUT_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    RESULTS_PATH.write_text(render_results(metrics), encoding="utf-8")
    joblib.dump(package, MODEL_PATH)

    print("\nMODEL TRAINED")
    print("Games:", package["training_matches"])
    print("Rows:", package["training_rows"])
    print("Positive states:", package["positive_rows"])
    print("Metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
