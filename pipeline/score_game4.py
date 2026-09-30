from pathlib import Path

import joblib
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data" / "out" / "question_selector.joblib"
)

GAME4_PATH = (
    ROOT / "data" / "out" / "game4_features.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "out" / "scored_game4.csv"
)

CONFIG_PATH = (
    ROOT / "pipeline" / "config" / "ffws2025.yaml"
)


def main():

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    features = package["features"]

    game4 = pd.read_csv(GAME4_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    game_start = 0.0
    game_end = float(config["game_end_s"])
    game4["time_s"] = pd.to_numeric(game4["time_s"], errors="raise")
    outside_game = ~game4["time_s"].between(
        game_start,
        game_end,
        inclusive="both",
    )
    if outside_game.any():
        invalid = game4.loc[outside_game, ["time_s", "squad"]].to_dict("records")
        raise ValueError(f"Game 4 observations fall outside the configured game clock: {invalid}")

    if "worth_asking" in game4.columns:
        game4 = game4.drop(
            columns=["worth_asking"]
        )

    if not (
        game4["match_id"]
        .astype(str)
        .eq(str(config["demo_match_id"]))
        .all()
    ):
        raise ValueError(
            "Game 4 file contains rows from another match."
        )

    # Make sure required model features exist.
    missing = [
        col
        for col in features
        if col not in game4.columns
    ]

    if missing:
        raise ValueError(
            f"Game 4 is missing model features: {missing}"
        )

    X = (
        game4[features]
        .fillna(0)
        .astype(float)
    )

    game4["worth_score"] = (
        model.predict_proba(X)[:, 1]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    game4.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Scored {len(game4)} Game 4 states."
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("Top model-selected states:")

    print(
        game4
        .sort_values(
            "worth_score",
            ascending=False
        )
        [
            [
                "time_s",
                "squad",
                "active",
                "knocked",
                "recent_active_drop",
                "teams_remaining",
                "worth_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
