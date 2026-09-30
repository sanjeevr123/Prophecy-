from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "pipeline" / "config" / "ffws2025.yaml"
MODEL_PATH = ROOT / "data" / "out" / "question_selector.joblib"
METRICS_PATH = ROOT / "data" / "out" / "model_metrics.json"
SCORED_PATH = ROOT / "data" / "out" / "scored_game4.csv"
CANDIDATES_PATH = ROOT / "data" / "out" / "candidates.csv"
QUESTIONS_PATH = ROOT / "data" / "out" / "questions.json"
QUALITY_PATH = ROOT / "data" / "out" / "data_quality.json"
REPORT_PATH = ROOT / "data" / "out" / "backend_validation.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise ValueError(message)
    checks.append(message)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    package = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
    scored = pd.read_csv(SCORED_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

    checks: list[str] = []
    require(
        str(config["demo_match_id"]) not in package["training_matches"],
        "held-out Game 4 is absent from model training",
        checks,
    )
    require(
        package["target"] == "worth_asking",
        "model target is worth_asking rather than a mislabeled survival probability",
        checks,
    )
    require(
        metrics["future_information_used_in_features"] is False,
        "model metadata declares no future features",
        checks,
    )
    require(
        metrics["ranking"].get("model_lift_claimed") is False,
        "model metadata does not claim superiority over the naive ranking rule",
        checks,
    )
    require(
        len(payload["questions"]) <= int(config["max_questions"]),
        "question count respects the configured maximum",
        checks,
    )
    require(
        payload.get("input_mode")
        == "prelabelled_broadcast_observations_emulating_future_ocr_live_ingest",
        "output identifies pre-labelled replay inputs as an emulation of future OCR/live ingest",
        checks,
    )
    require(
        payload["decision_policy"].get("future_results_used_for_model_features_or_ranking")
        is False,
        "output limits the no-future-result claim to model features and ranking",
        checks,
    )
    require(
        payload["decision_policy"].get("prototype_scheduling_uses_known_replay_milestones")
        is True,
        "output discloses the replay-only milestone scheduling guard",
        checks,
    )

    selected = candidates[candidates["selected"].astype(str).str.lower().eq("true")].copy()
    model_questions = [
        question
        for question in payload["questions"]
        if question.get("selection_method") == "ml_ranked_diversity_policy"
    ]
    require(
        len(selected) == len(model_questions),
        "every ML-ranked question maps to one selected model candidate",
        checks,
    )
    require(
        5 <= len(payload["questions"]) <= int(config["max_questions"]),
        "the showcase contains five to seven deliberately spaced questions",
        checks,
    )
    require(
        len(selected) <= int(config["max_ml_questions"]),
        "ML-selected question count respects the configured limit",
        checks,
    )
    require(
        int(payload["input_summary"]["scored_game4_states"]) == len(scored),
        "questions.json records the complete scored Game 4 row count",
        checks,
    )
    require(
        quality["source_time_bases"]["events_input"] == "video"
        and quality["source_time_bases"]["derived_features"] == "game",
        "kill-feed video timestamps are normalised to game time before feature construction",
        checks,
    )
    require(
        any("pre-labelled broadcast observations" in limitation for limitation in quality["important_limitations"]),
        "data-quality metadata discloses that replay observations emulate future OCR/live ingest",
        checks,
    )

    question_texts = [str(question["text"]) for question in payload["questions"]]
    require(
        len(question_texts) == len(set(question_texts)),
        "the showcase contains no duplicate question wording",
        checks,
    )
    model_squads = [str(question["squad"]) for question in model_questions]
    require(
        len(model_squads) == len(set(model_squads)),
        "the diversity policy selects at most one ML question per squad",
        checks,
    )
    binary_answers = {
        str(question["correct_option_id"])
        for question in model_questions
        if question["status"] != "void"
    }
    require(
        {"yes", "no"}.issubset(binary_answers),
        "the resolved ML showcase includes both YES and NO outcomes",
        checks,
    )
    showcase_questions = [
        question
        for question in payload["questions"]
        if question.get("selection_method") == "verified_broadcast_showcase_trigger"
    ]
    require(
        {question["text"] for question in showcase_questions}
        == {
            "Will FX keep all four players standing through this fight?",
            "Will GOW survive the next 15 seconds?",
        },
        "the two requested frame-verified showcase questions are present",
        checks,
    )
    require(
        payload["decision_policy"].get("card_sources")
        == {
            "ml_ranked": len(model_questions),
            "verified_showcase": len(showcase_questions),
            "verified_milestone": 2,
        },
        "output records the model, showcase and milestone card composition",
        checks,
    )

    previous_result_end = -1.0
    for question in payload["questions"]:
        question_id = str(question["id"])
        show = float(question["show_at_s"])
        close = float(question["close_at_s"])
        collapse = float(question["collapse_at_s"])
        window_end = float(question["window_end_s"])
        resolve = float(question["resolve_at_s"])
        result_hide = float(question["result_hide_at_s"])
        anchor_game_s = float(question["anchor_game_s"])
        require(show >= previous_result_end, f"{question_id} does not overlap a prior card", checks)
        observed_at_s = float(question["verification"]["observed_at_s"])
        require(
            abs(observed_at_s - (anchor_game_s - float(config["video_start_game_s"]))) < 0.01,
            f"{question_id} converts its verified game clock to video time correctly",
            checks,
        )
        require(
            0 <= show - observed_at_s <= float(config["milestone_card_delay_s"]),
            f"{question_id} appears from fresh verified evidence",
            checks,
        )
        require(
            abs((close - show) - float(config["answer_window_s"])) < 0.01,
            f"{question_id} has the configured answer window",
            checks,
        )
        require(
            close <= collapse <= window_end,
            f"{question_id} collapses after locking and before settlement",
            checks,
        )
        require(
            abs(float(question["collapse_delay_s"]) - float(config["lock_confirmation_s"])) < 0.01,
            f"{question_id} exposes the configured post-answer collapse delay",
            checks,
        )
        require(resolve >= window_end, f"{question_id} cannot resolve before its window", checks)
        require(result_hide >= resolve, f"{question_id} keeps its result visible after resolution", checks)
        require(
            result_hide <= float(config["video_duration_s"]),
            f"{question_id} remains inside the video duration",
            checks,
        )
        previous_result_end = result_hide

        if question.get("selection_method") != "ml_ranked_diversity_policy":
            continue
        contract_horizon_s = float(question["contract_horizon_s"])
        require(
            abs((window_end - show) - contract_horizon_s) < 0.01,
            f"{question_id} has its declared contract horizon",
            checks,
        )
        candidate = selected[selected["question_id"].astype(str).eq(question_id)]
        require(not candidate.empty, f"{question_id} has a selected candidate trace", checks)
        row = candidate.iloc[0]
        require(
            str(row["contract_type"]) == str(question["contract"]),
            f"{question_id} preserves the assigned diversity contract",
            checks,
        )
        require(
            str(row["panel_visible_at_anchor"]).lower() == "true",
            f"{question_id} begins from a scoreboard-verified anchor",
            checks,
        )
        require(
            int(row["recent_active_drop"]) > 0
            or int(row["knocked"]) > int(row["previous_knocked"]),
            f"{question_id} follows a newly verified negative roster transition",
            checks,
        )
        require(
            abs(float(row["worth_score"]) - float(question["model_worthiness_score"])) < 0.001,
            f"{question_id} preserves its frozen model score",
            checks,
        )
        require(
            abs(float(row["selection_score"]) - float(question["selection_score"])) < 0.001,
            f"{question_id} preserves its composite decision score",
            checks,
        )
        require(
            float(row["worth_score"]) >= float(config["min_worth_score"]),
            f"{question_id} clears the model threshold",
            checks,
        )
        require(
            float(question["state_age_at_anchor_s"]) <= 15,
            f"{question_id} uses fresh verified state",
            checks,
        )
        if question["status"] == "void":
            require(
                question["correct_option_id"] is None and bool(question.get("void_reason")),
                f"{question_id} voids without assigning a correct answer",
                checks,
            )
        else:
            require(
                question["correct_option_id"] in {"yes", "no"},
                f"{question_id} resolved answer is binary and explicit",
                checks,
            )
            threshold = {
                "recovery": 2,
                "two_player_hold": 2,
                "roster_hold": int(row["active"]),
                "final_four_reach": 1,
                "last_stand": 1,
                "survival": 1,
            }[str(question["contract"])]
            expected_answer = "yes" if int(row["settlement_active"]) >= threshold else "no"
            require(
                question["correct_option_id"] == expected_answer,
                f"{question_id} answer matches the reconstructed settlement state",
                checks,
            )
            require(
                question["status"] == str(row["settlement_status"]),
                f"{question_id} status matches its candidate audit row",
                checks,
            )
            require(
                question["verification"]["source"] == str(row["settlement_source"]),
                f"{question_id} exposes the recorded settlement source",
                checks,
            )

    report = {
        "status": "passed",
        "checks_passed": len(checks),
        "checks": checks,
        "training_matches": package["training_matches"],
        "training_rows": int(package["training_rows"]),
        "scored_game4_states": int(len(scored)),
        "selected_ml_questions": int(len(selected)),
        "resolved_ml_questions": int(selected["settlement_status"].eq("resolved").sum()),
        "void_ml_questions": int(selected["settlement_status"].eq("void").sum()),
        "questions_sha256": file_hash(QUESTIONS_PATH),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Backend validation passed: {len(checks)} checks.")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
