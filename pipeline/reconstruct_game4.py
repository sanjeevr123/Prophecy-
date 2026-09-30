from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "pipeline" / "config" / "ffws2025.yaml"
OBSERVATIONS_PATH = ROOT / "data" / "demo" / "game4_observations.csv"
EVENTS_PATH = ROOT / "data" / "labels" / "events.csv"
PANEL_PATH = ROOT / "data" / "labels" / "panel_visible.csv"
OUTPUT_PATH = ROOT / "data" / "out" / "game4_features.csv"
QUALITY_PATH = ROOT / "data" / "out" / "data_quality.json"

ALLOWED_EVENTS = {"knocked", "revived", "eliminated", "squad_eliminated"}
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.0}


def visible_overlap(panel: pd.DataFrame, start_s: float, end_s: float) -> float:
    visible = panel[panel["visible"]]
    overlap = 0.0
    for _, interval in visible.iterrows():
        overlap += max(
            0.0,
            min(end_s, float(interval["end_s"]))
            - max(start_s, float(interval["start_s"])),
        )
    return overlap


def panel_visible_at(panel: pd.DataFrame, time_s: float) -> bool:
    return bool(
        (
            panel["visible"]
            & panel["start_s"].astype(float).le(time_s)
            & panel["end_s"].astype(float).ge(time_s)
        ).any()
    )


def video_to_game_time(video_time_s: pd.Series | float, config: dict):
    return video_time_s + float(config["video_start_game_s"])


def validate_observations(observations: pd.DataFrame, config: dict) -> None:
    required = {
        "match_id",
        "time_s",
        "squad",
        "active",
        "knocked",
        "teams_remaining",
        "players_remaining",
        "label_confidence",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Game 4 observations are missing columns: {missing}")

    match_id = str(config["demo_match_id"])
    if not observations["match_id"].astype(str).eq(match_id).all():
        raise ValueError("Game 4 observations contain another match")

    known_squads = set(config["squads"])
    unknown_squads = sorted(set(observations["squad"].astype(str)) - known_squads)
    if unknown_squads:
        raise ValueError(f"Unknown Game 4 squad tags: {unknown_squads}")

    duplicate_mask = observations.duplicated(["match_id", "time_s", "squad"], keep=False)
    if duplicate_mask.any():
        duplicates = observations.loc[duplicate_mask, ["time_s", "squad"]].to_dict("records")
        raise ValueError(f"Duplicate Game 4 squad snapshots: {duplicates}")

    for column in ["time_s", "active", "knocked", "teams_remaining", "players_remaining"]:
        observations[column] = pd.to_numeric(observations[column], errors="raise")

    if (~observations["active"].between(0, 4)).any():
        raise ValueError("Active-player counts must be between 0 and 4")
    if (~observations["knocked"].between(0, 4)).any():
        raise ValueError("Knocked-player counts must be between 0 and 4")
    if ((observations["active"] + observations["knocked"]) > 4).any():
        raise ValueError("Active plus knocked players cannot exceed four")

    game_start = 0.0
    game_end = float(config["game_end_s"])
    if (~observations["time_s"].between(game_start, game_end, inclusive="both")).any():
        raise ValueError("Game 4 observation time falls outside the configured match")

    for time_s, group in observations.groupby("time_s"):
        if group["teams_remaining"].nunique() != 1:
            raise ValueError(f"Conflicting teams_remaining values at {time_s}s")
        if group["players_remaining"].nunique() != 1:
            raise ValueError(f"Conflicting players_remaining values at {time_s}s")


def validate_events(events: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, bool]:
    required = {"match_id", "time_s", "squad", "event"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events.csv is missing columns: {missing}")

    match_id = str(config["demo_match_id"])
    events = events[events["match_id"].astype(str).eq(match_id)].copy()
    events["time_s"] = pd.to_numeric(events["time_s"], errors="raise")
    events["event"] = events["event"].astype(str).str.strip().str.lower()
    events["squad"] = events["squad"].astype(str).str.strip()

    unknown_events = sorted(set(events["event"]) - ALLOWED_EVENTS)
    if unknown_events:
        raise ValueError(f"Unsupported event labels: {unknown_events}")
    unknown_squads = sorted(set(events["squad"]) - set(config["squads"]))
    if unknown_squads:
        raise ValueError(f"Unknown event squad tags: {unknown_squads}")
    if str(config.get("events_time_basis", "game")) == "video":
        if (~events["time_s"].between(0, float(config["video_duration_s"]), inclusive="both")).any():
            raise ValueError("Game 4 kill-feed video timestamp falls outside the prototype video")
        events["source_video_time_s"] = events["time_s"]
        events["time_s"] = video_to_game_time(events["time_s"], config)
    elif (~events["time_s"].between(0, float(config["game_end_s"]), inclusive="both")).any():
        raise ValueError("Game 4 kill-feed event falls outside the configured match")

    if (~events["time_s"].between(0, float(config["game_end_s"]), inclusive="both")).any():
        raise ValueError("Normalised Game 4 kill-feed event falls outside the match clock")

    event_order_was_monotonic = bool(events["time_s"].is_monotonic_increasing)
    events = events.sort_values(["time_s"], kind="stable").reset_index(drop=True)
    return events, not event_order_was_monotonic


def enrich_observations(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    event_window_s = float(config["recent_event_window_s"])
    change_window_s = float(config["state_change_window_s"])
    panel_window_s = float(config["panel_history_window_s"])
    video_start_game_s = float(config["video_start_game_s"])
    game_end_s = float(config["game_end_s"])

    enriched = observations.copy()
    for column in ["previous_active", "recent_active_drop", "match_progress"]:
        if column in enriched.columns:
            enriched[f"annotated_{column}"] = pd.to_numeric(enriched[column], errors="coerce")

    enriched = enriched.sort_values(["squad", "time_s"], kind="stable").reset_index(drop=True)
    enriched["previous_observed_at_s"] = enriched.groupby("squad")["time_s"].shift(1)
    enriched["previous_active"] = enriched.groupby("squad")["active"].shift(1).fillna(4).astype(int)
    enriched["previous_knocked"] = (
        enriched.groupby("squad")["knocked"].shift(1).fillna(0).astype(int)
    )
    enriched["observation_gap_s"] = enriched["time_s"] - enriched["previous_observed_at_s"]
    enriched["video_time_s"] = enriched["time_s"] - video_start_game_s
    enriched["match_progress"] = (enriched["time_s"] / game_end_s).clip(0, 1)

    recent_knocks: list[int] = []
    recent_eliminations: list[int] = []
    recent_revives: list[int] = []
    recent_event_total: list[int] = []
    last_event_at: list[float | None] = []
    last_event_type: list[str] = []
    panel_visible: list[bool] = []
    panel_duty_cycle: list[float] = []

    for _, row in enriched.iterrows():
        time_s = float(row["time_s"])
        squad_events = events[
            events["squad"].eq(str(row["squad"]))
            & events["time_s"].gt(time_s - event_window_s)
            & events["time_s"].le(time_s)
        ]
        counts = squad_events["event"].value_counts()
        recent_knocks.append(int(counts.get("knocked", 0)))
        recent_eliminations.append(int(counts.get("eliminated", 0) + counts.get("squad_eliminated", 0)))
        recent_revives.append(int(counts.get("revived", 0)))
        recent_event_total.append(int(len(squad_events)))
        if squad_events.empty:
            last_event_at.append(None)
            last_event_type.append("")
        else:
            latest_event = squad_events.iloc[-1]
            last_event_at.append(float(latest_event["time_s"]))
            last_event_type.append(str(latest_event["event"]))

        window_start = max(0.0, time_s - panel_window_s)
        window_width = max(1.0, time_s - window_start)
        panel_visible.append(panel_visible_at(panel, time_s))
        panel_duty_cycle.append(
            min(1.0, visible_overlap(panel, window_start, time_s) / window_width)
        )

    enriched["recent_knocks"] = recent_knocks
    enriched["recent_eliminations"] = recent_eliminations
    enriched["recent_revives"] = recent_revives
    enriched["recent_event_count"] = recent_event_total
    enriched["last_event_at_s"] = last_event_at
    enriched["last_event_type"] = last_event_type
    enriched["panel_visible_at_anchor"] = panel_visible
    enriched["panel_duty_cycle"] = panel_duty_cycle
    enriched["state_age_s"] = 0.0
    enriched["observation_confidence_weight"] = (
        enriched["label_confidence"]
        .astype(str)
        .str.lower()
        .map(CONFIDENCE_WEIGHTS)
        .fillna(0.0)
    )

    raw_drop = (enriched["previous_active"] - enriched["active"]).clip(lower=0)
    recently_observed = enriched["observation_gap_s"].fillna(np.inf).le(change_window_s)
    event_confirms_change = enriched["recent_event_count"].gt(0)
    first_observation = enriched["previous_observed_at_s"].isna()
    enriched["recent_active_drop"] = raw_drop.where(
        recently_observed | event_confirms_change | (first_observation & enriched["time_s"].le(change_window_s)),
        0,
    ).astype(int)

    enriched["event_signal"] = (
        (
            enriched["recent_knocks"]
            + 1.25 * enriched["recent_eliminations"]
            + 0.5 * enriched["recent_revives"]
        )
        / 3.0
    ).clip(0, 1)

    if "annotated_previous_active" in enriched.columns:
        enriched["previous_active_mismatch"] = (
            enriched["annotated_previous_active"].notna()
            & enriched["annotated_previous_active"].ne(enriched["previous_active"])
        )
    if "annotated_recent_active_drop" in enriched.columns:
        enriched["recent_active_drop_mismatch"] = (
            enriched["annotated_recent_active_drop"].notna()
            & enriched["annotated_recent_active_drop"].ne(enriched["recent_active_drop"])
        )
    if "annotated_match_progress" in enriched.columns:
        enriched["match_progress_delta"] = (
            enriched["annotated_match_progress"] - enriched["match_progress"]
        ).abs()

    return enriched.sort_values(["time_s", "squad"], kind="stable").reset_index(drop=True)


def build_quality_report(
    observations: pd.DataFrame,
    enriched: pd.DataFrame,
    events: pd.DataFrame,
    panel: pd.DataFrame,
    events_were_reordered: bool,
    config: dict,
) -> dict:
    timestamp_counts = observations.groupby("time_s")["squad"].nunique()
    exact_event_duplicates = int(
        events.duplicated(["time_s", "squad", "event"], keep=False).sum()
    )
    visible_duration = visible_overlap(panel, 0.0, float(config["game_end_s"]))

    return {
        "match_id": config["demo_match_id"],
        "observation_rows": int(len(observations)),
        "observation_timestamps": int(timestamp_counts.size),
        "average_squads_per_timestamp": round(float(timestamp_counts.mean()), 3),
        "complete_12_squad_timestamps": int(timestamp_counts.eq(len(config["squads"])).sum()),
        "killfeed_event_rows": int(len(events)),
        "killfeed_event_types": {
            str(key): int(value) for key, value in events["event"].value_counts().sort_index().items()
        },
        "events_input_required_sorting": events_were_reordered,
        "exact_duplicate_event_rows": exact_event_duplicates,
        "panel_visible_fraction": round(
            visible_duration / float(config["game_end_s"]), 4
        ),
        "previous_active_mismatches": int(
            enriched.get("previous_active_mismatch", pd.Series(dtype=bool)).sum()
        ),
        "recent_active_drop_mismatches": int(
            enriched.get("recent_active_drop_mismatch", pd.Series(dtype=bool)).sum()
        ),
        "important_limitations": [
            "The replay inputs are pre-labelled broadcast observations that emulate future OCR/live ingestion; this pipeline does not extract them directly from match.mp4.",
            "Game 4 observations are sparse candidate snapshots, not a complete 12-squad panel at every timestamp.",
            "Kill-feed labels identify the affected squad and event type but not a player ID or credited killer.",
            "Generic eliminated rows are player-level events and are not treated as squad eliminations.",
        ],
        "source_time_bases": {
            "observations": str(config.get("observations_time_basis", "game")),
            "events_input": str(config.get("events_time_basis", "game")),
            "panel_visible_input": str(config.get("panel_visible_time_basis", "game")),
            "derived_features": "game",
            "questions": "video",
        },
    }


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    observations = pd.read_csv(OBSERVATIONS_PATH)
    events = pd.read_csv(EVENTS_PATH)
    panel = pd.read_csv(PANEL_PATH)

    validate_observations(observations, config)
    events, events_were_reordered = validate_events(events, config)

    panel = panel[panel["match_id"].astype(str).eq(str(config["demo_match_id"]))].copy()
    panel["start_s"] = pd.to_numeric(panel["start_s"], errors="raise")
    panel["end_s"] = pd.to_numeric(panel["end_s"], errors="raise")
    panel["visible"] = panel["visible"].astype(str).str.lower().eq("true")
    if str(config.get("panel_visible_time_basis", "game")) == "video":
        panel["source_video_start_s"] = panel["start_s"]
        panel["source_video_end_s"] = panel["end_s"]
        panel["start_s"] = video_to_game_time(panel["start_s"], config)
        panel["end_s"] = video_to_game_time(panel["end_s"], config)

    enriched = enrich_observations(observations, events, panel, config)
    quality = build_quality_report(
        observations,
        enriched,
        events,
        panel,
        events_were_reordered,
        config,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)
    QUALITY_PATH.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Reconstructed {len(enriched)} held-out Game 4 squad states.")
    print(f"Enriched them with {len(events)} past-only kill-feed events.")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Data quality audit: {QUALITY_PATH}")


if __name__ == "__main__":
    main()
