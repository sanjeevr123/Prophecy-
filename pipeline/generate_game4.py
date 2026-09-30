from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "pipeline" / "config" / "ffws2025.yaml"
EVENTS_PATH = ROOT / "data" / "labels" / "events.csv"
STANDINGS_PATH = ROOT / "data" / "labels" / "standings.csv"
SCORED_PATH = ROOT / "data" / "out" / "scored_game4.csv"
MODEL_PATH = ROOT / "data" / "out" / "question_selector.joblib"
QUALITY_PATH = ROOT / "data" / "out" / "data_quality.json"
MILESTONES_PATH = ROOT / "data" / "demo" / "game4_milestones.csv"
OUTPUT_PATH = ROOT / "data" / "out" / "questions.json"
CANDIDATES_PATH = ROOT / "data" / "out" / "candidates.csv"


def yes_no_options() -> list[dict[str, str]]:
    return [{"id": "yes", "label": "YES"}, {"id": "no", "label": "NO"}]


def squad_options(squads: list[str]) -> list[dict[str, str]]:
    return [{"id": squad, "label": squad} for squad in squads]


def ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def to_video_time(game_time_s: float, config: dict) -> float:
    return round(float(game_time_s) - float(config["video_start_game_s"]), 2)


def normalise_events_to_game_time(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    normalised = events.copy()
    normalised["time_s"] = pd.to_numeric(normalised["time_s"], errors="raise")
    if str(config.get("events_time_basis", "game")) == "video":
        normalised["source_video_time_s"] = normalised["time_s"]
        normalised["time_s"] = normalised["time_s"] + float(config["video_start_game_s"])
    return normalised


def candidate_contract(candidate: pd.Series, config: dict) -> tuple[str, float]:
    active = int(candidate["active"])
    knocked = int(candidate["knocked"])
    teams_remaining = int(candidate["teams_remaining"])

    if teams_remaining <= 7 and (active <= 2 or knocked > 0):
        return "last_stand", float(config["endgame_survival_horizon_s"])
    if active == 1 and knocked >= 2:
        return "recovery", float(config["survival_horizon_s"])
    if active == 3 and int(candidate["recent_active_drop"]) > 0:
        return "roster_hold", float(config["short_roster_horizon_s"])
    if active == 2:
        return "two_player_hold", float(config["survival_horizon_s"])
    return "survival", float(config["survival_horizon_s"])


def tournament_lookup(standings: pd.DataFrame, config: dict) -> dict[str, dict]:
    table = standings[
        standings["match_number"].astype(int).eq(int(config["demo_match_number"]))
    ].copy()
    table["points_before_match"] = pd.to_numeric(table["points_before_match"], errors="raise")
    table["rank_before"] = pd.to_numeric(table["rank_before"], errors="raise").astype(int)
    points = table["points_before_match"].astype(float).tolist()

    lookup: dict[str, dict] = {}
    for _, row in table.iterrows():
        squad = str(row["squad"])
        squad_points = float(row["points_before_match"])
        other_points = [value for value in points if value != squad_points]
        nearest_gap = min((abs(squad_points - value) for value in other_points), default=10.0)
        lookup[squad] = {
            "rank_before": int(row["rank_before"]),
            "points_before_match": int(squad_points),
            "nearest_standings_gap": float(nearest_gap),
            "tournament_relevance": math.exp(-nearest_gap / 5.0),
        }
    return lookup


def evaluate_candidates(
    scored: pd.DataFrame,
    standings: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply only information available at or before each candidate anchor."""

    candidates = scored.copy()
    tournament = tournament_lookup(standings, config)
    candidates["model_rank"] = (
        candidates["worth_score"].rank(method="first", ascending=False).astype(int)
    )
    candidates["rank_before"] = candidates["squad"].map(
        lambda squad: tournament.get(str(squad), {}).get("rank_before")
    )
    candidates["points_before_match"] = candidates["squad"].map(
        lambda squad: tournament.get(str(squad), {}).get("points_before_match")
    )
    candidates["nearest_standings_gap"] = candidates["squad"].map(
        lambda squad: tournament.get(str(squad), {}).get("nearest_standings_gap", 10.0)
    )
    candidates["tournament_relevance"] = candidates["squad"].map(
        lambda squad: tournament.get(str(squad), {}).get("tournament_relevance", 0.0)
    )
    candidates["forward_settleability"] = candidates["panel_duty_cycle"].astype(float).clip(0, 1)

    weights = config["ranking_weights"]
    candidates["selection_score"] = candidates["observation_confidence_weight"].astype(float) * (
        float(weights["model"]) * candidates["worth_score"].astype(float)
        + float(weights["events"]) * candidates["event_signal"].astype(float)
        + float(weights["tournament"]) * candidates["tournament_relevance"].astype(float)
        + float(weights["settleability"]) * candidates["forward_settleability"].astype(float)
    )
    candidates["selection_rank"] = (
        candidates["selection_score"].rank(method="first", ascending=False).astype(int)
    )
    contract_specs = candidates.apply(lambda row: candidate_contract(row, config), axis=1)
    candidates["contract_type"] = contract_specs.map(lambda item: item[0])
    candidates["contract_horizon_s"] = contract_specs.map(lambda item: item[1])
    candidates["eligible"] = False
    candidates["rejection_reason"] = ""
    candidates["selected"] = False
    candidates["question_id"] = ""
    candidates["settlement_status"] = "not_evaluated"
    candidates["settlement_source"] = ""
    candidates["settlement_evidence_game_s"] = pd.NA
    candidates["settlement_game_s"] = pd.NA
    candidates["settlement_active"] = pd.NA
    candidates["settlement_knocked"] = pd.NA
    candidates["settlement_fact"] = ""

    threshold = float(config["min_worth_score"])
    final_three_game_s = float(config["final_three_at_game_s"])
    min_panel_duty = float(config["min_panel_duty_cycle"])

    for index, candidate in candidates.iterrows():
        reasons: list[str] = []
        active = int(candidate["active"])
        knocked = int(candidate["knocked"])
        target_game_s = float(candidate["time_s"]) + float(candidate["contract_horizon_s"])

        if float(candidate["worth_score"]) < threshold:
            reasons.append("below_model_threshold")
        if float(candidate["observation_confidence_weight"]) <= 0:
            reasons.append("observation_confidence_too_low")
        if not bool(candidate["panel_visible_at_anchor"]):
            reasons.append("squad_state_not_panel_verified_at_anchor")
        if float(candidate["panel_duty_cycle"]) < min_panel_duty:
            reasons.append("insufficient_past_panel_duty_cycle")
        if not 1 <= active <= 3:
            reasons.append("not_a_vulnerable_live_state")
        # A prompt needs a newly verified negative roster transition, not just
        # a high fragility score or a nearby kill-feed line. This prevents a
        # stable/recovered settlement snapshot from becoming a new question.
        previous_knocked = int(candidate["previous_knocked"])
        if (
            int(candidate["recent_active_drop"]) <= 0
            and knocked <= previous_knocked
        ):
            reasons.append("no_new_verified_roster_deterioration")
        if target_game_s >= final_three_game_s and str(candidate["contract_type"]) != "final_four_reach":
            reasons.append("collides_with_final_three")

        if reasons:
            candidates.at[index, "rejection_reason"] = ";".join(reasons)
        else:
            candidates.at[index, "eligible"] = True

    return candidates


def select_candidates(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    spacing_s = float(config["min_question_spacing_s"])
    limit = int(config["max_ml_questions"])
    max_per_squad = int(config.get("max_questions_per_squad", 1))
    eligible_indices = candidates[candidates["eligible"]].index.tolist()

    def valid(combo: tuple[int, ...]) -> bool:
        squad_counts: dict[str, int] = {}
        for index in combo:
            squad = str(candidates.loc[index, "squad"])
            squad_counts[squad] = squad_counts.get(squad, 0) + 1
            if squad_counts[squad] > max_per_squad:
                return False
        anchors = [float(candidates.loc[index, "time_s"]) for index in combo]
        return all(abs(left - right) >= spacing_s for left, right in combinations(anchors, 2))

    def phase(anchor: float) -> str:
        if anchor < 600:
            return "early"
        if anchor < 800:
            return "middle"
        return "late"

    best_combo: tuple[int, ...] = ()
    best_objective = (-1, -1, -1, -1.0)
    for size in range(1, min(limit, len(eligible_indices)) + 1):
        for combo in combinations(eligible_indices, size):
            if not valid(combo):
                continue
            contracts = {str(candidates.loc[index, "contract_type"]) for index in combo}
            phases = {phase(float(candidates.loc[index, "time_s"])) for index in combo}
            score = sum(float(candidates.loc[index, "selection_score"]) for index in combo)
            objective = (len(combo), len(contracts), len(phases), score)
            if objective > best_objective:
                best_objective = objective
                best_combo = combo

    selected_indices = list(best_combo)
    selected_set = set(selected_indices)
    for index in eligible_indices:
        if index not in selected_set and not str(candidates.at[index, "rejection_reason"]):
            candidates.at[index, "rejection_reason"] = "not_selected_by_showcase_diversity_policy"

    chronological = sorted(
        selected_indices,
        key=lambda item: float(candidates.loc[item, "time_s"]),
    )
    for number, index in enumerate(chronological, start=1):
        candidates.at[index, "selected"] = True
        candidates.at[index, "question_id"] = f"q{number}"

    return candidates


def apply_event(active: int, knocked: int, event: str) -> tuple[int, int, bool]:
    if event == "knocked":
        if active <= 0:
            return active, knocked, False
        return active - 1, min(4, knocked + 1), True
    if event == "revived":
        if knocked > 0:
            return min(4, active + 1), knocked - 1, True
        if active + knocked < 4:
            return active + 1, knocked, True
        return active, knocked, False
    if event == "eliminated":
        if knocked > 0:
            return active, knocked - 1, True
        if active > 0:
            return active - 1, knocked, True
        return active, knocked, False
    if event == "squad_eliminated":
        return 0, 0, True
    return active, knocked, False


def settle_candidate(
    candidate: pd.Series,
    scored: pd.DataFrame,
    events: pd.DataFrame,
    milestones: pd.DataFrame,
    config: dict,
) -> dict:
    """Settle after selection; future evidence never affects model features/rank."""

    squad = str(candidate["squad"])
    anchor_game_s = float(candidate["time_s"])
    target_game_s = anchor_game_s + float(candidate["contract_horizon_s"])
    latest_allowed_game_s = min(
        float(config["game_end_s"]),
        target_game_s + float(config["max_settlement_delay_s"]),
    )
    active = int(candidate["active"])
    knocked = int(candidate["knocked"])
    trace = [
        {
            "game_time_s": anchor_game_s,
            "players_active": active,
            "players_knocked": knocked,
            "source": "verified_squad_snapshot",
        }
    ]

    timeline: list[dict] = []
    future_events = events[
        events["squad"].astype(str).eq(squad)
        & events["time_s"].astype(float).gt(anchor_game_s)
        & events["time_s"].astype(float).le(latest_allowed_game_s)
    ]
    for _, row in future_events.iterrows():
        timeline.append(
            {
                "game_time_s": float(row["time_s"]),
                "priority": 0,
                "kind": "event",
                "event": str(row["event"]),
            }
        )

    future_snapshots = scored[
        scored["squad"].astype(str).eq(squad)
        & scored["time_s"].astype(float).gt(anchor_game_s)
        & scored["time_s"].astype(float).le(latest_allowed_game_s)
    ]
    for _, row in future_snapshots.iterrows():
        timeline.append(
            {
                "game_time_s": float(row["time_s"]),
                "priority": 1,
                "kind": "snapshot",
                "active": int(row["active"]),
                "knocked": int(row["knocked"]),
                "confidence": str(row["label_confidence"]),
            }
        )

    final_three_game_s = float(config["final_three_at_game_s"])
    final_three_squads = set(
        scored[
            scored["time_s"].astype(float).eq(final_three_game_s)
            & scored["teams_remaining"].astype(int).eq(3)
        ]["squad"].astype(str)
    )
    if (
        target_game_s <= final_three_game_s <= latest_allowed_game_s
        and len(final_three_squads) == 3
        and squad not in final_three_squads
    ):
        timeline.append(
            {
                "game_time_s": final_three_game_s,
                "priority": 2,
                "kind": "complete_remaining_roster",
                "roster_label": "final-three",
            }
        )

    if str(candidate["contract_type"]) == "final_four_reach":
        final_four = milestone(milestones, "final_four", config)
        final_four_squads = set(milestone_squads(final_four))
        if squad not in final_four_squads:
            timeline.append(
                {
                    "game_time_s": float(final_four["observed_game_s"]),
                    "priority": 2,
                    "kind": "complete_remaining_roster",
                    "roster_label": "final-four",
                }
            )

    timeline.sort(key=lambda item: (item["game_time_s"], item["priority"]))
    contradictions: list[str] = []

    # Apply every update at the same timestamp before deciding whether the
    # question can settle. A verified scoreboard snapshot is deliberately
    # ordered after kill-feed reconstruction, so it can correct an incomplete
    # event sequence (for example, a revival the feed did not expose).
    position = 0
    while position < len(timeline):
        time_s = float(timeline[position]["game_time_s"])
        same_time: list[dict] = []
        while (
            position < len(timeline)
            and float(timeline[position]["game_time_s"]) == time_s
        ):
            same_time.append(timeline[position])
            position += 1

        source: str | None = None
        fact: str | None = None
        explicitly_terminal = False
        for item in same_time:
            if item["kind"] == "event":
                active, knocked, valid = apply_event(active, knocked, str(item["event"]))
                if not valid:
                    contradictions.append(f"{item['event']}@{time_s:g}")
                    continue
                source = f"killfeed_{item['event']}"
                fact = (
                    f"Kill-feed {item['event']} update reconstructs {squad} at "
                    f"{active} active and {knocked} knocked."
                )
                explicitly_terminal = str(item["event"]) == "squad_eliminated"
            elif item["kind"] == "snapshot":
                active = int(item["active"])
                knocked = int(item["knocked"])
                source = "verified_squad_snapshot"
                fact = (
                    f"Verified squad snapshot shows {active} active and "
                    f"{knocked} knocked."
                )
                explicitly_terminal = False
            else:
                active = 0
                knocked = 0
                roster_label = str(item["roster_label"])
                source = f"verified_complete_{roster_label.replace('-', '_')}_roster"
                fact = f"The verified {roster_label} roster does not include {squad}."
                explicitly_terminal = True

            trace.append(
                {
                    "game_time_s": time_s,
                    "players_active": active,
                    "players_knocked": knocked,
                    "source": source,
                }
            )

        if (
            source is not None
            and active == 0
            and knocked == 0
            and str(candidate["contract_type"]) != "final_four_reach"
        ):
            explicitly_terminal = True
            if source.startswith("killfeed_"):
                source = "reconstructed_zero_roster_from_killfeed"
                fact = f"The verified kill-feed sequence reconstructs {squad} at zero remaining players."

        # A group containing only contradictory events is not evidence.
        if source is None or fact is None:
            continue
        if explicitly_terminal and time_s <= target_game_s:
            return {
                "status": "resolved",
                "evidence_game_s": time_s,
                "resolve_game_s": target_game_s,
                "active": 0,
                "knocked": 0,
                "source": source,
                "fact": fact,
                "trace": trace,
                "contradictions": contradictions,
            }
        if time_s >= target_game_s:
            return {
                "status": "resolved",
                "evidence_game_s": time_s,
                "resolve_game_s": time_s,
                "active": active,
                "knocked": knocked,
                "source": source,
                "fact": fact,
                "trace": trace,
                "contradictions": contradictions,
            }

    return {
        "status": "void",
        "resolve_game_s": latest_allowed_game_s,
        "source": "no_verified_post_window_state",
        "fact": "No reliable squad snapshot or state-changing kill-feed evidence returned after the window.",
        "trace": trace,
        "contradictions": contradictions,
    }


def standings_context(candidate: pd.Series, config: dict) -> str:
    squad = str(candidate["squad"])
    rank = int(candidate["rank_before"])
    gap = int(round(float(candidate["nearest_standings_gap"])))
    return (
        f"{squad} · {ordinal(rank)} before Game {config['demo_match_number']} · "
        f"{gap} point{'s' if gap != 1 else ''} from the nearest squad"
    )


def question_headline(active: int, knocked: int) -> str:
    if active == 1:
        return "LAST ACTIVE PLAYER"
    if knocked >= 2:
        return "REVIVAL WINDOW"
    if active == 2:
        return "TWO-PLAYER HOLD"
    return "ROSTER PRESSURE"


def build_model_question(
    candidate: pd.Series,
    settlement: dict,
    config: dict,
) -> dict:
    squad = str(candidate["squad"])
    contract = str(candidate["contract_type"])
    horizon_s = float(candidate["contract_horizon_s"])
    anchor_game_s = float(candidate["time_s"])
    show_at_s = to_video_time(anchor_game_s, config)
    close_at_s = show_at_s + float(config["answer_window_s"])
    window_end_s = show_at_s + horizon_s
    resolve_at_s = max(
        window_end_s,
        to_video_time(float(settlement["resolve_game_s"]), config),
    )
    active = int(candidate["active"])
    knocked = int(candidate["knocked"])
    status = str(settlement["status"])

    recent_events = []
    for label, column in [
        ("knock", "recent_knocks"),
        ("elimination", "recent_eliminations"),
        ("revive", "recent_revives"),
    ]:
        count = int(candidate[column])
        if count:
            recent_events.append(f"{count} {label}{'s' if count != 1 else ''}")
    event_phrase = ", ".join(recent_events) if recent_events else "no attributed kill-feed event"
    selection_summary = (
        f"Model rank {int(candidate['model_rank'])}; last {config['recent_event_window_s']}s: "
        f"{event_phrase}; {contract.replace('_', ' ')} contract chosen by the diversity policy; "
        f"past-only panel settleability {float(candidate['forward_settleability']):.0%}."
    )

    trace = [
        {
            **item,
            "at_s": to_video_time(float(item["game_time_s"]), config),
            "verified": True,
        }
        for item in settlement["trace"]
    ]

    success_threshold = {
        "recovery": 2,
        "two_player_hold": 2,
        "roster_hold": active,
        "final_four_reach": 1,
        "last_stand": 1,
        "survival": 1,
    }[contract]

    if contract == "recovery":
        headline = "REVIVAL WINDOW"
        text = f"Will {squad} rebuild to at least two active players {horizon_s:g} seconds from now?"
        settlement_note = f"Locks in 8 seconds and checks the next verified state at the {horizon_s:g}-second mark."
    elif contract == "two_player_hold":
        headline = "TWO-PLAYER HOLD"
        text = f"Will {squad} still have both active players {horizon_s:g} seconds from now?"
        settlement_note = f"Locks in 8 seconds and checks whether both remain active after {horizon_s:g} seconds."
    elif contract == "roster_hold":
        headline = "ROSTER HOLD"
        text = f"Will {squad} still have all {active} active players {horizon_s:g} seconds from now?"
        settlement_note = f"Locks in 8 seconds and checks the verified roster after {horizon_s:g} seconds."
    elif contract == "final_four_reach":
        headline = "FINAL-FOUR TEST"
        text = f"Will {squad} make the final four?"
        settlement_note = "Locks in 8 seconds and resolves when the final-four state is reached."
    elif contract == "last_stand":
        headline = "LAST STAND"
        text = f"Will {squad} still have an active player {horizon_s:g} seconds from now?"
        settlement_note = f"Locks in 8 seconds and checks the reconstructed roster after {horizon_s:g} seconds."
    else:
        headline = question_headline(active, knocked)
        text = f"Will {squad} still have at least one active player {horizon_s:g} seconds from now?"
        settlement_note = f"Locks in 8 seconds and checks the verified roster after {horizon_s:g} seconds."

    if status == "void":
        correct_option_id = None
        result_headline = "QUESTION VOIDED"
        result_detail = str(settlement["fact"])
        result_consequence = "No points are awarded or removed; the viewer's streak is preserved."
    else:
        settlement_active = int(settlement["active"])
        settlement_knocked = int(settlement["knocked"])
        succeeded = settlement_active >= success_threshold
        correct_option_id = "yes" if succeeded else "no"
        result_headline = {
            "recovery": f"{squad} REBUILD" if succeeded else f"{squad} CANNOT REBUILD",
            "two_player_hold": f"{squad} HOLD BOTH" if succeeded else f"{squad} LOSE THE HOLD",
            "roster_hold": f"{squad} HOLD THE ROSTER" if succeeded else f"{squad} DROP BELOW {active}",
            "final_four_reach": f"{squad} REACH FINAL FOUR" if succeeded else f"{squad} MISS FINAL FOUR",
            "last_stand": f"{squad} STAY ACTIVE" if succeeded else f"{squad} HAVE NO ACTIVE PLAYERS",
            "survival": f"{squad} STAY ACTIVE" if succeeded else f"{squad} ARE ELIMINATED",
        }[contract]
        result_detail = (
            f"Verified evidence at settlement shows {settlement_active} active and "
            f"{settlement_knocked} knocked."
        )
        result_consequence = (
            f"{squad} began Game {config['demo_match_number']} in "
            f"{ordinal(int(candidate['rank_before']))} place, "
            f"{int(round(float(candidate['nearest_standings_gap'])))} point"
            f"{'s' if int(round(float(candidate['nearest_standings_gap']))) != 1 else ''} "
            "from the nearest squad."
        )

    question = {
        "id": str(candidate["question_id"]),
        "contract": contract,
        "selection_method": "ml_ranked_diversity_policy",
        "squad": squad,
        "anchor_game_s": anchor_game_s,
        "show_at_s": show_at_s,
        "close_at_s": close_at_s,
        "collapse_at_s": close_at_s + float(config["lock_confirmation_s"]),
        "collapse_delay_s": float(config["lock_confirmation_s"]),
        "contract_horizon_s": horizon_s,
        "window_end_s": window_end_s,
        "resolve_at_s": resolve_at_s,
        "result_hide_at_s": min(
            resolve_at_s + float(config["result_display_s"]),
            float(config["video_duration_s"]),
        ),
        "headline": headline,
        "context": (
            f"{standings_context(candidate, config)} · last verified {active} active"
            + (f", {knocked} knocked" if knocked else "")
        ),
        "text": text,
        "settlement_note": settlement_note,
        "options": yes_no_options(),
        "correct_option_id": correct_option_id,
        "model_worthiness_score": round(float(candidate["worth_score"]), 4),
        "selection_score": round(float(candidate["selection_score"]), 4),
        "selection_summary": selection_summary,
        "selection_factors": {
            "model_rank": int(candidate["model_rank"]),
            "model_worthiness": round(float(candidate["worth_score"]), 4),
            "recent_killfeed_events": int(candidate["recent_event_count"]),
            "tournament_relevance": round(float(candidate["tournament_relevance"]), 4),
            "observation_confidence": str(candidate["label_confidence"]).lower(),
            "past_panel_duty_cycle": round(float(candidate["panel_duty_cycle"]), 4),
        },
        "state_age_at_anchor_s": float(candidate["state_age_s"]),
        "result_headline": result_headline,
        "result_detail": result_detail,
        "result_consequence": result_consequence,
        "verification": {
            "source": str(settlement["source"]),
            "observed_at_s": show_at_s,
            "observed_at_game_s": anchor_game_s,
            "settlement_evidence_game_s": settlement.get("evidence_game_s"),
            "confidence": str(candidate["label_confidence"]).lower(),
            "fact": str(settlement["fact"]),
            "state_reconstruction_warnings": settlement["contradictions"],
        },
        "live_state": trace,
        "status": status,
        "points": 100,
    }
    if status == "void":
        question["void_reason"] = str(settlement["fact"])
    return question


def milestone(milestones: pd.DataFrame, name: str, config: dict) -> pd.Series:
    rows = milestones[
        milestones["match_id"].astype(str).eq(str(config["demo_match_id"]))
        & milestones["milestone"].astype(str).eq(name)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {name} milestone; found {len(rows)}")
    row = rows.iloc[0]
    expected_video_s = to_video_time(float(row["observed_game_s"]), config)
    if abs(expected_video_s - float(row["observed_video_s"])) > 0.01:
        raise ValueError(f"{name} milestone has inconsistent game/video timestamps")
    return row


def milestone_squads(row: pd.Series) -> list[str]:
    return [squad for squad in str(row["squads"]).split("|") if squad]


def build_verified_showcase_question(
    milestones: pd.DataFrame,
    name: str,
    question_id: str,
    config: dict,
) -> dict:
    row = milestone(milestones, name, config)
    squad = milestone_squads(row)[0]
    show_at_s = float(row["observed_video_s"])
    close_at_s = show_at_s + float(config["answer_window_s"])
    resolve_at_s = float(row["result_video_s"])

    if name == "fx_full_strength_watch":
        contract = "full_strength_watch"
        headline = "FULL STRENGTH"
        context = "The broadcast verifies FX with all four players active as the camera follows the engagement"
        text = "Will FX keep all four players standing through this fight?"
        result_headline = "FX TAKE A LOSS"
        result_detail = "The kill feed confirms FX.Butzin as the first FX player eliminated."
        result_consequence = "The tournament leaders lose their full-strength roster advantage."
        fact = "FX have four active at card open; the broadcast kill feed confirms the first FX elimination at video 13:37."
        source = "verified_squad_panel_and_killfeed"
    elif name == "gow_survival_watch":
        contract = "survival_watch"
        headline = "SURVIVAL WATCH"
        context = "Six squads remain and GOW are present in the latest verified broadcast state"
        text = "Will GOW survive the next 15 seconds?"
        result_headline = "GOW ELIMINATED"
        result_detail = "The broadcast elimination banner confirms GOW finish fifth."
        result_consequence = "Only AG, EVOS, E1 and PE remain for the final-four prediction."
        fact = "GOW are present at video 15:00; the fifth-place elimination banner appears at video 15:15."
        source = "verified_broadcast_state_and_elimination_banner"
    else:
        raise ValueError(f"Unsupported verified showcase question: {name}")

    return {
        "id": question_id,
        "contract": contract,
        "selection_method": "verified_broadcast_showcase_trigger",
        "squad": squad,
        "anchor_game_s": float(row["observed_game_s"]),
        "show_at_s": show_at_s,
        "close_at_s": close_at_s,
        "collapse_at_s": close_at_s + float(config["lock_confirmation_s"]),
        "collapse_delay_s": float(config["lock_confirmation_s"]),
        "contract_horizon_s": resolve_at_s - show_at_s,
        "window_end_s": resolve_at_s,
        "resolve_at_s": resolve_at_s,
        "result_hide_at_s": min(
            resolve_at_s + float(config["result_display_s"]),
            float(config["video_duration_s"]),
        ),
        "headline": headline,
        "context": context,
        "text": text,
        "settlement_note": "Locks in 8 seconds and resolves from the stated broadcast-visible evidence.",
        "options": yes_no_options(),
        "correct_option_id": "no",
        "selection_summary": (
            "Frame-verified showcase trigger retained for contract variety; explicitly not ML-selected."
        ),
        "result_headline": result_headline,
        "result_detail": result_detail,
        "result_consequence": result_consequence,
        "state_age_at_show_s": 0.0,
        "verification": {
            "source": source,
            "observed_at_s": show_at_s,
            "observed_at_game_s": float(row["observed_game_s"]),
            "settlement_evidence_at_s": resolve_at_s,
            "confidence": str(row["confidence"]),
            "fact": fact,
        },
        "status": "resolved",
        "points": 100,
    }


def build_final_four_question(milestones: pd.DataFrame, question_id: str, config: dict) -> dict:
    row = milestone(milestones, "final_four", config)
    squads = milestone_squads(row)
    if len(squads) != 4:
        raise ValueError(f"Expected four squads in final-four milestone; found {squads}")
    observed_at_s = float(row["observed_video_s"])
    show_at_s = observed_at_s + float(config["milestone_card_delay_s"])
    close_at_s = show_at_s + float(config["answer_window_s"])
    resolve_at_s = float(row["result_video_s"])
    next_out = str(row["result_squad"])
    if next_out not in squads:
        raise ValueError("Final-four result squad is not in the verified roster")

    return {
        "id": question_id,
        "contract": "next_squad_out",
        "selection_method": "deterministic_verified_milestone",
        "anchor_game_s": float(row["observed_game_s"]),
        "show_at_s": show_at_s,
        "close_at_s": close_at_s,
        "collapse_at_s": close_at_s + float(config["lock_confirmation_s"]),
        "collapse_delay_s": float(config["lock_confirmation_s"]),
        "window_end_s": resolve_at_s,
        "resolve_at_s": resolve_at_s,
        "result_hide_at_s": resolve_at_s + float(config["result_display_s"]),
        "headline": "FINAL FOUR",
        "context": f"{', '.join(squads[:-1])} and {squads[-1]} remain at the verified overlay",
        "text": "Which squad drops before the final three?",
        "settlement_note": "Locks in 8 seconds and resolves at the verified final-three update.",
        "options": squad_options(squads),
        "correct_option_id": next_out,
        "selection_summary": "Triggered deterministically by the complete verified four-squad broadcast overlay.",
        "result_headline": f"{next_out} EXIT IN FOURTH",
        "result_detail": f"The next verified overlay removes {next_out} and shows three squads remaining.",
        "result_consequence": "The lobby is reduced to AG, EVOS and PE for the final prediction.",
        "state_age_at_show_s": show_at_s - observed_at_s,
        "verification": {
            "source": "verified_final_four_and_final_three_overlays",
            "observed_at_s": observed_at_s,
            "observed_at_game_s": float(row["observed_game_s"]),
            "settlement_evidence_at_s": resolve_at_s,
            "confidence": str(row["confidence"]),
            "fact": f"The final-four overlay shows {', '.join(squads)}; the final-three overlay excludes {next_out}.",
        },
        "status": "resolved",
        "points": 100,
    }


def build_final_three_question(
    scored: pd.DataFrame,
    milestones: pd.DataFrame,
    question_id: str,
    config: dict,
) -> dict:
    row = milestone(milestones, "final_three", config)
    final_three_game_s = float(row["observed_game_s"])
    squads = milestone_squads(row)
    snapshot_squads = set(
        scored[
            scored["time_s"].astype(float).eq(final_three_game_s)
            & scored["teams_remaining"].astype(int).eq(3)
        ]["squad"].astype(str)
    )
    if set(squads) != snapshot_squads or len(squads) != 3:
        raise ValueError(
            f"Final-three milestone {squads} does not match scored snapshots {sorted(snapshot_squads)}"
        )

    observed_at_s = float(row["observed_video_s"])
    show_at_s = observed_at_s + float(config["milestone_card_delay_s"])
    close_at_s = show_at_s + float(config["answer_window_s"])
    resolve_at_s = float(row["result_video_s"])
    booyah_squad = str(row["result_squad"])
    if booyah_squad not in squads:
        raise ValueError("Booyah squad is not present in the final-three milestone")

    return {
        "id": question_id,
        "contract": "final_three",
        "selection_method": "deterministic_verified_milestone",
        "anchor_game_s": final_three_game_s,
        "show_at_s": show_at_s,
        "close_at_s": close_at_s,
        "collapse_at_s": close_at_s + float(config["lock_confirmation_s"]),
        "collapse_delay_s": float(config["lock_confirmation_s"]),
        "window_end_s": resolve_at_s,
        "resolve_at_s": resolve_at_s,
        "result_hide_at_s": float(config["video_duration_s"]),
        "headline": "THREE LEFT",
        "context": f"{', '.join(squads[:-1])} and {squads[-1]} remain at the verified update",
        "text": "Who takes the Booyah?",
        "settlement_note": "Locks in 8 seconds and resolves when Game 4 ends.",
        "options": squad_options(squads),
        "correct_option_id": booyah_squad,
        "selection_summary": "Triggered deterministically by the verified complete three-squad roster.",
        "result_headline": f"{booyah_squad} TAKE THE BOOYAH",
        "result_detail": f"{booyah_squad} win Game {config['demo_match_number']}.",
        "result_consequence": "The match result now feeds the tournament standings update.",
        "state_age_at_show_s": show_at_s - observed_at_s,
        "verification": {
            "source": "verified_final_three_snapshot_and_broadcast_result",
            "observed_at_s": observed_at_s,
            "observed_at_game_s": final_three_game_s,
            "settlement_evidence_at_s": resolve_at_s,
            "confidence": str(row["confidence"]),
            "fact": f"Final-three snapshot contains {', '.join(squads)}; broadcast result names {booyah_squad}.",
        },
        "status": "resolved",
        "points": 100,
    }


def validate_questions(questions: list[dict], config: dict) -> None:
    duration = float(config["video_duration_s"])
    answer_window_s = float(config["answer_window_s"])
    ids: set[str] = set()
    texts: set[str] = set()
    model_squads: set[str] = set()
    previous_end = -1.0

    if len(questions) > int(config["max_questions"]):
        raise ValueError("Question count exceeds configured maximum")

    for question in questions:
        question_id = str(question["id"])
        if question_id in ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        ids.add(question_id)
        text = str(question["text"])
        if text in texts:
            raise ValueError(f"Duplicate question wording: {text}")
        texts.add(text)

        show = float(question["show_at_s"])
        close = float(question["close_at_s"])
        collapse = float(question["collapse_at_s"])
        window_end = float(question["window_end_s"])
        resolve = float(question["resolve_at_s"])
        result_hide = float(question["result_hide_at_s"])

        if not (0 <= show < close < collapse <= window_end <= resolve <= result_hide <= duration):
            raise ValueError(f"Invalid lifecycle for {question_id}")
        if abs((close - show) - answer_window_s) > 0.01:
            raise ValueError(f"Answer window must be {answer_window_s:g}s for {question_id}")
        if question.get("selection_method") == "ml_ranked_diversity_policy":
            expected_horizon = float(question["contract_horizon_s"])
            if abs((window_end - show) - expected_horizon) > 0.01:
                raise ValueError(f"Contract horizon must be {expected_horizon:g}s for {question_id}")
            squad = str(question["squad"])
            if squad in model_squads:
                raise ValueError(f"Showcase diversity allows only one ML-selected card for {squad}")
            model_squads.add(squad)
        if show < previous_end:
            raise ValueError(f"Question {question_id} overlaps the preceding result")
        previous_end = result_hide

        option_ids = {option["id"] for option in question["options"]}
        if question["status"] == "void":
            if question["correct_option_id"] is not None or "void_reason" not in question:
                raise ValueError(f"Void contract is incomplete for {question_id}")
        elif question["correct_option_id"] not in option_ids:
            raise ValueError(f"Correct option missing for {question_id}")

    binary_answers = {
        question["correct_option_id"]
        for question in questions
        if {option["id"] for option in question["options"]} == {"yes", "no"}
    }
    if not {"yes", "no"}.issubset(binary_answers):
        raise ValueError("Showcase set must contain both YES and NO resolved outcomes")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    scored = pd.read_csv(SCORED_PATH)
    events = pd.read_csv(EVENTS_PATH)
    standings = pd.read_csv(STANDINGS_PATH)
    milestones = pd.read_csv(MILESTONES_PATH)
    model_package = joblib.load(MODEL_PATH)
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

    match_id = str(config["demo_match_id"])
    scored = scored[scored["match_id"].astype(str).eq(match_id)].copy()
    events = events[events["match_id"].astype(str).eq(match_id)].copy()
    events["event"] = events["event"].astype(str).str.strip().str.lower()
    scored["time_s"] = pd.to_numeric(scored["time_s"], errors="raise")
    events = normalise_events_to_game_time(events, config)
    events = events.sort_values("time_s", kind="stable").reset_index(drop=True)

    candidates = evaluate_candidates(scored, standings, config)
    candidates = select_candidates(candidates, config)

    questions: list[dict] = []
    selected_indices = candidates[candidates["selected"]].sort_values("time_s").index.tolist()
    for index in selected_indices:
        candidate = candidates.loc[index]
        settlement = settle_candidate(candidate, scored, events, milestones, config)
        candidates.at[index, "settlement_status"] = settlement["status"]
        candidates.at[index, "settlement_source"] = settlement["source"]
        candidates.at[index, "settlement_evidence_game_s"] = settlement.get("evidence_game_s")
        candidates.at[index, "settlement_game_s"] = settlement["resolve_game_s"]
        candidates.at[index, "settlement_active"] = settlement.get("active")
        candidates.at[index, "settlement_knocked"] = settlement.get("knocked")
        candidates.at[index, "settlement_fact"] = settlement["fact"]
        questions.append(build_model_question(candidate, settlement, config))

    for milestone_name in ("fx_full_strength_watch", "gow_survival_watch"):
        questions.append(
            build_verified_showcase_question(
                milestones,
                milestone_name,
                f"q{len(questions) + 1}",
                config,
            )
        )

    questions.append(
        build_final_four_question(milestones, f"q{len(questions) + 1}", config)
    )
    questions.append(
        build_final_three_question(
            scored,
            milestones,
            f"q{len(questions) + 1}",
            config,
        )
    )
    questions = sorted(questions, key=lambda question: float(question["show_at_s"]))
    for number, question in enumerate(questions, start=1):
        old_id = str(question["id"])
        question["id"] = f"q{number}"
        candidates.loc[candidates["question_id"].eq(old_id), "question_id"] = question["id"]

    validate_questions(questions, config)
    candidates = candidates.sort_values(["time_s", "squad"], kind="stable").reset_index(drop=True)
    candidates.to_csv(CANDIDATES_PATH, index=False)

    metrics = model_package["metrics"]
    output = {
        "schema_version": 5,
        "match_id": match_id,
        "prototype_mode": "recorded_match_emulating_live_ingest",
        "input_mode": "prelabelled_broadcast_observations_emulating_future_ocr_live_ingest",
        "video_file": config["video_file"],
        "video_duration_s": float(config["video_duration_s"]),
        "video_starts_at_game_s": float(config["video_start_game_s"]),
        "selection_engine": (
            "Frozen L2 logistic-regression worthiness model plus past-only kill-feed, "
            "observation-confidence and tournament context; a deterministic diversity policy "
            "chooses distinct squads and question contracts"
        ),
        "model": {
            "target": model_package["target"],
            "features": model_package["features"],
            "training_matches": model_package["training_matches"],
            "training_rows": int(model_package["training_rows"]),
            "positive_rows": int(model_package["positive_rows"]),
            "mean_balanced_accuracy": round(
                float(metrics["validation"]["mean_balanced_accuracy"]), 4
            ),
            "mean_average_precision": round(
                float(metrics["validation"]["mean_average_precision"]), 4
            ),
            "important_note": "model_worthiness_score is not a survival probability",
        },
        "decision_policy": {
            "feature_cutoff": "all selection inputs are at or before anchor_game_s",
            "model_is_primary": f"{float(config['ranking_weights']['model']):.0%} of composite rank",
            "future_results_used_for_model_features_or_ranking": False,
            "prototype_scheduling_uses_known_replay_milestones": True,
            "settlement": "computed only after candidate selection",
            "showcase_diversity": (
                "maximum one ML-selected card per squad; maximise card count, contract variety, "
                "match-phase coverage and then composite score"
            ),
            "verified_showcase_triggers": ["fx_full_strength_watch", "gow_survival_watch"],
            "deterministic_milestones": ["final_four", "final_three"],
            "card_sources": {
                "ml_ranked": len(selected_indices),
                "verified_showcase": 2,
                "verified_milestone": 2,
            },
            "max_questions": int(config["max_questions"]),
        },
        "input_summary": {
            "scored_game4_states": int(len(scored)),
            "killfeed_events": int(quality["killfeed_event_rows"]),
            "panel_visible_fraction": quality["panel_visible_fraction"],
        },
        "questions": questions,
    }

    encoded = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(encoded, encoding="utf-8")

    selected = candidates[candidates["selected"]]
    resolved = selected["settlement_status"].eq("resolved").sum()
    print(
        f"Generated {len(selected)} ML-ranked diversified questions "
        f"({resolved} resolved, {len(selected) - resolved} void), two frame-verified showcase "
        "questions and two verified milestone questions."
    )
    for question in questions:
        model_score = question.get("model_worthiness_score")
        score_text = f" | worth {model_score:.3f}" if model_score is not None else ""
        print(
            f"{question['id']} | video {question['show_at_s']:>6.1f}s | "
            f"game {question['anchor_game_s']:>6.1f}s | {question['contract']} "
            f"| {question['status']}{score_text}"
        )
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Decision trace: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
