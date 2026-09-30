# Prophecy prototype disclosure

This document defines what the current repository demonstrates and the claims it can support.

## Experience being demonstrated

Prophecy is designed as a companion to a live Free Fire broadcast. The repository uses a recorded FFWS 2025 Game 4 video so question timing, interaction and settlement can be replayed consistently.

The interface uses “live” language to represent the intended viewer experience. The current demonstration is not a live match, live prediction service or live data feed.

## Data access

The system is limited to broadcast-visible evidence. It has no Garena game-server telemetry, private tournament feed or public Free Fire gameplay API.

The current backend consumes pre-labelled scoreboard, player-bar, kill-feed and milestone observations. These labels emulate the output of future OCR and video-event ingestion; the pipeline does not presently extract them from `match.mp4`.

The data can support claims about:

- what an overlay or kill feed showed at a recorded timestamp;
- the latest verified active/knocked roster state;
- global teams and players remaining;
- standings before Game 4; and
- later broadcast evidence used to settle a prediction.

It cannot support claims about locations, proximity, health, inventory, weapons, tactical intent, hidden fights or current off-camera state.

## Model role

The classifier predicts `worth_asking`: similarity to historical squad states labelled as suitable viewer-prediction moments. It does not predict survival, victory, fan sentiment or the correct answer.

Game 4 is held out from training. Later settlement evidence does not enter model features or the composite candidate rank. Question wording and answer contracts are deterministic once a state has been selected.

The historical and Game 4 labels are sparse human-selected samples rather than continuous full-lobby observations. The model therefore ranks the supplied candidate states, not all 12 squads at every moment. The current ranking comparison does not demonstrate an advantage over the naive fewest-active baseline, and no such performance claim is made.

## Question provenance

The schema-v5 backend output contains seven questions:

- three `ml_ranked_diversity_policy` cards;
- two `verified_broadcast_showcase_trigger` cards; and
- two `deterministic_verified_milestone` cards.

The FX full-strength and GOW survival prompts are frame-reviewed showcase triggers retained for contract variety. The final-four and final-three prompts are deterministic broadcast milestones. Neither category is represented as ML-selected.

The deployed viewer loads the schema-v5 `data/out/questions.json` directly, so the seven generated questions are the active card set. The schema-v2 `app/questions.json` is marked `legacy_unused`, retained only for historical reference and not read by the application.

## Timing and settlement

Question outcomes and timings are precomputed because the prototype operates on a recorded match. For ML-ranked questions, future evidence is evaluated only after selection. If later evidence is insufficient, the backend is designed to emit a void result rather than silently remove the question or assign an unsupported answer.

A replay-only scheduling guard uses the known final-three timestamp to protect the curated endgame sequence. This timing guard is disclosed in the schema-v5 metadata and is not described as a live model decision.

Kill-feed labels identify an affected squad and event type but not a player ID or credited killer. Repeated same-time labels may describe multiple players, and verified later snapshots are used to correct incomplete event reconstruction.

## Viewer behavior

The schema-v5 backend payload contains no model outcome probability or measured fan-pulse value. The viewer suppresses the fan-pulse display when that data is absent. Its result view currently derives an illustrative crowd-correct percentage deterministically from the question ID and timestamp; this is simulated UI content, not audience data.

Viewer points, streaks, accuracy and rewards are illustrative, stored only in browser memory and reset on reload. They have no monetary value.

The viewer follows `show_at_s`, `close_at_s`, `window_end_s`, `resolve_at_s` and `result_hide_at_s`, and it applies scoring only at resolution. It does not yet render `collapse_at_s` as a separate compact pending state. Backend lifecycle validation covers the generated payload but does not constitute automated browser-conformance testing.


## Safe summary claim

The repository demonstrates a reproducible, auditable workflow in which a trained question-worthiness model helps rank pre-labelled broadcast states, deterministic policies turn selected states into prediction contracts, and later verified broadcast evidence resolves those contracts during a replay-based viewer experience.
