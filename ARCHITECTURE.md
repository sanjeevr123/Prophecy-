# Prophecy architecture

## 1. Purpose and system boundary

Prophecy is a prediction companion for Free Fire esports broadcasts. Its core responsibility is to choose which broadcast-visible state is worth turning into a viewer prediction.

The intended production system ingests a live broadcast. This repository uses a recorded FFWS 2025 Game 4 video and pre-labelled observations to make the same downstream behavior deterministic and auditable. The CSV inputs stand in for future OCR and video-event extraction; no current script decodes the video or performs OCR.

Three invariants govern the design:

1. **Broadcast evidence only.** No game-server telemetry, private API or off-camera state is assumed.
2. **Last verified, never hidden current.** A squad state is stored with the timestamp at which the broadcast verified it.
3. **Selection before settlement.** Later evidence may answer a selected question but may not become a model feature or composite-ranking input.

## 2. Repository-level data flow

```text
TRAINING
data/training/ffws2025_g{1,2,3,5,6,7}_labels.csv
        |
        v
pipeline/train_selector.py
        |
        +--> data/out/question_selector.joblib
        +--> data/out/model_metrics.json
        +--> data/out/training_oof_predictions.csv

HELD-OUT GAME 4
game4_observations.csv + events.csv + panel_visible.csv
        |
        v
pipeline/reconstruct_game4.py
        |
        +--> data/out/game4_features.csv
        +--> data/out/data_quality.json
        |
        v
pipeline/score_game4.py + frozen model
        |
        +--> data/out/scored_game4.csv
        |
        v
standings.csv + game4_milestones.csv
        |
        v
pipeline/generate_game4.py
        |
        +--> data/out/candidates.csv
        +--> data/out/questions.json       schema v5, seven cards
        |
        v
pipeline/validate_backend.py
        |
        +--> data/out/backend_validation.json

VIEWER
data/out/questions.json                   schema v5, seven generated cards
        |
        v
app/app.js + app/index.html + app/styles.css
        |
        v
video-synchronised browser experience

app/questions.json                        schema v2, legacy unused reference
```

The backend and viewer are connected at the payload boundary: `pipeline/run.py` writes `data/out/questions.json`, and `app/app.js` fetches that schema-v5 artifact. The pipeline still does not edit files under `app/`. The older `app/questions.json` is retained only as an explicitly marked legacy reference.

## 3. Data sources and time bases

### Historical training labels

Each row in `data/training/` represents one squad's broadcast-observable state at a labelled timestamp. The row contains the player state, global lobby state, recent roster change, match progress, a binary `worth_asking` label and a confidence level.

The historical sheets are sparse human-selected examples rather than full 12-squad snapshots at every timestamp.

### Held-out Game 4 inputs

| File | Time basis | Meaning |
|---|---|---|
| `data/demo/game4_observations.csv` | In-game match clock | Verified squad snapshots and global counters |
| `data/labels/events.csv` | `video.currentTime` | Squad-attributed knock, elimination and revival labels |
| `data/labels/panel_visible.csv` | `video.currentTime` | Interval labels for scoreboard availability |
| `data/labels/standings.csv` | Before Game 4 | Tournament rank and points context |
| `data/demo/game4_milestones.csv` | Both clocks | Frame-reviewed showcase and endgame milestones |

The replay begins at in-game `00:29`, so the configured conversion is:

```text
game_time_s  = video_time_s + 29
video_time_s = game_time_s - 29
```

All model features and candidate anchors use game time. All `*_at_s` fields consumed by a browser use video time. `anchor_game_s` is retained in the output for auditing.

## 4. Training and evaluation

`pipeline/train_selector.py` loads six historical games and rejects the run if Game 4 appears in the training directory.

The model is an L2-regularised logistic regression with balanced class weighting. Its target is `worth_asking`, using:

- `active`
- `knocked`
- `teams_remaining`
- `players_remaining`
- `recent_active_drop`
- `match_progress`

The fitted package records the model, feature order, target, training matches and evaluation metadata. The score measures similarity to historical question-worthy states; it is not calibrated or presented as survival probability.

Evaluation uses leave-one-match-out cross-validation. The current dataset contains 178 rows and 32 positives. Mean balanced accuracy is 0.928 and mean average precision is 0.892. The model and naive fewest-active ranking both obtain a 42.6% top-positive rate on the sparse timestamp groups, so no ranking lift over that rule is claimed.

## 5. Game 4 reconstruction

`pipeline/reconstruct_game4.py` validates and enriches the 45 held-out observations. For each observation it derives only information available at or before the anchor:

- previous observed active and knocked counts;
- previous observation timestamp and observation gap;
- roster deterioration within the configured state-change window;
- squad-attributed events in the previous 45 seconds;
- most recent attributed event and event intensity;
- panel visibility and trailing panel duty cycle;
- observation-confidence weight; and
- normalised match progress and video timestamp.

Game 4 contains 22 observation timestamps and averages approximately two labelled squads per timestamp. There is no complete 12-squad timestamp. Consequently, the prototype ranks the supplied labelled states; it does not continuously compare every squad in the lobby.

The event labels do not include player identity or credited killer. Exact duplicate squad/event/timestamp rows can represent multiple affected players but are indistinguishable from duplicate annotation. Later verified snapshots therefore take precedence when they correct an incomplete kill-feed reconstruction.

## 6. Frozen-model scoring

`pipeline/score_game4.py` loads `question_selector.joblib`, verifies the held-out match ID and required features, then applies `predict_proba` without fitting or tuning on Game 4.

The result, `data/out/scored_game4.csv`, preserves the source observation and derived evidence alongside `worth_score`.

## 7. Eligibility, ranking and arbitration

`pipeline/generate_game4.py` builds a candidate audit row for every scored Game 4 observation.

An ML candidate must:

- meet the configured model threshold;
- have non-low observation confidence;
- be panel-verified at its anchor;
- meet the trailing panel-duty requirement;
- contain one to three active players;
- follow a newly verified active-player drop or increase in knocked players; and
- fit before the curated endgame sequence.

The composite rank is:

```text
observation_confidence * (
    0.70 * model_worthiness
  + 0.12 * recent_event_signal
  + 0.10 * tournament_relevance
  + 0.08 * past_panel_availability
)
```

Tournament relevance uses only the standings before Game 4. It is deterministic and separate from the model.

The arbitration step searches valid candidate combinations and prioritises, in order:

1. maximum card count up to the configured ML limit;
2. distinct contract types;
3. early, middle and late match coverage; and
4. total composite score.

It also enforces minimum anchor spacing and at most one ML-selected question per squad.

Future question answers are not model features or ranking inputs. The recorded prototype does use the known final-three milestone as a scheduling boundary so ordinary cards do not collide with its curated endgame sequence. This is replay orchestration, is declared in `questions.json`, and would be replaced by current lobby state and operational deadlines in production.

## 8. Question sources and contracts

The schema-v5 backend set has three explicitly identified sources:

| `selection_method` | Count | Construction |
|---|---:|---|
| `ml_ranked_diversity_policy` | 3 | Ranked state plus deterministic contract template |
| `verified_broadcast_showcase_trigger` | 2 | Frame-reviewed FX and GOW milestone labels |
| `deterministic_verified_milestone` | 2 | Complete final-four and final-three overlays |

ML-selected state contracts include:

- `recovery`: rebuild to a declared active-player threshold;
- `two_player_hold`: retain both active players;
- `roster_hold`: retain the current reduced roster for a shorter window;
- `last_stand`: keep at least one active player in a late-game window; and
- `survival`: retain at least one active player over the standard horizon.

Question wording is deterministic. The model selects question-worthiness; it does not generate free-form copy or predict the correct option.

## 9. Settlement

Settlement runs only after selection. For an ML card, the generator starts from the anchor snapshot and walks later same-squad events and snapshots up to the contract boundary plus the configured grace period.

- State-changing kill-feed events update the reconstructed active/knocked counts.
- A verified later snapshot overrides event reconstruction at the same timestamp.
- A verified complete remaining-roster milestone can establish that a squad is absent.
- A terminal zero-player state may settle early, while its result is displayed at the declared contract boundary.
- If no reliable evidence appears by the end of the grace period, the question is emitted as `void` with no correct option.

The schema-v5 lifecycle is:

```text
show_at_s
  -> close_at_s             answer window closes
  -> collapse_at_s          intended compact pending state
  -> window_end_s           contract boundary
  -> resolve_at_s           verified result or void
  -> result_hide_at_s       result display ends
```

The generator validates ordering, duration bounds, option integrity, unique IDs and wording, outcome variety and non-overlap.

## 10. Viewer architecture

The viewer is a dependency-free static application:

- `app/index.html` contains the remote release-video source and interface structure;
- `app/styles.css` renders the responsive video overlay, prediction card, session score and rewards; and
- `app/app.js` follows `video.currentTime`, loads questions, renders options and maintains in-memory session state.

The current viewer loads the schema-v5 `data/out/questions.json`, containing seven generated cards. It accepts arbitrary option counts and normalises optional older field names for compatibility.

The viewer uses `show_at_s`, `close_at_s`, `window_end_s`, `resolve_at_s` and `result_hide_at_s`. It awards points only when `resolve_at_s` is reached. The schema-v5 `collapse_at_s` field is not yet rendered as a separate compact pending state, so an answered card remains expanded until its next lifecycle transition.

No outcome probability is inferred when the backend omits one. The result view currently includes a deterministic illustrative crowd percentage; that value is UI simulation rather than measured audience data and is disclosed accordingly.

The UI uses “live” language to demonstrate the intended product experience, but playback is a recorded replay. Session points, streaks and rewards are local, illustrative and reset on reload.

## 11. Validation and reproducibility

`pipeline/validate_backend.py` performs cross-artifact checks covering:

- Game 4 training isolation;
- model target and no-future-feature metadata;
- input-mode and replay-scheduling disclosures;
- source time-base normalisation;
- question-to-candidate lineage;
- threshold, roster-transition and panel-evidence gates;
- contract horizons and reconstructed answers;
- question source composition;
- lifecycle order, video bounds and non-overlap;
- void semantics and outcome diversity; and
- the final `questions.json` SHA-256 hash.

The current output passes 121 checks. A clean isolated run reproduces all generated artifacts byte-for-byte under the tested Python 3.11 environment.

## 12. Production extension

The natural production boundary is ahead of `reconstruct_game4.py`:

```text
live video
  -> OCR and overlay detection
  -> attributed event extraction
  -> timestamped last-verified state store
  -> the existing feature, selection, contract and settlement stages
```

Production readiness would additionally require full-lobby observation coverage, player-level event identities, deduplication, confidence calibration, latency monitoring, stronger ranking baselines and evaluation across tournaments.
