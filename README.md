# Prophecy

Prophecy is a model-informed prediction companion for Free Fire esports broadcasts. It turns consequential, uncertain changes in the broadcast-visible tournament state into short prediction prompts, giving viewers a reason to follow more than the fight currently selected by the observer.

**Prototype:** [yothisisash.github.io/techgarena/app](https://yothisisash.github.io/techgarena/app/)

## Why Prophecy

A Free Fire match can contain 12 squads of four players on the same island. The broadcast can show only one perspective at a time, while the tournament result depends on points accumulated across multiple games. A squad outside the camera view can therefore lose players, approach a standings threshold, or become relevant to the overall result without the audience immediately understanding why it matters.

Prophecy reconstructs the latest **verified** state available from the same broadcast the viewer receives. When an observable squad state is consequential, uncertain and reliably settleable, it creates a prediction card such as:

> **SURVIVAL WATCH**
>
> Six squads remain and GOW are present in the latest verified broadcast state.
>
> Will GOW survive the next 15 seconds?

The differentiator is the selection decision: deciding what is worth asking and when. Prophecy does not control the camera, reveal off-camera footage or claim access to hidden game state.

## Prototype scope

The intended product operates alongside a live broadcast. This repository demonstrates that experience deterministically with a 970-second recording of FFWS 2025 Game 4.

| Layer | Current repository | Intended production system |
|---|---|---|
| Broadcast | Recorded Game 4 replay | Live tournament stream |
| State ingestion | Pre-labelled scoreboard, player-bar and kill-feed observations | OCR and video-event extraction |
| State model | Timestamped, broadcast-visible snapshots | Continuously updated last-verified state |
| Selection | Frozen classifier plus evidence and diversity rules | The same decision layer operating as observations arrive |
| Settlement | Precomputed from later verified replay evidence | Resolved from later live broadcast evidence |

The current pipeline does **not** run OCR or extract labels directly from `match.mp4`. Its CSV inputs emulate the structured observations that a future ingestion service would emit.

## End-to-end workflow

```text
Historical Games 1, 2, 3, 5, 6 and 7
  labelled broadcast-visible squad states + worth_asking
                              |
                              v
Train L2 logistic-regression question-worthiness model
                              |
                              v
Pre-labelled Game 4 snapshots + kill feed + panel visibility + standings
                              |
                              v
Reconstruct past-only features and apply the frozen model
                              |
                              v
Evidence gates + composite ranking + showcase diversity policy
                              |
                              v
Deterministic question contracts
                              |
                              v
Later verified evidence resolves the answer, or the question is voided
                              |
                              v
data/out/questions.json + candidate and validation audit trails
```

The backend currently generates seven questions:

| Source | Count | Role |
|---|---:|---|
| ML-ranked state | 3 | Model-ranked recovery, hold or last-stand situations |
| Verified showcase trigger | 2 | Frame-reviewed FX and GOW prompts retained for contract variety |
| Deterministic milestone | 2 | Complete final-four and final-three broadcast states |

Every question records its provenance in `selection_method`. Question language is produced by deterministic templates; the model decides question-worthiness, not free-form wording or the future answer.

## What the system knows

Prophecy uses only information visible in the broadcast:

- squad active and knocked-player counts when the relevant overlay is readable;
- the global teams- and players-remaining counters;
- tournament standings shown before Game 4;
- attributed knock, elimination and revival events from the kill feed;
- match time, match progress and panel visibility; and
- later broadcast-visible evidence used to settle a question.

It does not know or claim:

- player locations, proximity or off-camera fights;
- health, inventory, weapons or tactical intent;
- current state after the most recent verified observation;
- game-server telemetry or a Garena API; or
- that a surfaced state is the most important fight on the island.

The governing rule is: **store the last verified observation with its timestamp; never present it as hidden current state.**

## The model's role

[`pipeline/train_selector.py`](pipeline/train_selector.py) trains an L2-regularised logistic regression on 178 labelled states from six historical games. FFWS 2025 Game 4 is rejected if it appears in training and is used only after the final model is fitted.

The target is `worth_asking`. Its six features are:

- `active`
- `knocked`
- `teams_remaining`
- `players_remaining`
- `recent_active_drop`
- `match_progress`

`model_worthiness_score` means that a state resembles historical states labelled as engaging prediction opportunities. It is **not** a survival probability, win probability or fan-vote estimate.

The candidate rank also uses recent kill-feed activity, pre-match tournament context, observation confidence and past panel availability. Actual future results are evaluated only after candidate selection. A replay-only scheduling guard uses the known endgame milestone time to keep ordinary cards out of the curated final sequence; that guard is disclosed in the output metadata.

The evaluation demonstrates a reproducible training and held-out scoring workflow. It does not establish that the current model outperforms a simple rules baseline: the model and naive fewest-active ranking both achieve a 42.6% top-positive rate on the sparse historical timestamp groups.

## Repository structure

```text
app/
  index.html                 Static viewer and video player
  app.js                     Video-synchronised card and session logic
  styles.css                 Responsive overlay and dashboard styling
  questions.json             Legacy six-card reference; retained but not loaded

data/
  training/                  Historical Games 1, 2, 3, 5, 6 and 7 labels
  demo/                      Held-out Game 4 observations and milestones
  labels/                    Kill feed, panel visibility and standings labels
  out/                       Model, features, questions and audit artifacts
  raw/match.mp4              Full-resolution local source; Git-ignored

pipeline/
  config/ffws2025.yaml       Match clocks, thresholds, windows and policies
  train_selector.py          Training and leave-one-match-out evaluation
  reconstruct_game4.py       Past-only Game 4 feature reconstruction
  score_game4.py             Frozen-model inference
  generate_game4.py          Eligibility, ranking, contracts and settlement
  validate_backend.py        Cross-artifact and lifecycle validation
  run.py                     End-to-end pipeline runner

eval/results.md              Human-readable model evaluation
ARCHITECTURE.md              Detailed system and data-flow design
DISCLOSURE.md                Claims, provenance and prototype limitations
```

## Run the viewer

The frontend is static and requires no build step. From the repository root:

```powershell
python -m http.server 8000
```

Open [http://localhost:8000/app/](http://localhost:8000/app/).

The player streams the web-optimised match video from the repository's `DemoVideo` GitHub Release, so an internet connection is required for playback.

### Frontend question source

The current viewer loads the generated [`data/out/questions.json`](data/out/questions.json) payload directly. The relative fetch path in `app/app.js` includes a cache-busting query parameter, but the underlying source remains the schema-v5 seven-question backend artifact.

[`app/questions.json`](app/questions.json) is a retained schema-v2 reference from the earlier six-card showcase. It is explicitly marked `legacy_unused` and is not read by the current application. Regenerating the backend can therefore change the questions and timings presented by the viewer after a reload.

## Regenerate the backend

Python 3.11 is the tested runtime.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline\run.py
```

The runner executes:

1. `train_selector.py`
2. `reconstruct_game4.py`
3. `score_game4.py`
4. `generate_game4.py`
5. `validate_backend.py`

A successful run ends with `Backend validation passed: 121 checks` and writes the canonical backend payload to `data/out/questions.json`.

## Outputs and auditability

| Artifact | Purpose |
|---|---|
| `data/out/question_selector.joblib` | Fitted model, features and training metadata |
| `data/out/model_metrics.json` | Grouped validation metrics and coefficients |
| `data/out/training_oof_predictions.csv` | Out-of-fold score for every historical row |
| `data/out/game4_features.csv` | Reconstructed held-out Game 4 features |
| `data/out/scored_game4.csv` | Frozen-model score for every Game 4 observation |
| `data/out/candidates.csv` | Eligibility, ranking, rejection and settlement trace |
| `data/out/questions.json` | Schema-v5 question payload |
| `data/out/data_quality.json` | Input counts, time bases and known limitations |
| `data/out/backend_validation.json` | Validation results and output hash |
| `eval/results.md` | Human-readable evaluation summary |

Current verified figures:

- 6 training matches, 178 labelled rows and 32 positive `worth_asking` labels;
- 45 held-out Game 4 squad states across 22 observation timestamps;
- 144 labelled Game 4 kill-feed events;
- mean leave-one-match-out balanced accuracy of 0.928;
- mean leave-one-match-out average precision of 0.892; and
- 121 backend consistency and lifecycle checks.

These metrics describe the current labelled prototype dataset. They are not evidence of production-scale accuracy, complete 12-squad coverage or causal model explanations.

## Further documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) explains data flow, time bases, selection, settlement and the frontend boundary.
- [`DISCLOSURE.md`](DISCLOSURE.md) defines the claims the prototype can and cannot support.
- [`eval/results.md`](eval/results.md) records the current model evaluation.
