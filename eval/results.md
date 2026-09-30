# Model evaluation

The selector is an L2-regularised logistic regression trained to predict `worth_asking`. It does **not** predict survival probability. FFWS 2025 Game 4 is held out completely and is used only after the model is frozen.

## Dataset

- Training matches: ffws2025-g1, ffws2025-g2, ffws2025-g3, ffws2025-g5, ffws2025-g6, ffws2025-g7
- Training rows: 178
- Positive question-worthy rows: 32
- Positive prevalence: 0.180
- Features: `active`, `knocked`, `teams_remaining`, `players_remaining`, `recent_active_drop`, `match_progress`

## Leave-one-match-out validation

| Held-out match | Balanced accuracy | Average precision | Brier score |
|---|---:|---:|---:|
| ffws2025-g1 | 0.952 | 1.000 | 0.035 |
| ffws2025-g2 | 0.935 | 0.643 | 0.102 |
| ffws2025-g3 | 0.985 | 1.000 | 0.024 |
| ffws2025-g5 | 0.856 | 0.890 | 0.073 |
| ffws2025-g6 | 0.900 | 0.817 | 0.124 |
| ffws2025-g7 | 0.938 | 1.000 | 0.029 |

- Mean balanced accuracy: **0.928**
- Mean average precision: **0.892**
- Pooled model Brier score: **0.064**
- Fold-prevalence baseline Brier score: **0.150**

## Ranking comparison

Across 68 labelled timestamp groups:

- Top model-ranked row labelled worth asking: **42.6%**
- Naive fewest-active row labelled worth asking: **42.6%**

This ranking comparison is descriptive because the historical sheets contain a sparse, human-preselected subset rather than all 12 squads at every timestamp. It must not be presented as conclusive ranking lift.

The two top-positive rates are identical, so this evaluation does not establish that the current model outperforms the naive rule. It demonstrates the reproducible model-training and held-out-scoring workflow only.

## Standardised model coefficients

| Feature | Coefficient |
|---|---:|
| `active` | -1.6355 |
| `knocked` | +1.2665 |
| `teams_remaining` | +0.6268 |
| `players_remaining` | -0.3420 |
| `recent_active_drop` | +0.7198 |
| `match_progress` | +0.9398 |

Positive coefficients increase similarity to historical states labelled worth asking; negative coefficients decrease it. These values are not causal explanations.

## Held-out Game 4 output

The frozen selector scores Game 4 states after `reconstruct_game4.py` derives past-only change and kill-feed context. `generate_game4.py` then applies observation confidence, recent-event, tournament-context, forward settleability and spacing rules. Outcomes are computed separately from later evidence and never enter the model features.
