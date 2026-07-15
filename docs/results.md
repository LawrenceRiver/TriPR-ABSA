# Reported results

All values and descriptions on this page come from
[`results/reported_metrics.json`](../results/reported_metrics.json). Accuracy is
correct predictions divided by all test examples. Macro-F1 is the unweighted mean
of per-class F1. The class order is `positive`, `negative`, `neutral`. These
experiments have release status `reported, not rerun during release preparation`.

## SemEval-2014 Restaurant

The primary table is reported from three selected checkpoints and aggregated as
their mean on the test split.

| Strategy | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| baseline | 0.8382 | 0.7480 |
| fact | 0.8406 | 0.7522 |
| comparison | 0.8394 | 0.7499 |
| intensity | 0.8388 | 0.7488 |
| fact+comparison | 0.8418 | 0.7541 |
| all | 0.8424 | 0.7547 |

> Reported result from 3 selected checkpoints; not independently rerun during release preparation and not a state-of-the-art claim.

## SemEval-2014 Laptop

This test table uses a single checkpoint with no cross-checkpoint aggregation.

| Strategy | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| baseline | 0.8022 | 0.7701 |
| fact | 0.7991 | 0.7676 |
| comparison | 0.8038 | 0.7724 |
| intensity | 0.8022 | 0.7703 |
| fact+comparison | 0.8006 | 0.7700 |
| all | 0.8022 | 0.7714 |

> Single-checkpoint preliminary cross-domain evidence with mixed, small changes; not evidence of universal improvement and not a state-of-the-art claim.

## Twitter ABSA

This test table uses a single checkpoint with no cross-checkpoint aggregation.

| Strategy | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| baseline | 0.7578 | 0.7450 |
| fact | 0.7592 | 0.7463 |
| comparison | 0.7563 | 0.7439 |
| intensity | 0.7548 | 0.7447 |
| fact+comparison | 0.7578 | 0.7452 |
| all | 0.7548 | 0.7449 |

> Single-checkpoint preliminary cross-domain evidence: fact is best; all is approximately neutral in macro-F1 while accuracy decreases; not evidence of universal improvement and not a state-of-the-art claim.

## SemEval-2014 Restaurant across backbones

These are reported model evaluations on the test split; most rows use one
checkpoint.

| Backbone | Base accuracy | Base Macro-F1 | Best strategy | Residual accuracy | Residual Macro-F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| text-gt-bert | 0.8647 | 0.8088 | all | 0.8702 | 0.8171 |
| text-gt | 0.8186 | 0.7419 | fact+comparison | 0.8257 | 0.7617 |
| text-tg | 0.8195 | 0.7260 | fact+comparison | 0.8329 | 0.7610 |
| text-gin | 0.8034 | 0.7339 | fact+comparison | 0.8097 | 0.7530 |
| text-transformer | 0.8097 | 0.7153 | all | 0.8195 | 0.7474 |
| gnn-transformer | 0.8088 | 0.7046 | all | 0.8275 | 0.7433 |
| transformer-gnn | 0.8070 | 0.7284 | all | 0.8123 | 0.7494 |
| parallel-gt | 0.8275 | 0.7514 | fact+comparison | 0.8365 | 0.7737 |

> Adapter-compatibility evidence, not a multi-seed model leaderboard.
