# Reported results

The tables on this page are transcribed into
[`results/reported_metrics.json`](../results/reported_metrics.json). Accuracy is
the fraction of correct predictions. Macro-F1 is the unweighted mean of
per-class F1. The class order is `positive`, `negative`, `neutral`.

These experiments have release status `reported, not rerun during release
preparation`. The repository provides the recorded values, code, and tests; it
does not claim that the complete GPU experiments were independently reproduced
for this release.

## Main Restaurant comparison (Table I)

The main SemEval-2014 Restaurant comparison reports the mean and standard
deviation over three random seeds.

| Model | Accuracy | Macro-F1 | Delta Macro-F1 |
| --- | ---: | ---: | ---: |
| Text-Transformer | 0.8022 +/- 0.0021 | 0.6914 +/- 0.0101 | -0.0566 |
| Text-GT | 0.8067 +/- 0.0078 | 0.7023 +/- 0.0165 | -0.0457 |
| BERT-SPC | 0.8389 +/- 0.0101 | 0.7588 +/- 0.0156 | +0.0108 |
| AEN-BERT | 0.8034 +/- 0.0118 | 0.6826 +/- 0.0186 | -0.0654 |
| LCF-BERT | 0.8332 +/- 0.0078 | 0.7482 +/- 0.0142 | +0.0001 |
| TextGT-BERT | 0.8383 +/- 0.0176 | 0.7480 +/- 0.0395 | - |
| TextGT-BERT + fact | 0.8406 +/- 0.0209 | 0.7522 +/- 0.0445 | +0.0042 |
| TextGT-BERT + comparison | 0.8394 +/- 0.0172 | 0.7499 +/- 0.0384 | +0.0019 |
| TextGT-BERT + intensity | 0.8388 +/- 0.0194 | 0.7488 +/- 0.0424 | +0.0008 |
| TextGT-BERT + fact+comparison | 0.8418 +/- 0.0205 | 0.7541 +/- 0.0434 | +0.0060 |
| TextGT-BERT + all | **0.8424 +/- 0.0223** | **0.7547 +/- 0.0465** | **+0.0067** |

Within the TextGT-BERT variants, all three residuals give the highest reported
accuracy and Macro-F1. Across the entire table, the full adapter has the highest
accuracy, while BERT-SPC has the highest Macro-F1. The reported standard
deviations are large relative to the gains, and no significance test was
independently verified during release preparation.

Table I reports the TextGT-BERT baseline accuracy as `0.8383`; the module
ablation below reports `0.8382`. The repository preserves both values exactly as
recorded instead of silently reconciling the difference.

## Module ablation (Table II)

Every row uses the same TextGT-BERT checkpoints and test split. The backbone was
not retrained for each residual combination. `Changed`, `Fixed`, and `Broken`
are reported checkpoint means.

| Strategy | Base Acc. | Acc. | Base Macro-F1 | Macro-F1 | Delta F1 | Changed | Fixed | Broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact | 0.8382 | 0.8406 | 0.7480 | 0.7522 | +0.0042 | 4.33 | 3.33 | 0.67 |
| comparison | 0.8382 | 0.8394 | 0.7480 | 0.7499 | +0.0019 | 1.33 | 1.33 | 0.00 |
| intensity | 0.8382 | 0.8388 | 0.7480 | 0.7488 | +0.0008 | 2.67 | 1.33 | 0.67 |
| fact+comparison | 0.8382 | 0.8418 | 0.7480 | 0.7541 | +0.0060 | 5.67 | 4.67 | 0.67 |
| all | 0.8382 | **0.8424** | 0.7480 | **0.7547** | **+0.0067** | 8.00 | 6.00 | 1.33 |

The fact branch is the strongest individual residual at `+0.0042` Macro-F1.
Comparison has a smaller gain but no broken samples in the reported checkpoint
mean. Intensity is the weakest individual branch. Combining branches gives the
largest gains: the full adapter fixes 6.00 samples and breaks 1.33 on average,
which also shows that the method can introduce errors rather than acting as a
monotonic correction.

## Baseline error analysis (Table III)

The TextGT-BERT baseline has 151 reported Restaurant errors.

| Gold -> predicted | Count | Percentage |
| --- | ---: | ---: |
| neutral -> positive | 51 | 33.8% |
| positive -> neutral | 30 | 19.9% |
| positive -> negative | 20 | 13.2% |
| negative -> neutral | 20 | 13.2% |
| neutral -> negative | 16 | 10.6% |
| negative -> positive | 14 | 9.3% |

Neutral-to-positive is the largest error pair. This matches the purpose of the
fact branch: objective descriptions and menu-style listings can be mistaken for
positive opinions.

The report also assigns errors to qualitative buckets:

| Error bucket | Count |
| --- | ---: |
| far context | 114 |
| multi aspect | 112 |
| long sentence | 41 |
| negation | 40 |
| contrast | 24 |
| intensifier | 16 |
| comparative | 13 |
| plain | 13 |
| hedge | 9 |
| weak praise | 5 |

The bucket counts sum to more than 151. They should therefore be treated as
overlapping or non-exclusive until the raw bucket assignments are audited. The
table still shows that far-context and multi-aspect attribution are frequent
failure modes, but it does not support adding the bucket counts as if they were
disjoint classes.

## Cross-domain generalization (Table IV)

Each domain uses a phrase prior reconstructed from that domain's training split.
This tests training-domain adaptation, not zero-shot transfer.

| Strategy | Restaurant | Laptop | Twitter |
| --- | ---: | ---: | ---: |
| baseline | 0.7480 | 0.7701 | 0.7450 |
| fact | 0.7522 | 0.7676 | 0.7463 |
| comparison | 0.7499 | 0.7724 | 0.7439 |
| intensity | 0.7488 | 0.7703 | 0.7447 |
| all | 0.7547 | 0.7714 | 0.7449 |

Restaurant shows the clearest improvement. On Laptop, comparison is the best
single branch, while the full adapter gains only `0.0013` Macro-F1 over the
baseline. Twitter changes are close to zero and the full adapter is slightly
below the baseline. These mixed results are evidence of domain sensitivity, not
universal cross-domain improvement.

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

This is adapter-compatibility evidence, not a multi-seed model leaderboard.

## Limitations

- The full Restaurant gain is `+0.0067` Macro-F1. It is useful for analyzing
  specific pragmatic errors but remains below `0.01`; no state-of-the-art or
  statistical-significance claim is made.
- The intensity branch can use an optionally LLM-annotated phrase prior. The LLM
  is not called for validation or test inference, but prior construction can add
  an external resource dependency. The public builder also supports a fully
  offline path.
- Cross-domain gains are mixed. Each domain rebuilds its prior from its own
  training split, so the reported experiment is not zero-shot generalization.
- The current residuals use handcrafted cues, scope rules, and fixed weights.
  The `Broken` column confirms that the adapter can over-correct some samples.
- The release records reported experiments but does not include a fresh GPU
  rerun. Restaurant uses three selected checkpoints; Laptop and Twitter are
  preliminary single-checkpoint results.
- Table I and Table II differ by `0.0001` in the reported TextGT-BERT baseline
  accuracy. Error-bucket counts are also non-exclusive or otherwise require a
  raw-assignment audit. Both reporting boundaries are kept visible here.
