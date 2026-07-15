[English](README.md) | [简体中文](README.zh-CN.md)

# TriPR-ABSA

**Tri-Branch Pragmatic Residual Adaptation for Aspect-Based Sentiment Analysis**

TriPR-ABSA adds three lightweight pragmatic residual branches to an existing
three-class ABSA model. It leaves the backbone unchanged and corrects its logits
for fact-opinion boundaries, comparison direction, and sentiment intensity.

[![CI](https://github.com/LawrenceRiver/TriPR-ABSA/actions/workflows/ci.yml/badge.svg)](https://github.com/LawrenceRiver/TriPR-ABSA/actions/workflows/ci.yml)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-0b77be)
![Python 3.9 and 3.10](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?logo=python&logoColor=white)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)
[![MIT software license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream TextGT](https://img.shields.io/badge/upstream-shuoyinn%2FTextGT-181717?logo=github)](https://github.com/shuoyinn/TextGT)
[![Base paper DOI](https://img.shields.io/badge/base%20paper-10.1609%2Faaai.v38i17.29911-2f6f9f)](https://doi.org/10.1609/aaai.v38i17.29911)

## Upstream relationship

This repository is an independent research fork of
[shuoyinn/TextGT](https://github.com/shuoyinn/TextGT). TextGT-BERT is the main
baseline used in the reported experiments; TriPR-ABSA is the residual method
built on top of its logits. The project is maintained separately and is not an
official TextGT release. Upstream code and paper attribution are retained in
[NOTICE](NOTICE).

## Architecture

![TriPR-ABSA architecture](assets/architecture.png)

The adapter receives backbone logits and one parsed sample. The three residual
branches run in sequence, and the composer adds their corrections at logit
level. No backbone layer is replaced or retrained by this package. See
[docs/method.md](docs/method.md) for the interface, equations, and error
contracts.

## Method

TriPR-ABSA uses the class order `positive`, `negative`, `neutral`.

### Fact

The fact-opinion branch looks for objective listings and descriptions around the
current aspect. When the baseline is positive but the evidence is primarily
factual, the branch shifts probability toward neutral.

### Comparison

The comparison branch determines whether the current aspect wins or loses an
explicit comparison. It handles direction words and external references before
adjusting the positive and negative logits.

### Intensity

The intensity branch matches phrases from a validated train-split prior. Phrase
scope, local negation, token distance, and the current aspect all affect the
residual. Without a prior, this branch is a no-op.

### Composer

The composer applies selected branches in the order `fact`, `comparison`,
`intensity`. Each branch sees probabilities derived from the logits produced by
the previous branch. Public entry points are `apply_pragmatic_residual`,
`apply_batch`, and `load_prior`.

## Reported Restaurant results

![Macro-F1 comparison of residual strategies](assets/figures/strategy-comparison.png)

The primary SemEval-2014 Restaurant table reports the mean of three selected
checkpoints. Machine-readable values are stored in
[results/reported_metrics.json](results/reported_metrics.json).

| Strategy | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Baseline | 0.8382 | 0.7480 |
| Fact | 0.8406 | 0.7522 |
| Comparison | 0.8394 | 0.7499 |
| Intensity | 0.8388 | 0.7488 |
| Fact + comparison | 0.8418 | 0.7541 |
| All residuals | 0.8424 | 0.7547 |

<details>
<summary>Module ablation and limitations</summary>

All variants use the same TextGT-BERT checkpoints and test split. `Changed`,
`Fixed`, and `Broken` are reported checkpoint means.

| Strategy | Accuracy | Macro-F1 | Delta F1 | Changed | Fixed | Broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fact | 0.8406 | 0.7522 | +0.0042 | 4.33 | 3.33 | 0.67 |
| Comparison | 0.8394 | 0.7499 | +0.0019 | 1.33 | 1.33 | 0.00 |
| Intensity | 0.8388 | 0.7488 | +0.0008 | 2.67 | 1.33 | 0.67 |
| Fact + comparison | 0.8418 | 0.7541 | +0.0060 | 5.67 | 4.67 | 0.67 |
| All residuals | 0.8424 | 0.7547 | +0.0067 | 8.00 | 6.00 | 1.33 |

Fact is the strongest single branch. Comparison changes fewer samples and has
no broken cases in the reported mean. The full adapter gives the best combined
result but can still over-correct individual samples.

The reported gain is modest (`+0.0067` Macro-F1), cross-domain results are mixed,
and the current method uses handcrafted cues and fixed residual weights. The
results were not independently rerun during release preparation. The full
tables, baseline error distribution, reporting inconsistencies, and limitation
notes are in [docs/results.md](docs/results.md).

</details>

<details>
<summary>Cross-domain and multi-backbone figures</summary>

![Cross-domain Macro-F1](assets/figures/cross-domain.png)

![Residual compatibility across backbones](assets/figures/multi-backbone.png)

</details>

These numbers were not independently rerun during release preparation and are
not presented as a state-of-the-art claim. Dataset-specific limits and the full
tables are documented in [docs/results.md](docs/results.md).

## Installation

The residual package supports Python 3.9 and 3.10.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install `requirements-dev.txt` to run the checks. Full upstream model training
uses `requirements.txt` and separately obtained datasets and model resources.
This repository does not redistribute datasets or pretrained checkpoints.
Follow the [upstream preparation instructions](https://github.com/shuoyinn/TextGT#priliminaries)
for the TextGT baseline.

The upstream code and datasets also credit
[DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA),
[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), and
[CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA). Compatible
preprocessed datasets are available from
[SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA). Source preprocessing
requires [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/); non-BERT
training also uses [Stanford GloVe](https://nlp.stanford.edu/projects/glove/).

## Quick start

This example enables the comparison branch and does not require a phrase prior.

```python
import torch

from pragmatic_residual import apply_pragmatic_residual

sample = {
    "text_list": ["I", "have", "had", "better", "food", "elsewhere", "."],
    "aspect": "food",
    "aspect_post": [4, 5],
}
logits = torch.tensor([1.2, 0.3, 0.1])

adjusted, details = apply_pragmatic_residual(
    logits,
    sample,
    modules=("comparison",),
    return_details=True,
)

print(adjusted)
print(details["actions"])
```

## Configuration

Pass any subset of `("fact", "comparison", "intensity")` through `modules`.
With `modules=None`, all three branches run in their default order. The `prior`
argument accepts a validated mapping or a JSON path produced by
`scripts/build_phrase_prior.py`. The intensity branch is disabled when no prior
is supplied.

The prior builder is offline by default. `DEEPSEEK_API_KEY` is read only when
`--provider deepseek` is selected for optional remote phrase annotation. The
key is not needed for inference, tests, visualization, or the offline builder.

## Prior construction

The builder reads the training split only. Its default provider makes no network
request:

```bash
python scripts/build_phrase_prior.py \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

Optional DeepSeek annotation sends train-derived phrase candidates and aggregate
counts, not complete dataset rows or labels. Supply the key through the process
environment; never place it in a source file or configuration committed to Git.

```bash
export DEEPSEEK_API_KEY="your-key"
python scripts/build_phrase_prior.py \
  --provider deepseek \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

Use of the remote provider is subject to its privacy policy and service terms.

## Reproducibility

[docs/reproducibility.md](docs/reproducibility.md) lists CPU checks, upstream GPU
baseline commands, data preparation links, and the offline prior command. The
release keeps reported metrics in JSON rather than inferring them from figures.

## Contributors

TriPR-ABSA is maintained by Lawrence River, QCYTSN, and jason0917-eng. Roles are
listed in [AUTHORS.md](AUTHORS.md). See [CONTRIBUTING.md](CONTRIBUTING.md) before
submitting a change and [SECURITY.md](SECURITY.md) for private vulnerability
reporting and credential handling.

## Citation

[CITATION.cff](CITATION.cff) contains the TriPR-ABSA software metadata and the
upstream TextGT paper reference. Cite both when this repository and its baseline
are used together.

## License

The [MIT License](LICENSE) covers the upstream software and the software changes
in this repository. Third-party datasets, pretrained models, and GloVe resources
retain their own terms. The original project figures remain copyrighted by their
contributors; see [NOTICE](NOTICE) for attribution and reuse boundaries.
