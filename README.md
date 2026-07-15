[English](README.md) | [简体中文](README.zh-CN.md)

# TextGT Pragmatic Residual Adapter

An independent TextGT research fork that adds a model-agnostic pragmatic residual to three-class aspect-based sentiment logits.

[![CI](https://github.com/LawrenceRiver/TextGT/actions/workflows/ci.yml/badge.svg)](https://github.com/LawrenceRiver/TextGT/actions/workflows/ci.yml)
![Python 3.9 and 3.10](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?logo=python&logoColor=white)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)
[![MIT software license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream TextGT](https://img.shields.io/badge/upstream-shuoyinn%2FTextGT-181717?logo=github)](https://github.com/shuoyinn/TextGT)
[![Base paper DOI](https://img.shields.io/badge/base%20paper-10.1609%2Faaai.v38i17.29911-2f6f9f)](https://doi.org/10.1609/aaai.v38i17.29911)

## Upstream relationship

This repository is an independent research fork of [shuoyinn/TextGT](https://github.com/shuoyinn/TextGT). It is maintained separately and is not endorsed by or affiliated with the upstream authors. The original implementation and citation remain credited in [NOTICE](NOTICE).

## Architecture

![TextGT pragmatic residual architecture](assets/architecture.png)

The adapter reads a backbone's three logits and one parsed sample. It returns adjusted logits without changing the backbone. The full interface and equations are in [docs/method.md](docs/method.md).

## Method

The class order is `positive`, `negative`, `neutral`.

### Fact

The fact module detects objective listings and descriptions near the aspect. Under its confidence gates, it moves a positive prediction toward neutral.

### Comparison

The comparison module resolves whether the current aspect wins or loses an explicit comparison, then adjusts positive and negative logits in that direction.

### Intensity

The intensity module matches phrases from a validated train-split prior. It accounts for local negation, clause boundaries, distance, and aspect scope. With no prior, this module is a no-op.

### Composer

The composer applies selected modules in order. Each module sees probabilities from the logits produced by the preceding module. The public entry points are `apply_pragmatic_residual`, `apply_batch`, and `load_prior`.

## Reported Restaurant results

The primary SemEval-2014 Restaurant test table is the mean of three selected checkpoints. Values come from [results/reported_metrics.json](results/reported_metrics.json).

| Strategy | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Baseline | 0.8382 | 0.7480 |
| Fact | 0.8406 | 0.7522 |
| Comparison | 0.8394 | 0.7499 |
| Intensity | 0.8388 | 0.7488 |
| Fact + comparison | 0.8418 | 0.7541 |
| All | 0.8424 | 0.7547 |

Reported result from 3 selected checkpoints; not independently rerun during release preparation and not a state-of-the-art claim. Laptop, Twitter, and multi-backbone tables with their limits are in [docs/results.md](docs/results.md).

## Installation

The residual package supports Python 3.9 and 3.10.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development checks, install `requirements-dev.txt`. Full upstream model training uses `requirements.txt` and separately obtained datasets and model resources.

Datasets are not redistributed here. Follow the [upstream preparation instructions](https://github.com/shuoyinn/TextGT#priliminaries). The upstream code and datasets credit [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA), [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), and [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA); compatible preprocessed datasets are also available from [SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA). Preparing data from source requires [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/). Non-BERT training also requires [Stanford GloVe](https://nlp.stanford.edu/projects/glove/), specifically `glove.840B.300d.zip` for the upstream commands.

## Quick start

This example uses only the comparison module, so it does not need a phrase prior.

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

The residual API uses the fixed class order `positive`, `negative`, `neutral`.
Pass any subset of `("fact", "comparison", "intensity")` through `modules`; the
composer applies them in that order. The `prior` argument accepts either a
validated mapping or a JSON path created by `scripts/build_phrase_prior.py`.
Without a prior, the intensity module is a no-op.

The default prior builder is offline. `DEEPSEEK_API_KEY` is read only when
`--provider deepseek` is selected for optional remote phrase annotation. It is
not required for inference, testing, visualization, or the offline builder.

## Prior construction

The builder reads the training split only. Its default provider is offline and makes no network request:

```bash
python scripts/build_phrase_prior.py \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

DeepSeek annotation is optional. It sends train-derived phrase candidates and aggregate counts to the configured API, not complete dataset rows or labels. The API key is read from an environment variable and is never written to the prior:

```bash
export DEEPSEEK_API_KEY="your-key"
python scripts/build_phrase_prior.py \
  --provider deepseek \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

Using the remote path is subject to the provider's privacy policy and service terms.

## Reproducibility

[docs/reproducibility.md](docs/reproducibility.md) lists the CPU checks, upstream GPU baseline commands, data preparation links, and offline prior command. Reported metrics are stored in a machine-readable file rather than inferred from the figures.

## Contributors

Software and figure contributors are listed in [AUTHORS.md](AUTHORS.md).
Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

[CITATION.cff](CITATION.cff) contains the software metadata and the upstream
TextGT paper reference. GitHub and citation tools can read it directly.

## License

The [MIT License](LICENSE) applies to the upstream software and this repository's
software modifications. Third-party datasets, pretrained models, and GloVe
resources retain their own terms. Original architecture and result figures
remain copyrighted by their contributors unless separately licensed. See
[NOTICE](NOTICE) for attribution and boundaries.
