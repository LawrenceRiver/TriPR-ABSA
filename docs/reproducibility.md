# Reproducibility

Run commands from the repository root. Python 3.9 and 3.10 are the supported
versions.

## Data and external resources

Datasets are not redistributed in this repository. Follow the
[upstream TextGT dataset preparation instructions](https://github.com/shuoyinn/TextGT#priliminaries)
and place the prepared directories under `dataset/`, as expected by `train.py`.
The upstream instructions point to preprocessed ABSA data from
[SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA),
[DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA), and
[CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA). Their original terms
still apply.

The upstream code and datasets credit DualGCN,
[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), and CDT_ABSA.
Processing raw data requires [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).
Non-BERT training requires [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
and the upstream setup uses `glove.840B.300d.zip`.

## CPU tests

Install the development dependencies, then run the residual unit tests and the
full release suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest \
  tests/test_apply.py \
  tests/test_fact.py \
  tests/test_comparison.py \
  tests/test_intensity.py \
  tests/test_prior_builder.py
python -m pytest
```

The full suite checks the reported metrics, generated figures, architecture
asset, redacted paper, and upstream CLI import in addition to the unit tests.

## GPU baselines

Install `requirements.txt` and prepare the Restaurant data first. For the
non-BERT run, download GloVe and build the vocabulary:

```bash
python -m pip install -r requirements.txt
python prepare_vocab.py \
  --data_dir ./dataset/Restaurants_corenlp \
  --vocab_dir ./dataset/Restaurants_corenlp
```

The following commands reproduce the upstream Restaurant configurations exposed
by `train.py`. Select the CUDA device with `--cuda`:

```bash
python train.py \
  --cuda 0 \
  --seed 1000 \
  --model_name text-gt \
  --num_layers 8 \
  --scheduler linear \
  --warmup 2 \
  --ffn_dropout 0.4 \
  --attn_dropout 0.2 \
  --balance_loss

python train.py \
  --cuda 0 \
  --seed 1000 \
  --model_name text-gt-bert \
  --ffn_dropout 0.5 \
  --attn_dropout 0.2 \
  --balance_loss
```

The BERT run downloads `bert-base-uncased` through Transformers unless it is
already cached. These are training commands, not reruns of the selected
checkpoints recorded in `results/reported_metrics.json`.

## Offline train-split prior

The default provider is offline. This command reads only the Restaurant training
file and makes no network request:

```bash
python scripts/build_phrase_prior.py \
  --train-file ./dataset/Restaurants_corenlp/train.json \
  --output ./artifacts/restaurant-train-prior.json
```

The output metadata records `split: train`, `provider: offline`, and
`uses_test_text_or_label: false`. Load it with:

```python
from pragmatic_residual import load_prior

prior = load_prior("artifacts/restaurant-train-prior.json")
```

The optional DeepSeek path is documented in the main README. It is not needed for
the offline result.

## Metrics and figures

The checked-in metric source is
[`results/reported_metrics.json`](../results/reported_metrics.json). Validate its
contract with:

```bash
python -m pytest tests/test_results.py
```

Regenerate the three result figures from that file with:

```bash
python scripts/visualize_results.py \
  --results results/reported_metrics.json \
  --output-dir assets/figures
```
