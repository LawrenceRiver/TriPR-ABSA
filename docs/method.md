# TriPR-ABSA method

The package adjusts logits produced by a three-class aspect-based sentiment
classifier. It does not train, wrap, or modify the classifier.

## Class order and public API

Every logit tensor uses this order:

```text
index 0 = positive
index 1 = negative
index 2 = neutral
```

The package exports:

```python
from pragmatic_residual import apply_batch, apply_pragmatic_residual, load_prior
```

`apply_pragmatic_residual` accepts logits shaped `[3]`. It handles one parsed
sample. `apply_batch` accepts logits shaped `[batch, 3]`. It also accepts a sample
sequence of the same length as the batch. Both return a new tensor and leave the
input logits unchanged.

## Sample schema

A sample is a Python dictionary with two required fields:

| Field | Form | Meaning |
| --- | --- | --- |
| `text_list` | token sequence | Original sentence tokens in order |
| `aspect_post` | two-item list or tuple | Half-open aspect token span `[start, end]` |

`aspect` is optional but helps the intensity module keep service-related phrases
with a service aspect. `pos` is optional and supplies part-of-speech tags to the
fact and comparison feature extractor. Other parsed fields may remain in the
dictionary; the residual API ignores them.

## Composer

Let `z_0` be the backbone logits. For selected modules `m_1, ..., m_k`, the
composer performs these steps in order:

```text
p_i = softmax(z_i)
delta_i = m_i(sample, p_i)
z_(i+1) = z_i + delta_i
output = z_k
```

With `modules=None`, the order is `fact`, `comparison`, `intensity`. Passing an
empty tuple returns a clone of the original logits. `return_details=True` also
returns the selected module names, action labels, each module residual, and the
total residual. Intensity details include the scalar score and matched phrases.

## Fact residual

The fact module derives two feature totals from the parsed sample:

```text
fact_score = fact_listing
           + there_be_existence
           + menu_enumeration
           + objective_description

relation_score = relation_fact_object_score
               + relation_object_list_density
               + relation_low_subjectivity_descriptor

f = max(fact_score / 3, relation_score / 3)
r = relation_opinion_render_score
```

Let `base` be `argmax(p)`, and let `margin` be the difference between the two
largest probabilities. The neutral-boundary gate is:

```text
boundary = (base == positive)
           and (p_neutral >= 0.035 or margin <= 0.78 or f >= 0.95)
```

When `f >= 0.45`, `r < 0.30`, and `boundary` is true, the residual is:

```text
delta_fact = f * [-1.10, -0.65, +2.90]
```

Otherwise it is `[0, 0, 0]`.

## Comparison residual

The feature extractor assigns non-negative strengths `c_pos` and `c_neg` to an
explicit comparison in which the current aspect wins or loses. The module adds:

```text
delta_comparison = c_pos * [+2.00, -1.05, -0.35]
                 + c_neg * [-1.10, +2.25, -0.35]
```

Patterns separated by punctuation, negated comparisons, and references outside
the current aspect clause do not receive a comparison residual.

## Intensity residual

`load_prior` converts a validated train-only JSON prior into a phrase mapping.
For each non-overlapping phrase match in the aspect clause, the intensity module
starts with the signed prior value. Local negation can dampen or weakly flip it;
matches more than four tokens from the aspect receive the default distance factor
`0.55`. Matches farther than ten tokens are ignored.

The score is the sum of effective positive values minus the magnitude of effective
negative values, clipped to `[-1.5, 1.5]`. With the default parameters:

```text
q = min(1, abs(score))

if score >= 0.4:
    delta_intensity = q * [+1.00, -0.25, -0.45]
elif score <= -0.2:
    delta_intensity = q * [-0.25, +1.00, -0.45]
else:
    delta_intensity = [0, 0, 0]
```

Supplying `intensity_params` can replace the scale, thresholds, neutral
suppression, negation, mixed-evidence, and distance defaults. A missing or empty
prior disables only the intensity adjustment.

## Error behavior

The API fails before composition for these contract errors:

- Non-tensor logits raise `TypeError`.
- Single-sample logits with any other shape raise `ValueError`.
- Batch logits with any other shape raise `ValueError`.
- Missing `text_list` or `aspect_post` raises `ValueError` and names the field.
- An `aspect_post` value that is not a two-item list or tuple raises `ValueError`.
- Unknown module names raise `ValueError`.
- A mismatch between the batch and sample counts raises `ValueError`.

`load_prior` raises a path-qualified `ValueError` for invalid JSON, a non-training
split, a malformed prior list, invalid polarity, or intensity outside `(0, 1]`.
A path that does not exist raises `FileNotFoundError`.

## Runnable synthetic example

Run this from the repository root after `python -m pip install -e .`:

```bash
python - <<'PY'
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

print(adjusted.tolist())
print(details["actions"])
PY
```

The comparison is negative for the current `food` aspect. The adjusted negative
logit is therefore larger, and the action list contains
`comparison_negative(1.00)`.
