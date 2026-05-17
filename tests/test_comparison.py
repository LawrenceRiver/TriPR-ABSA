import torch

from pragmatic_residual.comparison import comparison_residual


def test_better_elsewhere_is_negative_for_current_food() -> None:
    sample = {
        "text_list": ["I", "have", "had", "better", "food", "elsewhere", "."],
        "aspect": "food",
        "aspect_post": [4, 5],
    }
    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))
    assert residual[1] > 0
    assert residual[0] < 0
    assert actions and actions[0].startswith("comparison_negative")
