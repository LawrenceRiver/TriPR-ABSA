import torch

from pragmatic_residual.fact import fact_residual


def test_menu_listing_moves_positive_prediction_toward_neutral() -> None:
    sample = {
        "text_list": ["The", "menu", "includes", "sushi", ",", "pasta", "and", "wine", "."],
        "aspect": "menu",
        "aspect_post": [1, 2],
    }
    residual, actions = fact_residual(sample, torch.tensor([0.70, 0.05, 0.25]))
    assert residual[2] > 0
    assert residual[0] < 0
    assert actions and actions[0].startswith("fact_neutral")
