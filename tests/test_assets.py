import hashlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_image_is_high_resolution() -> None:
    path = ROOT / "assets" / "architecture.png"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d6749a417a382410a008e64c0e9b391c99a97ea23e8c1474a700257c8206450d"
    )
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (1683, 934)
