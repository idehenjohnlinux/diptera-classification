from pathlib import Path
from src.config import ConfigManager

def test_dataset_structure():
    cfg = ConfigManager()

    image_root = Path(cfg.paths["images"]["root"])

    assert image_root.exists(), "Images folder missing"

    specimens = list(image_root.iterdir())

    assert len(specimens) > 0, "No specimen folders found"

    print(f"Found {len(specimens)} specimens")
