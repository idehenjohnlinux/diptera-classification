from pathlib import Path

import pandas as pd
from PIL import Image

from src.core.validator import DatasetValidator


def test_validate_image_valid(tmp_path):
    image_path = tmp_path / "FLT.jpg"

    image = Image.new("RGB", (100, 80), color="white")
    image.save(image_path)

    validator = DatasetValidator()
    result = validator.validate_image(image_path)

    assert result["is_valid"] is True
    assert result["width"] == 100
    assert result["height"] == 80
    assert result["mode"] == "RGB"
    assert result["error"] is None


def test_validate_image_invalid(tmp_path):
    image_path = tmp_path / "bad_image.jpg"
    image_path.write_text("this is not a real image")

    validator = DatasetValidator()
    result = validator.validate_image(image_path)

    assert result["is_valid"] is False
    assert result["error"] is not None


def test_build_validation_report_missing_folder(tmp_path, monkeypatch):
    images_root = tmp_path / "images"
    images_root.mkdir()

    audit_csv = tmp_path / "specimen_image_check.csv"
    output_csv = tmp_path / "image_validation_report.csv"

    pd.DataFrame(
        [
            {
                "numCol": "IHMT-E99999",
                "scientific_name": "Test species",
                "folder_exists": False,
                "num_images": 0,
                "status": "MISSING_FOLDER",
            }
        ]
    ).to_csv(audit_csv, index=False)

    validator = DatasetValidator()
    validator.images_root = images_root
    validator.audit_csv = audit_csv
    validator.output_csv = output_csv

    validator.load_audit_table()
    validator.build_validation_report()

    assert validator.validation_table is not None
    assert len(validator.validation_table) == 1
    assert not bool(validator.validation_table.iloc[0]["is_valid"])
    assert validator.validation_table.iloc[0]["error"] == "Missing specimen folder"

def test_save_report(tmp_path):
    output_csv = tmp_path / "image_validation_report.csv"

    validator = DatasetValidator()
    validator.output_csv = output_csv
    validator.validation_table = pd.DataFrame(
        [
            {
                "specimen_id": "IHMT-E52334",
                "image_path": "images/IHMT-E52334/FLT.jpg",
                "image_name": "FLT.jpg",
                "view_code": "FLT",
                "is_valid": True,
                "width": 100,
                "height": 80,
                "mode": "RGB",
                "error": None,
            }
        ]
    )

    validator.save_report()

    assert output_csv.exists()

    saved = pd.read_csv(output_csv)
    assert len(saved) == 1
    assert saved.iloc[0]["specimen_id"] == "IHMT-E52334"
