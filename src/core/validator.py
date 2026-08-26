"""
Dataset Validator
=================

Validates image files and selects only taxonomically identified,
valid images for CNN training.

Inputs:
- metadata/specimen_image_check.csv
- images/

Outputs:
- metadata/image_validation_report.csv
- metadata/training_dataset.csv
"""

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class DatasetValidator:
    """Validate images and select training-ready records."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    EXPECTED_VIEWS = {"FLT", "FLP", "FFF", "FDT"}
    REQUIRED_TAXONOMY_COLUMNS = ["family", "genus", "specific_epithet"]

    def __init__(self) -> None:
        paths = CONFIG.path()

        self.images_root = Path(paths["images"]["root"])
        self.audit_csv = Path("metadata/specimen_image_check.csv")
        self.output_csv = Path("metadata/image_validation_report.csv")
        self.training_csv = Path("metadata/training_dataset.csv")

        self.audit_table: pd.DataFrame | None = None
        self.validation_table: pd.DataFrame | None = None

    def load_audit_table(self) -> None:
        """Load specimen audit table."""
        logger.info("Loading audit table: %s", self.audit_csv)

        if not self.audit_csv.exists():
            raise FileNotFoundError(
                f"Audit file not found: {self.audit_csv}. "
                "Run dataset_audit first."
            )

        self.audit_table = pd.read_csv(self.audit_csv)

        required_columns = [
            "numCol",
            "family",
            "genus",
            "specific_epithet",
            "scientific_name",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.audit_table.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing required columns in audit table: {missing_columns}"
            )

    def is_taxonomically_identified(self, row: pd.Series) -> bool:
        """Return True if family, genus, and specific epithet are present."""
        for column in self.REQUIRED_TAXONOMY_COLUMNS:
            value = row.get(column)

            if pd.isna(value) or str(value).strip() == "":
                return False

        return True

    def validate_image(self, image_path: Path) -> dict:
        """Validate a single image file."""
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "view_code": image_path.stem.upper().split("-")[0],
            "is_valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": None,
            "status": "INVALID_IMAGE",
        }

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                result["width"] = img.width
                result["height"] = img.height
                result["mode"] = img.mode
                result["is_valid"] = True
                result["status"] = "VALID_IMAGE"

        except (UnidentifiedImageError, OSError, ValueError) as error:
            result["error"] = str(error)

        return result

    def make_error_row(
        self,
        specimen_id: str,
        status: str,
        error: str,
        row: pd.Series | None = None,
        view_code: str | None = None,
    ) -> dict:
        """Create a standardized non-valid row."""
        taxonomy_identified = (
            self.is_taxonomically_identified(row)
            if row is not None
            else False
        )

        return {
            "specimen_id": specimen_id,
            "family": row.get("family") if row is not None else None,
            "genus": row.get("genus") if row is not None else None,
            "specific_epithet": row.get("specific_epithet") if row is not None else None,
            "scientific_name": row.get("scientific_name") if row is not None else None,
            "taxonomy_identified": taxonomy_identified,
            "image_path": None,
            "image_name": None,
            "view_code": view_code,
            "is_valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "status": status,
            "error": error,
        }

    def build_validation_report(self) -> None:
        """Validate all images and mark training eligibility."""
        assert self.audit_table is not None

        rows = []

        metadata_ids = set(
            self.audit_table["numCol"]
            .astype(str)
            .str.strip()
        )

        folder_ids = (
            {
                folder.name
                for folder in self.images_root.iterdir()
                if folder.is_dir()
            }
            if self.images_root.exists()
            else set()
        )

        extra_folders = sorted(folder_ids - metadata_ids)

        for specimen_id in extra_folders:
            rows.append(
                self.make_error_row(
                    specimen_id=specimen_id,
                    status="EXTRA_FOLDER",
                    error="Folder exists but specimen is missing from metadata",
                    row=None,
                )
            )

        for _, row in self.audit_table.iterrows():
            specimen_id = str(row["numCol"]).strip()
            folder = self.images_root / specimen_id

            taxonomy_identified = self.is_taxonomically_identified(row)

            if not taxonomy_identified:
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="EXCLUDED_NOT_IDENTIFIED",
                        error="Specimen excluded from training: missing family, genus, or specific epithet",
                        row=row,
                    )
                )
                continue

            if not folder.exists() or not folder.is_dir():
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="MISSING_FOLDER",
                        error="Missing specimen folder",
                        row=row,
                    )
                )
                continue

            images = [
                file
                for file in folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in self.IMAGE_EXTENSIONS
            ]

            if not images:
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="EMPTY_FOLDER",
                        error="No valid image files found",
                        row=row,
                    )
                )
                continue

            view_codes = [image.stem.upper().split("-")[0] for image in images]

            invalid_views = sorted(set(view_codes) - self.EXPECTED_VIEWS)
            missing_views = sorted(self.EXPECTED_VIEWS - set(view_codes))
            duplicate_views = sorted(
                {view for view in view_codes if view_codes.count(view) > 1}
            )

            for view in invalid_views:
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="INVALID_VIEW",
                        error="Invalid image filename/view code",
                        row=row,
                        view_code=view,
                    )
                )

            for view in missing_views:
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="MISSING_VIEW",
                        error="Missing expected view",
                        row=row,
                        view_code=view,
                    )
                )

            for view in duplicate_views:
                rows.append(
                    self.make_error_row(
                        specimen_id=specimen_id,
                        status="DUPLICATE_VIEW",
                        error="Duplicate view image",
                        row=row,
                        view_code=view,
                    )
                )

            for image_path in sorted(images):
                image_result = self.validate_image(image_path)

                image_result["specimen_id"] = specimen_id
                image_result["family"] = row.get("family")
                image_result["genus"] = row.get("genus")
                image_result["specific_epithet"] = row.get("specific_epithet")
                image_result["scientific_name"] = row.get("scientific_name")
                image_result["taxonomy_identified"] = taxonomy_identified

                rows.append(image_result)

        self.validation_table = pd.DataFrame(rows)

        logger.info("Validated %d records.", len(self.validation_table))

    def save_report(self) -> None:
        """Save full validation report."""
        assert self.validation_table is not None

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.validation_table.to_csv(self.output_csv, index=False)

        logger.info("Saved validation report: %s", self.output_csv)

    def save_training_dataset(self) -> None:
        """Save only valid images from identified specimens."""
        assert self.validation_table is not None

        training_data = self.validation_table[
            (self.validation_table["is_valid"] == True)
            & (self.validation_table["taxonomy_identified"] == True)
            & (self.validation_table["status"] == "VALID_IMAGE")
            & (self.validation_table["error"].isna())
        ].copy()

        self.training_csv.parent.mkdir(parents=True, exist_ok=True)
        training_data.to_csv(self.training_csv, index=False)

        logger.info("Saved training dataset: %s", self.training_csv)
        logger.info("Training images available: %d", len(training_data))

    def run(self) -> None:
        """Run validation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Validation")
        logger.info("=" * 60)

        self.load_audit_table()
        self.build_validation_report()
        self.save_report()
        self.save_training_dataset()

        logger.info("Dataset Validation completed successfully.")


if __name__ == "__main__":
    setup_logging()

    validator = DatasetValidator()
    validator.run()
