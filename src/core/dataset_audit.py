"""
Dataset Audit

Checks whether each specimen listed in the Excel file has a matching image folder
and verifies whether expected image views are present.

Inputs:
- metadata/BaseDadosCol-2026.xlsx
- images/IHMT-EXXXXX/

Outputs:
- metadata/specimen_image_check.csv
- metadata/dataset_summary.json
"""

from pathlib import Path
import json

import pandas as pd

from src.utils.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetAudit:
    """Audit IHMT specimen metadata against image folders."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self) -> None:
        paths = CONFIG.path()

        self.excel_file = Path(paths["metadata"]["excel"])
        self.images_root = Path(paths["images"]["root"])

        self.output_csv = Path("metadata/specimen_image_check.csv")
        self.output_json = Path("metadata/dataset_summary.json")

        self.expected_views = set(CONFIG.view()["views"].keys())

        self.metadata: pd.DataFrame | None = None
        self.audit_table: pd.DataFrame | None = None

    def load_excel(self) -> None:
        """Load Excel metadata."""
        logger.info("Loading Excel metadata: %s", self.excel_file)

        if not self.excel_file.exists():
            raise FileNotFoundError(
                f"Excel file not found: {self.excel_file}"
            )

        self.metadata = pd.read_excel(
            self.excel_file,
            sheet_name="DadosDipteraMoscasIHMT",
        )

        logger.info(
            "Loaded %d specimens from Excel.",
            len(self.metadata),
        )

    def validate_columns(self) -> None:
        """Check required Excel columns."""
        required_columns = [
            "numCol",
            "Family",
            "Genus",
            "Specific epiteth",
            "scientificName",
        ]

        assert self.metadata is not None

        missing = [
            col for col in required_columns if col not in self.metadata.columns
        ]

        if missing:
            raise ValueError(f"Missing required Excel columns: {missing}")

        logger.info("Excel columns validated successfully.")

    def count_images_and_views(self, folder: Path) -> tuple[int, set[str]]:
        """Count valid images and extract view codes from filenames."""
        views_found: set[str] = set()
        image_count = 0

        if not folder.exists():
            return 0, views_found

        for image in folder.iterdir():
            if image.is_file() and image.suffix.lower() in self.IMAGE_EXTENSIONS:
                image_count += 1
                views_found.add(image.stem.upper())

        return image_count, views_found

    def build_audit_table(self) -> None:
        """Build specimen-level audit table."""
        assert self.metadata is not None

        rows = []

        for _, row in self.metadata.iterrows():
            specimen_id = str(row["numCol"]).strip()
            folder = self.images_root / specimen_id

            folder_exists = folder.exists() and folder.is_dir()
            image_count, views_found = self.count_images_and_views(folder)

            missing_views = sorted(self.expected_views - views_found)

            if not folder_exists:
                status = "MISSING_FOLDER"
            elif image_count == 0:
                status = "EMPTY_FOLDER"
            elif missing_views:
                status = "MISSING_VIEW"
            else:
                status = "COMPLETE"

            audit_row = {
                "numCol": specimen_id,
                "family": row["Family"],
                "genus": row["Genus"],
                "specific_epithet": row["Specific epiteth"],
                "scientific_name": row["scientificName"],
                "folder_exists": folder_exists,
                "num_images": image_count,
                "views_found": ",".join(sorted(views_found)),
                "missing_views": ",".join(missing_views),
                "status": status,
            }

            for view in sorted(self.expected_views):
                audit_row[view] = view in views_found

            rows.append(audit_row)

        self.audit_table = pd.DataFrame(rows)

        logger.info("Audit table built with %d rows.", len(self.audit_table))

    def save_audit_csv(self) -> None:
        """Save specimen audit CSV."""
        assert self.audit_table is not None

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.audit_table.to_csv(self.output_csv, index=False)

        logger.info("Saved audit CSV: %s", self.output_csv)

    def save_summary_json(self) -> None:
        """Save dataset summary JSON."""
        assert self.audit_table is not None

        total_specimens = len(self.audit_table)
        folders_found = int(self.audit_table["folder_exists"].sum())
        missing_folders = int((self.audit_table["status"] == "MISSING_FOLDER").sum())
        empty_folders = int((self.audit_table["status"] == "EMPTY_FOLDER").sum())
        complete_specimens = int((self.audit_table["status"] == "COMPLETE").sum())
        incomplete_specimens = total_specimens - complete_specimens
        total_images = int(self.audit_table["num_images"].sum())

        summary = {
            "total_specimens_in_excel": total_specimens,
            "folders_found": folders_found,
            "missing_folders": missing_folders,
            "empty_folders": empty_folders,
            "complete_specimens": complete_specimens,
            "incomplete_specimens": incomplete_specimens,
            "total_images_found": total_images,
            "average_images_per_specimen": (
                round(total_images / total_specimens, 2)
                if total_specimens > 0
                else 0
            ),
        }

        self.output_json.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_json, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        logger.info("Saved dataset summary JSON: %s", self.output_json)

    def run(self) -> None:
        """Run complete audit pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Audit")
        logger.info("=" * 60)

        self.load_excel()
        self.validate_columns()
        self.build_audit_table()
        self.save_audit_csv()
        self.save_summary_json()

        logger.info("Dataset Audit completed successfully.")


if __name__ == "__main__":
    audit = DatasetAudit()
    audit.run()
