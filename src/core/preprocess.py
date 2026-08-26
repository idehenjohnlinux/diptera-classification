"""Image-driven preprocessing pipeline for the Brachycera CNN project.

The image folders are the primary dataset.

Workflow
--------
1. Scan all specimen folders inside images/.
2. Read taxonomy from the Excel worksheet DadosDipteraMoscasIHMT.
3. Match each image-folder specimen ID with Excel numCol.
4. Preprocess every readable image.
5. Generate image-level and specimen-level CSV files.

Outputs
-------
metadata/master_dataset.csv
metadata/identified_family_genus.csv
metadata/unidentified_family_genus.csv
metadata/unmatched_specimens.csv
metadata/preprocessing_summary.csv
metadata/preprocessing_summary.json

Rules
-----
identified:
    family exists OR genus exists

unidentified:
    family missing AND genus missing
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_ROOT = PROJECT_ROOT / "images"
PROCESSED_ROOT = PROJECT_ROOT / "processed" / "images"

EXCEL_PATH = PROJECT_ROOT / "metadata" / "BaseDadosCol-2026.xlsx"
EXCEL_SHEET = "DadosDipteraMoscasIHMT"

MASTER_DATASET_PATH = (
    PROJECT_ROOT / "metadata" / "master_dataset.csv"
)
SUPERVISED_DATASET_PATH = (
    PROJECT_ROOT / "metadata" / "supervised_dataset.csv"
)

IDENTIFIED_PATH = (
    PROJECT_ROOT / "metadata" / "identified_family_genus.csv"
)

UNIDENTIFIED_PATH = (
    PROJECT_ROOT / "metadata" / "unidentified_family_genus.csv"
)

UNMATCHED_PATH = (
    PROJECT_ROOT / "metadata" / "unmatched_specimens.csv"
)

PREPROCESSING_REPORT_PATH = (
    PROJECT_ROOT / "metadata" / "preprocessing_summary.csv"
)

PREPROCESSING_JSON_PATH = (
    PROJECT_ROOT / "metadata" / "preprocessing_summary.json"
)

LOG_PATH = (
    PROJECT_ROOT / "reports" / "logs" / "pipeline.log"
)


# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

IMAGE_SIZE = 224
JPEG_QUALITY = 95
PADDING_VALUE = 255
OVERWRITE_EXISTING = True

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

EXPECTED_VIEWS = {
    "FDT",
    "FFF",
    "FLP",
    "FLT",
}

MISSING_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unlabelled",
    "unidentified",
    "not identified",
    "não identificado",
    "nao identificado",
    "sem identificação",
    "sem identificacao",
    "-",
}


# ============================================================
# LOGGING
# ============================================================

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value: Any) -> str:
    """Convert a value to clean text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_taxonomy_label(value: Any) -> str:
    """Clean family or genus values."""

    text = clean_text(value)

    if text.lower() in MISSING_VALUES:
        return "unlabelled"

    return text.strip().capitalize()


def normalize_specimen_id(value: Any) -> str:
    """Normalize specimen identifiers.

    Examples
    --------
    IHMT-E52334
    ihmt-e52334
    IHMT_E52334
    IHMT - E52334

    All become:

    IHMT-E52334
    """

    text = clean_text(value).upper()

    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "-")

    match = re.search(r"IHMT-?E-?(\d+)", text)

    if match:
        return f"IHMT-E{match.group(1)}"

    return text


def normalize_view_code(filename_stem: str) -> str:
    """Extract a known view code or preserve the filename stem."""

    stem = clean_text(filename_stem).upper()

    cleaned = re.sub(
        r"[^A-Z0-9_-]+",
        "_",
        stem,
    )

    if cleaned in EXPECTED_VIEWS:
        return cleaned

    match = re.search(
        r"(?:^|[_-])(FDT|FFF|FLP|FLT)(?:$|[_-])",
        cleaned,
    )

    if match:
        return match.group(1)

    return cleaned


def determine_taxonomy_status(
    family: str,
    genus: str,
) -> str:
    """Classify a specimen as identified or unidentified."""

    family_exists = family != "unlabelled"
    genus_exists = genus != "unlabelled"

    if family_exists or genus_exists:
        return "identified"

    return "unidentified"


# ============================================================
# MAIN CLASS
# ============================================================

class ImageDrivenPreprocessor:
    """Preprocess images using image folders as the primary dataset."""

    def __init__(self) -> None:
        """Create the required output directories."""

        PROCESSED_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        MASTER_DATASET_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the required files and directories."""

        if not IMAGES_ROOT.exists():
            raise FileNotFoundError(
                f"Images directory not found: {IMAGES_ROOT}"
            )

        if not IMAGES_ROOT.is_dir():
            raise NotADirectoryError(
                f"Images path is not a directory: {IMAGES_ROOT}"
            )

        if not EXCEL_PATH.exists():
            raise FileNotFoundError(
                f"Excel metadata file not found: {EXCEL_PATH}"
            )

    # --------------------------------------------------------
    # STEP 1: SCAN IMAGE FOLDERS
    # --------------------------------------------------------

    def scan_specimen_folders(self) -> pd.DataFrame:
        """Scan specimen folders inside images/.

        Only folders containing at least one supported image are included.
        """

        records: list[dict[str, Any]] = []

        specimen_folders = sorted(
            folder
            for folder in IMAGES_ROOT.iterdir()
            if folder.is_dir()
        )

        logger.info(
            "Found %s directories inside images/.",
            len(specimen_folders),
        )

        for folder in specimen_folders:
            image_files = sorted(
                file
                for file in folder.rglob("*")
                if (
                    file.is_file()
                    and file.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )

            if not image_files:
                logger.warning(
                    "Skipping folder with no supported images: %s",
                    folder.name,
                )
                continue

            specimen_id = normalize_specimen_id(
                folder.name
            )

            records.append(
                {
                    "numCol": specimen_id,
                    "specimen_folder": folder.name,
                    "specimen_folder_path": str(
                        folder.relative_to(PROJECT_ROOT)
                    ),
                    "number_of_images_found": len(image_files),
                }
            )

        specimen_table = pd.DataFrame(records)

        if specimen_table.empty:
            raise RuntimeError(
                "No specimen folders containing supported images were found."
            )

        duplicate_ids = specimen_table[
            specimen_table.duplicated(
                subset=["numCol"],
                keep=False,
            )
        ]

        if not duplicate_ids.empty:
            logger.warning(
                "Duplicate normalized specimen IDs were found: %s",
                duplicate_ids["numCol"].unique().tolist(),
            )

        specimen_table = specimen_table.drop_duplicates(
            subset=["numCol"],
            keep="first",
        ).reset_index(drop=True)

        logger.info(
            "Image-driven dataset contains %s specimen folders.",
            len(specimen_table),
        )

        return specimen_table

    # --------------------------------------------------------
    # STEP 2: LOAD EXCEL TAXONOMY
    # --------------------------------------------------------

    def load_taxonomy(self) -> pd.DataFrame:
        """Load taxonomy from DadosDipteraMoscasIHMT."""

        logger.info(
            "Reading taxonomy from worksheet '%s'.",
            EXCEL_SHEET,
        )

        excel_file = pd.ExcelFile(EXCEL_PATH)

        if EXCEL_SHEET not in excel_file.sheet_names:
            raise ValueError(
                f"Worksheet '{EXCEL_SHEET}' was not found. "
                f"Available worksheets: {excel_file.sheet_names}"
            )

        dataframe = pd.read_excel(
            EXCEL_PATH,
            sheet_name=EXCEL_SHEET,
            dtype=str,
        )

        required_columns = {
            "numCol",
            "Family",
            "Genus",
        }

        missing_columns = (
            required_columns - set(dataframe.columns)
        )

        if missing_columns:
            raise KeyError(
                "The taxonomy worksheet is missing columns: "
                f"{sorted(missing_columns)}"
            )

        taxonomy = dataframe[
            [
                "numCol",
                "Family",
                "Genus",
            ]
        ].copy()

        taxonomy = taxonomy.rename(
            columns={
                "Family": "family",
                "Genus": "genus",
            }
        )

        taxonomy["numCol"] = taxonomy[
            "numCol"
        ].apply(normalize_specimen_id)

        taxonomy["family"] = taxonomy[
            "family"
        ].apply(clean_taxonomy_label)

        taxonomy["genus"] = taxonomy[
            "genus"
        ].apply(clean_taxonomy_label)

        taxonomy = taxonomy[
            taxonomy["numCol"] != ""
        ].copy()

        self.report_taxonomy_conflicts(taxonomy)

        taxonomy = taxonomy.drop_duplicates(
            subset=["numCol"],
            keep="first",
        ).reset_index(drop=True)

        logger.info(
            "Loaded %s unique taxonomy records from Excel.",
            len(taxonomy),
        )

        return taxonomy

    def report_taxonomy_conflicts(
        self,
        taxonomy: pd.DataFrame,
    ) -> None:
        """Report duplicate specimen IDs with conflicting labels."""

        grouped = (
            taxonomy.groupby("numCol")
            .agg(
                family_variants=(
                    "family",
                    lambda values: values.nunique(
                        dropna=False
                    ),
                ),
                genus_variants=(
                    "genus",
                    lambda values: values.nunique(
                        dropna=False
                    ),
                ),
            )
        )

        conflicts = grouped[
            (grouped["family_variants"] > 1)
            | (grouped["genus_variants"] > 1)
        ]

        if not conflicts.empty:
            logger.warning(
                "%s specimen IDs have conflicting taxonomy. "
                "The first record will be used.",
                len(conflicts),
            )

    # --------------------------------------------------------
    # STEP 3: MATCH THE 383 IMAGE SPECIMENS WITH EXCEL
    # --------------------------------------------------------

    def match_specimens_with_taxonomy(
        self,
        specimen_table: pd.DataFrame,
        taxonomy: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach family and genus only to specimens with images."""

        matched = specimen_table.merge(
            taxonomy,
            on="numCol",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        matched["metadata_match"] = (
            matched["_merge"] == "both"
        )

        matched = matched.drop(
            columns=["_merge"]
        )

        matched["family"] = matched[
            "family"
        ].apply(clean_taxonomy_label)

        matched["genus"] = matched[
            "genus"
        ].apply(clean_taxonomy_label)

        matched["use_for_family"] = (
            matched["family"] != "unlabelled"
        )

        matched["use_for_genus"] = (
            matched["genus"] != "unlabelled"
        )

        matched["taxonomy_status"] = matched.apply(
            lambda row: determine_taxonomy_status(
                family=row["family"],
                genus=row["genus"],
            ),
            axis=1,
        )

        unmatched = matched.loc[
            ~matched["metadata_match"]
        ].copy()

        unmatched.to_csv(
            UNMATCHED_PATH,
            index=False,
        )

        logger.info(
            "Image specimens matched with Excel: %s.",
            int(matched["metadata_match"].sum()),
        )

        logger.info(
            "Image specimens not found in Excel: %s.",
            int((~matched["metadata_match"]).sum()),
        )

        logger.info(
            "Identified image specimens: %s.",
            int(
                (
                    matched["taxonomy_status"]
                    == "identified"
                ).sum()
            ),
        )

        logger.info(
            "Unidentified image specimens: %s.",
            int(
                (
                    matched["taxonomy_status"]
                    == "unidentified"
                ).sum()
            ),
        )

        return matched

    # --------------------------------------------------------
    # STEP 4: CREATE IMAGE-LEVEL TABLE
    # --------------------------------------------------------

    def build_image_table(
        self,
        specimen_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one record for every image in the selected folders."""

        records: list[dict[str, Any]] = []

        for _, specimen in specimen_table.iterrows():
            folder = (
                IMAGES_ROOT
                / specimen["specimen_folder"]
            )

            image_files = sorted(
                file
                for file in folder.rglob("*")
                if (
                    file.is_file()
                    and file.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            )

            for image_file in image_files:
                view_code = normalize_view_code(
                    image_file.stem
                )

                records.append(
                    {
                        "numCol": specimen["numCol"],
                        "specimen_folder": specimen[
                            "specimen_folder"
                        ],
                        "family": specimen["family"],
                        "genus": specimen["genus"],
                        "metadata_match": bool(
                            specimen["metadata_match"]
                        ),
                        "taxonomy_status": specimen[
                            "taxonomy_status"
                        ],
                        "use_for_family": bool(
                            specimen["use_for_family"]
                        ),
                        "use_for_genus": bool(
                            specimen["use_for_genus"]
                        ),
                        "view_code": view_code,
                        "expected_view": (
                            view_code in EXPECTED_VIEWS
                        ),
                        "source_image_name": (
                            image_file.name
                        ),
                        "source_image_path": str(
                            image_file.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "source_absolute_path": str(
                            image_file
                        ),
                    }
                )

        image_table = pd.DataFrame(records)

        if image_table.empty:
            raise RuntimeError(
                "No images were found in the specimen folders."
            )

        logger.info(
            "Found %s images across %s image specimens.",
            len(image_table),
            image_table["numCol"].nunique(),
        )

        return image_table

    # --------------------------------------------------------
    # STEP 5: PREPROCESS IMAGES
    # --------------------------------------------------------

    def create_output_path(
        self,
        specimen_id: str,
        view_code: str,
        occurrence: int,
    ) -> Path:
        """Create a unique output path."""

        output_folder = (
            PROCESSED_ROOT / specimen_id
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        if occurrence == 1:
            filename = f"{view_code}.jpg"
        else:
            filename = (
                f"{view_code}_{occurrence:02d}.jpg"
            )

        return output_folder / filename

    def preprocess_one_image(
        self,
        source_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """Preprocess one image."""

        result: dict[str, Any] = {
            "original_width": None,
            "original_height": None,
            "original_mode": None,
            "processed_width": None,
            "processed_height": None,
            "processed_mode": None,
            "preprocessing_status": "failed",
            "error_message": "",
        }

        if output_path.exists() and not OVERWRITE_EXISTING:
            try:
                with Image.open(output_path) as image:
                    result["processed_width"] = image.width
                    result["processed_height"] = image.height
                    result["processed_mode"] = image.mode

                result["preprocessing_status"] = "existing"

                return result

            except OSError:
                logger.warning(
                    "Existing processed image is unreadable: %s",
                    output_path,
                )

        try:
            with Image.open(source_path) as image:
                image = ImageOps.exif_transpose(image)

                result["original_width"] = image.width
                result["original_height"] = image.height
                result["original_mode"] = image.mode

                image = image.convert("RGB")

                image.thumbnail(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.LANCZOS,
                )

                canvas = Image.new(
                    "RGB",
                    (IMAGE_SIZE, IMAGE_SIZE),
                    (
                        PADDING_VALUE,
                        PADDING_VALUE,
                        PADDING_VALUE,
                    ),
                )

                x_position = (
                    IMAGE_SIZE - image.width
                ) // 2

                y_position = (
                    IMAGE_SIZE - image.height
                ) // 2

                canvas.paste(
                    image,
                    (x_position, y_position),
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                canvas.save(
                    output_path,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True,
                )

                result["processed_width"] = canvas.width
                result["processed_height"] = canvas.height
                result["processed_mode"] = canvas.mode
                result["preprocessing_status"] = "processed"

        except UnidentifiedImageError:
            result["error_message"] = (
                "Pillow could not identify this image."
            )

        except OSError as error:
            result["error_message"] = (
                f"Image input/output error: {error}"
            )

        except Exception as error:
            result["error_message"] = (
                f"Unexpected error: {error}"
            )

        return result

    def preprocess_all_images(
        self,
        image_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Preprocess every image in the image-driven dataset."""

        results: list[dict[str, Any]] = []

        occurrence_counter: Counter[
            tuple[str, str]
        ] = Counter()

        total_images = len(image_table)

        for position, (_, row) in enumerate(
            image_table.iterrows(),
            start=1,
        ):
            key = (
                row["numCol"],
                row["view_code"],
            )

            occurrence_counter[key] += 1

            output_path = self.create_output_path(
                specimen_id=row["numCol"],
                view_code=row["view_code"],
                occurrence=occurrence_counter[key],
            )

            image_result = self.preprocess_one_image(
                source_path=Path(
                    row["source_absolute_path"]
                ),
                output_path=output_path,
            )

            results.append(
                {
                    "numCol": row["numCol"],
                    "specimen_folder": row[
                        "specimen_folder"
                    ],
                    "family": row["family"],
                    "genus": row["genus"],
                    "metadata_match": bool(
                        row["metadata_match"]
                    ),
                    "taxonomy_status": row[
                        "taxonomy_status"
                    ],
                    "use_for_family": bool(
                        row["use_for_family"]
                    ),
                    "use_for_genus": bool(
                        row["use_for_genus"]
                    ),
                    "view_code": row["view_code"],
                    "expected_view": bool(
                        row["expected_view"]
                    ),
                    "source_image_name": row[
                        "source_image_name"
                    ],
                    "source_image_path": row[
                        "source_image_path"
                    ],
                    "processed_image_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    **image_result,
                }
            )

            if image_result[
                "preprocessing_status"
            ] == "failed":
                logger.error(
                    "Failed to preprocess %s: %s",
                    row["source_image_path"],
                    image_result["error_message"],
                )

            if position % 100 == 0 or position == total_images:
                logger.info(
                    "Processed %s of %s images.",
                    position,
                    total_images,
                )

        return pd.DataFrame(results)

    # --------------------------------------------------------
    # STEP 6: CREATE OUTPUT CSV FILES
    # --------------------------------------------------------

    def create_master_dataset(
        self,
        preprocessing_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create master_dataset.csv with successful images only."""

        successful_statuses = {
            "processed",
            "existing",
        }

        master = preprocessing_results.loc[
            preprocessing_results[
                "preprocessing_status"
            ].isin(successful_statuses)
        ].copy()

        master = master[
            [
                "numCol",
                "specimen_folder",
                "family",
                "genus",
                "metadata_match",
                "taxonomy_status",
                "use_for_family",
                "use_for_genus",
                "view_code",
                "expected_view",
                "source_image_name",
                "source_image_path",
                "processed_image_path",
                "original_width",
                "original_height",
                "original_mode",
                "processed_width",
                "processed_height",
                "processed_mode",
            ]
        ]

        master = master.sort_values(
            by=[
                "numCol",
                "view_code",
                "source_image_name",
            ]
        ).reset_index(drop=True)

        master.to_csv(
            MASTER_DATASET_PATH,
            index=False,
        )

        logger.info(
            "Saved master_dataset.csv with %s image rows.",
            len(master),
        )

        return master
        
    def create_supervised_dataset(
        self,
        master: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create an image-level dataset for supervised learning.

        An image is included when its specimen has at least one
        usable taxonomic label:

        family identified OR genus identified

        The file still keeps separate flags for family and genus
        because not every image can be used for both models.
        """

        supervised_mask = (
            master["use_for_family"]
            | master["use_for_genus"]
        )

        supervised = master.loc[
            supervised_mask
        ].copy()

        supervised = supervised[
            [
            "numCol",
            "specimen_folder",
            "family",
            "genus",
            "metadata_match",
            "taxonomy_status",
            "use_for_family",
            "use_for_genus",
            "view_code",
            "expected_view",
            "source_image_name",
            "source_image_path",
            "processed_image_path",
            "original_width",
            "original_height",
            "original_mode",
            "processed_width",
            "processed_height",
            "processed_mode",
            ]
        ]

        supervised = supervised.sort_values(
            by=[
                "numCol",
                "view_code",
                "source_image_name",
            ]
        ).reset_index(drop=True)

        supervised.to_csv(
            SUPERVISED_DATASET_PATH,
            index=False,
        )

        logger.info(
            "Saved supervised_dataset.csv with %s image rows "
            "from %s specimens.",
            len(supervised),
            supervised["numCol"].nunique(),
        )

        logger.info(
            "Family-eligible image rows: %s.",
            int(supervised["use_for_family"].sum()),
        )

        logger.info(
            "Genus-eligible image rows: %s.",
            int(supervised["use_for_genus"].sum()),
        )

        return supervised    

    def create_specimen_level_files(
        self,
        master: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create identified and unidentified specimen CSVs."""

        specimen_summary = (
            master.groupby(
                "numCol",
                as_index=False,
            )
            .agg(
                specimen_folder=(
                    "specimen_folder",
                    "first",
                ),
                family=(
                    "family",
                    "first",
                ),
                genus=(
                    "genus",
                    "first",
                ),
                metadata_match=(
                    "metadata_match",
                    "first",
                ),
                taxonomy_status=(
                    "taxonomy_status",
                    "first",
                ),
                use_for_family=(
                    "use_for_family",
                    "first",
                ),
                use_for_genus=(
                    "use_for_genus",
                    "first",
                ),
                number_of_images=(
                    "processed_image_path",
                    "count",
                ),
                available_views=(
                    "view_code",
                    lambda values: ";".join(
                        sorted(set(values))
                    ),
                ),
            )
        )

        identified_mask = (
            specimen_summary["use_for_family"]
            | specimen_summary["use_for_genus"]
        )

        identified = specimen_summary.loc[
            identified_mask
        ].copy()

        unidentified = specimen_summary.loc[
            ~identified_mask
        ].copy()

        identified = identified.sort_values(
            "numCol"
        ).reset_index(drop=True)

        unidentified = unidentified.sort_values(
            "numCol"
        ).reset_index(drop=True)

        identified.to_csv(
            IDENTIFIED_PATH,
            index=False,
        )

        unidentified.to_csv(
            UNIDENTIFIED_PATH,
            index=False,
        )

        logger.info(
            "Saved identified_family_genus.csv: %s specimens.",
            len(identified),
        )

        logger.info(
            "Saved unidentified_family_genus.csv: %s specimens.",
            len(unidentified),
        )

        return identified, unidentified

    def save_preprocessing_reports(
        self,
        preprocessing_results: pd.DataFrame,
        master: pd.DataFrame,
        supervised: pd.DataFrame,
        identified: pd.DataFrame,
        unidentified: pd.DataFrame,
        matched_specimens: pd.DataFrame,
    ) -> None:
        """Save CSV and JSON reports."""

        preprocessing_results.to_csv(
            PREPROCESSING_REPORT_PATH,
            index=False,
        )

        failed = preprocessing_results[
            preprocessing_results[
                "preprocessing_status"
            ] == "failed"
        ]

        summary = {
            "image_folder_specimens": int(
                len(matched_specimens)
            ),
            "specimens_matched_to_excel": int(
                matched_specimens[
                    "metadata_match"
                ].sum()
            ),
            "specimens_not_matched_to_excel": int(
                (
                    ~matched_specimens[
                        "metadata_match"
                    ]
                ).sum()
            ),
            "identified_specimens": int(
                len(identified)
            ),
            "unidentified_specimens": int(
                len(unidentified)
            ),
            "total_images_attempted": int(
                len(preprocessing_results)
            ),
            "successfully_processed_images": int(
                len(master)
            ),
            "supervised_image_rows": int(
                len(supervised)
            ),
            "supervised_specimens": int(
                supervised["numCol"].nunique()
            ),
            "family_eligible_image_rows": int(
                supervised["use_for_family"].sum()
            ),
            "genus_eligible_image_rows": int(
                supervised["use_for_genus"].sum()
            ),
            "failed_images": int(
                len(failed)
            ),
            "family_eligible_specimens": int(
                identified[
                    "use_for_family"
                ].sum()
            ),
            "genus_eligible_specimens": int(
                identified[
                    "use_for_genus"
                ].sum()
            ),
            "image_size": IMAGE_SIZE,
            "excel_sheet": EXCEL_SHEET,
        }

        with PREPROCESSING_JSON_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Saved preprocessing summary reports."
        )

    # --------------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------------

    def run(self) -> None:
        """Run the full image-driven preprocessing pipeline."""

        logger.info(
            "Starting image-driven preprocessing."
        )

        self.validate_inputs()

        # The dataset starts from the folders inside images/.
        image_specimens = self.scan_specimen_folders()

        # Excel is used only to provide family and genus.
        taxonomy = self.load_taxonomy()

        matched_specimens = (
            self.match_specimens_with_taxonomy(
                specimen_table=image_specimens,
                taxonomy=taxonomy,
            )
        )

        image_table = self.build_image_table(
            specimen_table=matched_specimens,
        )

        preprocessing_results = (
            self.preprocess_all_images(
                image_table=image_table,
            )
        )

        master = self.create_master_dataset(
            preprocessing_results=(
                preprocessing_results
            )
        )

        if master.empty:
            raise RuntimeError(
                "No images were successfully preprocessed."
            )
        supervised = self.create_supervised_dataset(
            master=master,
        )

        identified, unidentified = (
            self.create_specimen_level_files(
                master=master,
            )
        )

        self.save_preprocessing_reports(
            preprocessing_results=preprocessing_results,
            master=master,
            supervised=supervised,
            identified=identified,
            unidentified=unidentified,
            matched_specimens=matched_specimens,
        )

        logger.info(
            "Preprocessing completed successfully."
        )

        logger.info(
            "Image specimens: %s | "
            "Identified: %s | "
            "Unidentified: %s | "
            "Processed images: %s | "
            "Supervised images: %s",
            len(matched_specimens),
            len(identified),
            len(unidentified),
            len(master),
            len(supervised),
        )


def main() -> None:
    """Run preprocessing."""

    preprocessor = ImageDrivenPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
