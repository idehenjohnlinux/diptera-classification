"""Reporting utilities for Brachycera CNN experiments.




"""
# import modules
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.dataset import BrachyceraImageDataset


# ============================================================
# DATASET SUMMARY
# ============================================================

#Adding dataset summary for easy explanation 

def summarize_dataset(
    dataset: BrachyceraImageDataset,
    level: str,
) -> pd.DataFrame:
    """Create a class-level dataset summary.

    
    """

    dataframe = dataset.get_dataframe()

    required_columns = {
        level,
        "numCol",
        "processed_image_path",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot generate dataset summary. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot summarize an empty dataset."
        )

    summary = (
        dataframe
        .groupby(level, dropna=False)
        .agg(
            Images=(
                "processed_image_path",
                "count",
            ),
            Specimens=(
                "numCol",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                level: "Class",
            }
        )
        .sort_values(
            by="Class",
        )
        .reset_index(
            drop=True,
        )
    )

    total_images = int(
        summary["Images"].sum()
    )

    total_specimens = int(
        summary["Specimens"].sum()
    )

    summary["Image_Percentage"] = (
        summary["Images"]
        .div(total_images)
        .mul(100)
        .round(2)
    )

    summary["Specimen_Percentage"] = (
        summary["Specimens"]
        .div(total_specimens)
        .mul(100)
        .round(2)
    )

    return summary


# ============================================================
# TOTALS
# ============================================================
# calculation of the dataset total 
def calculate_dataset_totals(
    dataset: BrachyceraImageDataset,
) -> dict[str, int]:
    """Calculate overall image and specimen counts."""

    dataframe = dataset.get_dataframe()

    required_columns = {
        "numCol",
        "processed_image_path",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot calculate totals. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    return {
        "images": len(dataframe),
        "specimens": int(
            dataframe["numCol"].nunique()
        ),
    }


# ============================================================
# TERMINAL OUTPUT
# ============================================================
# creates the output 
def print_dataset_summary(
    summary: pd.DataFrame,
    title: str,
) -> None:
    """Print a formatted dataset summary table."""

    required_columns = {
        "Class",
        "Images",
        "Specimens",
        "Image_Percentage",
        "Specimen_Percentage",
    }

    missing_columns = (
        required_columns - set(summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Dataset summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    separator = "=" * 94
    row_separator = "-" * 94

    print()
    print(separator)
    print(title.upper())
    print(separator)

    print(
        f"{'Class':<32}"
        f"{'Images':>10}"
        f"{'Specimens':>13}"
        f"{'Image %':>14}"
        f"{'Specimen %':>16}"
    )

    print(row_separator)

    for row in summary.itertuples(
        index=False,
    ):
        class_name = str(row.Class)

        print(
            f"{class_name:<32}"
            f"{int(row.Images):>10}"
            f"{int(row.Specimens):>13}"
            f"{float(row.Image_Percentage):>13.2f}%"
            f"{float(row.Specimen_Percentage):>15.2f}%"
        )

    print(row_separator)

    total_images = int(
        summary["Images"].sum()
    )

    total_specimens = int(
        summary["Specimens"].sum()
    )

    print(
        f"{'Total':<32}"
        f"{total_images:>10}"
        f"{total_specimens:>13}"
        f"{100.00:>13.2f}%"
        f"{100.00:>15.2f}%"
    )

    print(separator)


# ============================================================
# FILE EXPORT
# ============================================================
# export the files 
def save_dataset_summary_csv(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save a dataset summary as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_dataset_summary_json(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    level: str,
    subset: str,
) -> Path:
    """Save a dataset summary as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, Any]] = (
        summary.to_dict(
            orient="records",
        )
    )

    payload = {
        "title": title,
        "taxonomic_level": level,
        "subset": subset,
        "number_of_classes": len(summary),
        "total_images": int(
            summary["Images"].sum()
        ),
        "total_specimens": int(
            summary["Specimens"].sum()
        ),
        "classes": records,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# COMPLETE REPORT CREATION
# ============================================================

def create_dataset_report(
    dataset: BrachyceraImageDataset,
    level: str,
    subset: str,
    output_directory: Path,
    print_report: bool = True,
) -> pd.DataFrame:
    """Create, display and save one dataset report.

    Parameters
    ----------
    dataset:
        Training or validation dataset.
    level:
        Taxonomic level, normally family or genus.
    subset:
        Dataset subset name, normally training or validation.
    output_directory:
        Directory where the report files are saved.
    print_report:
        Print the summary table to the terminal.

    Returns
    -------
    pandas.DataFrame
        Generated class summary.
    """

    normalized_subset = (
        str(subset)
        .strip()
        .lower()
    )

    valid_subsets = {
        "training",
        "validation",
        "train",
        "valid",
    }

    if normalized_subset not in valid_subsets:
        raise ValueError(
            f"Invalid subset: {subset!r}. "
            f"Expected one of {sorted(valid_subsets)}."
        )

    if normalized_subset == "train":
        normalized_subset = "training"

    if normalized_subset == "valid":
        normalized_subset = "validation"

    summary = summarize_dataset(
        dataset=dataset,
        level=level,
    )

    title = (
        f"{normalized_subset.capitalize()} "
        f"dataset summary"
    )

    if print_report:
        print_dataset_summary(
            summary=summary,
            title=title,
        )

    csv_path = (
        output_directory
        / f"{normalized_subset}_dataset_summary.csv"
    )

    json_path = (
        output_directory
        / f"{normalized_subset}_dataset_summary.json"
    )

    save_dataset_summary_csv(
        summary=summary,
        output_path=csv_path,
    )

    save_dataset_summary_json(
        summary=summary,
        output_path=json_path,
        title=title,
        level=level,
        subset=normalized_subset,
    )

    return summary
    
# ============================================================
# GLOBAL DATASET REPORT
# ============================================================

# create a global dataset reports which shows the results of all inputs 
def create_global_dataset_report(
    level: str,
    views: list[str],
    output_directory: Path,
    reference_fold: int = 1,
    print_report: bool = True,
) -> pd.DataFrame:
    """Create a global report across all selected anatomical views.

   
    
        Global class-level dataset summary.
    """

    from src.ml.dataset import create_datasets

    valid_levels = {
        "family",
        "genus",
    }

    normalized_level = str(level).strip().lower()

    if normalized_level not in valid_levels:
        raise ValueError(
            f"Invalid taxonomic level: {level!r}. "
            f"Expected one of {sorted(valid_levels)}."
        )

    valid_views = {
        "FDT",
        "FFF",
        "FLP",
        "FLT",
    }

    normalized_views = [
        str(view).strip().upper()
        for view in views
    ]

    invalid_views = (
        set(normalized_views)
        - valid_views
    )

    if invalid_views:
        raise ValueError(
            f"Invalid anatomical views: {sorted(invalid_views)}. "
            f"Expected values from {sorted(valid_views)}."
        )

    if not normalized_views:
        raise ValueError(
            "At least one anatomical view must be provided."
        )

    complete_dataframes: list[pd.DataFrame] = []

    for view_code in normalized_views:
        (
            training_dataset,
            validation_dataset,
            _,
        ) = create_datasets(
            level=normalized_level,
            view_code=view_code,
            validation_fold=reference_fold,
            train_transform=None,
            validation_transform=None,
            return_metadata=False,
            save_mapping=False,
        )

        training_dataframe = (
            training_dataset
            .get_dataframe()
            .copy()
        )

        validation_dataframe = (
            validation_dataset
            .get_dataframe()
            .copy()
        )

        view_dataframe = pd.concat(
            [
                training_dataframe,
                validation_dataframe,
            ],
            ignore_index=True,
        )

        view_dataframe["report_view"] = view_code

        complete_dataframes.append(
            view_dataframe
        )

    global_dataframe = pd.concat(
        complete_dataframes,
        ignore_index=True,
    )

    # Remove accidental duplicate image records.
    global_dataframe = (
        global_dataframe
        .drop_duplicates(
            subset=["processed_image_path"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    required_columns = {
        normalized_level,
        "numCol",
        "processed_image_path",
        "report_view",
    }

    missing_columns = (
        required_columns
        - set(global_dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot create the global report. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    if global_dataframe.empty:
        raise ValueError(
            "The global dataset is empty."
        )

    global_summary = (
        global_dataframe
        .groupby(
            normalized_level,
            dropna=False,
        )
        .agg(
            Images=(
                "processed_image_path",
                "count",
            ),
            Specimens=(
                "numCol",
                "nunique",
            ),
            Views=(
                "report_view",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                normalized_level: "Class",
            }
        )
        .sort_values("Class")
        .reset_index(drop=True)
    )

    total_images = int(
        global_summary["Images"].sum()
    )

    # Do not calculate the overall specimen total by summing the
    # class-level specimen values when a specimen could appear in
    # multiple rows because of inconsistent metadata.
    total_unique_specimens = int(
        global_dataframe["numCol"].nunique()
    )

    global_summary["Image_Percentage"] = (
        global_summary["Images"]
        .div(total_images)
        .mul(100)
        .round(2)
    )

    global_summary["Specimen_Percentage"] = (
        global_summary["Specimens"]
        .div(total_unique_specimens)
        .mul(100)
        .round(2)
    )

    global_summary["Images_Per_Specimen"] = (
        global_summary["Images"]
        .div(global_summary["Specimens"])
        .round(2)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / f"global_{normalized_level}_dataset_summary.csv"
    )

    json_path = (
        output_directory
        / f"global_{normalized_level}_dataset_summary.json"
    )

    global_summary.to_csv(
        csv_path,
        index=False,
    )

    payload = {
        "report_type": "global_dataset_summary",
        "taxonomic_level": normalized_level,
        "reference_fold": reference_fold,
        "included_views": normalized_views,
        "number_of_classes": int(
            len(global_summary)
        ),
        "total_images": total_images,
        "total_specimens": total_unique_specimens,
        "average_images_per_specimen": round(
            total_images / total_unique_specimens,
            2,
        ),
        "classes": global_summary.to_dict(
            orient="records",
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    if print_report:
        print_global_dataset_summary(
            summary=global_summary,
            level=normalized_level,
            views=normalized_views,
            total_unique_specimens=total_unique_specimens,
        )

    return global_summary

# prints out dataset summary file
def print_global_dataset_summary(
    summary: pd.DataFrame,
    level: str,
    views: list[str],
    total_unique_specimens: int,
) -> None:
    """Print the global dataset summary."""

    separator = "=" * 118
    row_separator = "-" * 118

    total_images = int(
        summary["Images"].sum()
    )

    average_images_per_specimen = (
        total_images / total_unique_specimens
        if total_unique_specimens > 0
        else 0.0
    )

    print()
    print(separator)
    print(
        f"GLOBAL {level.upper()} DATASET SUMMARY"
    )
    print(separator)

    print(
        f"Included views         : {', '.join(views)}"
    )
    print(
        f"Number of classes      : {len(summary)}"
    )
    print(
        f"Total images           : {total_images}"
    )
    print(
        f"Unique specimens       : {total_unique_specimens}"
    )
    print(
        f"Images per specimen    : "
        f"{average_images_per_specimen:.2f}"
    )

    print(row_separator)

    print(
        f"{'Class':<30}"
        f"{'Images':>10}"
        f"{'Specimens':>13}"
        f"{'Views':>9}"
        f"{'Image %':>13}"
        f"{'Specimen %':>16}"
        f"{'Img/Specimen':>17}"
    )

    print(row_separator)

    for row in summary.itertuples(index=False):
        print(
            f"{str(row.Class):<30}"
            f"{int(row.Images):>10}"
            f"{int(row.Specimens):>13}"
            f"{int(row.Views):>9}"
            f"{float(row.Image_Percentage):>12.2f}%"
            f"{float(row.Specimen_Percentage):>15.2f}%"
            f"{float(row.Images_Per_Specimen):>17.2f}"
        )

    print(row_separator)

    print(
        f"{'Total':<30}"
        f"{total_images:>10}"
        f"{total_unique_specimens:>13}"
        f"{len(views):>9}"
        f"{100.00:>12.2f}%"
        f"{100.00:>15.2f}%"
        f"{average_images_per_specimen:>17.2f}"
    )

    print(separator)
