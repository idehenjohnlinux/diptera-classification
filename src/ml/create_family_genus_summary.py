from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "results/production/hierarchical/evaluation/review_priority.csv"
)

OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "specimen_prediction_table.csv"


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "specimen_id",
    "metadata_family",
    "metadata_genus",
    "predicted_family",
    "family_confidence",
    "predicted_genus",
    "genus_confidence",
    "family_correct",
    "genus_correct",
    "priority",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# PREPARE TABLE
# ============================================================

table = df[required_columns].copy()

# Convert confidence values from proportions to percentages
table["family_confidence"] = (
    table["family_confidence"] * 100
).round(2)

table["genus_confidence"] = (
    table["genus_confidence"] * 100
).round(2)

# Convert Boolean values to readable labels
table["family_correct"] = table["family_correct"].map(
    {
        True: "Yes",
        False: "No",
        1: "Yes",
        0: "No",
    }
)

table["genus_correct"] = table["genus_correct"].map(
    {
        True: "Yes",
        False: "No",
        1: "Yes",
        0: "No",
    }
)

# Rename columns for the dissertation table
table = table.rename(
    columns={
        "specimen_id": "Specimen ID",
        "metadata_family": "Metadata Family",
        "predicted_family": "Predicted Family",
        "family_confidence": "Family Confidence (%)",
        "family_correct": "Family Correct",
        "metadata_genus": "Metadata Genus",
        "predicted_genus": "Predicted Genus",
        "genus_confidence": "Genus Confidence (%)",
        "genus_correct": "Genus Correct",
        "priority": "Review Priority",
    }
)

# Arrange columns
table = table[
    [
        "Specimen ID",
        "Metadata Family",
        "Predicted Family",
        "Family Confidence (%)",
        "Family Correct",
        "Metadata Genus",
        "Predicted Genus",
        "Genus Confidence (%)",
        "Genus Correct",
        "Review Priority",
    ]
]

# Sort by specimen identifier
table = table.sort_values(
    by="Specimen ID",
    na_position="last"
)

table.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Table created successfully: {OUTPUT_FILE}")
print(f"Number of specimens: {len(table)}")
