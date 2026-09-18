from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "metadata" / "master_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_summary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "training_dataset_summary.csv"
VIEW_FILE = OUTPUT_DIR / "training_images_per_view.csv"
FAMILY_FILE = OUTPUT_DIR / "family_distribution.csv"
GENUS_FILE = OUTPUT_DIR / "genus_distribution.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def to_boolean(series: pd.Series) -> pd.Series:
    """Convert common Boolean representations to True or False."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "sim"])
    )


def clean_labels(series: pd.Series) -> pd.Series:
    """Clean taxonomic labels and remove invalid values."""
    cleaned = series.astype("string").str.strip()

    invalid_values = {
        "",
        "nan",
        "none",
        "null",
        "unknown",
        "unlabelled",
        "unidentified",
        "not identified",
        "não identificado",
        "nao identificado",
    }

    return cleaned[
        cleaned.notna()
        & ~cleaned.str.lower().isin(invalid_values)
    ]


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset was not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)
data.columns = data.columns.str.strip()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "numCol",
    "family",
    "genus",
    "use_for_family",
    "use_for_genus",
    "view_code",
    "processed_image_path",
}

missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise KeyError(
        f"Missing columns in master_dataset.csv: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

data["use_for_family"] = to_boolean(data["use_for_family"])
data["use_for_genus"] = to_boolean(data["use_for_genus"])

data["numCol"] = data["numCol"].astype("string").str.strip()
data["family"] = data["family"].astype("string").str.strip()
data["genus"] = data["genus"].astype("string").str.strip()

data["view_code"] = (
    data["view_code"]
    .astype("string")
    .str.strip()
    .str.upper()
)

family_data = data.loc[data["use_for_family"]].copy()
genus_data = data.loc[data["use_for_genus"]].copy()

not_used_data = data.loc[
    ~data["use_for_family"]
    & ~data["use_for_genus"]
].copy()


# ============================================================
# GENERAL COUNTS
# ============================================================

total_images = len(data)
total_specimens = data["numCol"].dropna().nunique()

family_training_images = len(family_data)
family_training_specimens = (
    family_data["numCol"].dropna().nunique()
)

genus_training_images = len(genus_data)
genus_training_specimens = (
    genus_data["numCol"].dropna().nunique()
)

excluded_images = len(not_used_data)
excluded_specimens = (
    not_used_data["numCol"].dropna().nunique()
)

family_labels = clean_labels(family_data["family"])
genus_labels = clean_labels(genus_data["genus"])

n_training_families = family_labels.str.casefold().nunique()
n_training_genera = genus_labels.str.casefold().nunique()


# ============================================================
# IMAGES PER VIEW
# ============================================================

expected_views = ["FLT", "FLP", "FFF", "FDT"]

family_view_counts = (
    family_data["view_code"]
    .value_counts()
    .reindex(expected_views, fill_value=0)
)

genus_view_counts = (
    genus_data["view_code"]
    .value_counts()
    .reindex(expected_views, fill_value=0)
)

view_summary = pd.DataFrame({
    "View": expected_views,
    "Family_training_images": [
        int(family_view_counts[view])
        for view in expected_views
    ],
    "Genus_training_images": [
        int(genus_view_counts[view])
        for view in expected_views
    ],
})


# ============================================================
# TAXONOMIC DISTRIBUTIONS
# ============================================================

family_distribution = (
    family_data.loc[
        ~family_data["family"]
        .str.lower()
        .isin(["unlabelled", "unidentified", "nan"])
    ]
    .groupby("family")
    .agg(
        Specimens=("numCol", "nunique"),
        Images=("processed_image_path", "count"),
    )
    .sort_values(
        ["Specimens", "Images"],
        ascending=False,
    )
    .reset_index()
)

genus_distribution = (
    genus_data.loc[
        ~genus_data["genus"]
        .str.lower()
        .isin(["unlabelled", "unidentified", "nan"])
    ]
    .groupby("genus")
    .agg(
        Specimens=("numCol", "nunique"),
        Images=("processed_image_path", "count"),
    )
    .sort_values(
        ["Specimens", "Images"],
        ascending=False,
    )
    .reset_index()
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame([
    {
        "Characteristic": "Total processed images",
        "Value": total_images,
    },
    {
        "Characteristic": "Total specimens with processed images",
        "Value": total_specimens,
    },
    {
        "Characteristic": "Images eligible for family training",
        "Value": family_training_images,
    },
    {
        "Characteristic": "Specimens eligible for family training",
        "Value": family_training_specimens,
    },
    {
        "Characteristic": "Families used for training",
        "Value": n_training_families,
    },
    {
        "Characteristic": "Images eligible for genus training",
        "Value": genus_training_images,
    },
    {
        "Characteristic": "Specimens eligible for genus training",
        "Value": genus_training_specimens,
    },
    {
        "Characteristic": "Genera used for training",
        "Value": n_training_genera,
    },
    {
        "Characteristic": "Images excluded from both models",
        "Value": excluded_images,
    },
    {
        "Characteristic": "Specimens excluded from both models",
        "Value": excluded_specimens,
    },
])


# ============================================================
# SAVE RESULTS
# ============================================================

summary.to_csv(SUMMARY_FILE, index=False)
view_summary.to_csv(VIEW_FILE, index=False)
family_distribution.to_csv(FAMILY_FILE, index=False)
genus_distribution.to_csv(GENUS_FILE, index=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== TRAINING DATASET SUMMARY ==========\n")
print(summary.to_string(index=False))

print("\n========== TRAINING IMAGES PER VIEW ==========\n")
print(view_summary.to_string(index=False))

print("\nFiles saved to:")

for output_file in [
    SUMMARY_FILE,
    VIEW_FILE,
    FAMILY_FILE,
    GENUS_FILE,
]:
    print(f"  {output_file}")
