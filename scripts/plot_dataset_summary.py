import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# DATA
# -----------------------------
views = ["FLT", "FLP", "FFF", "FDT"]

family_images = [327, 507, 378, 287]
genus_images = [184, 294, 211, 172]

total_processed = 2429
family_training = 1506
genus_training = 862
excluded = 923

# -----------------------------
# FIGURE 7.2
# Images per Anatomical View
# -----------------------------

x = np.arange(len(views))
width = 0.35

plt.figure(figsize=(8,6))

plt.bar(
    x - width/2,
    family_images,
    width,
    label="Family",
)

plt.bar(
    x + width/2,
    genus_images,
    width,
    label="Genus",
)

plt.xticks(x, views, fontsize=12)
plt.yticks(fontsize=11)

plt.xlabel("Anatomical View", fontsize=13)
plt.ylabel("Number of Images", fontsize=13)

plt.title(
    "Images Available per Anatomical View",
    fontsize=14,
    weight="bold"
)

plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "figure_7_2_views.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# FIGURE 7.3
# Dataset Composition
# -----------------------------

labels = [
    "Family Training",
    "Genus Training",
    "Excluded",
]

sizes = [
    family_training,
    genus_training,
    excluded,
]

plt.figure(figsize=(7,7))

plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
)

plt.title(
    "Dataset Composition After Taxonomic Filtering",
    fontsize=14,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    "figure_7_3_dataset_composition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figures successfully created.")
