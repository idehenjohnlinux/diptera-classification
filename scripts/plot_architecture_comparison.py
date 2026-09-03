import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "results/evaluation/architecture_comparison.csv"

OUTPUT_TABLE = "table_7_3_architecture_comparison.csv"
OUTPUT_FIGURE = "figure_7_3_architecture_comparison.png"

# ============================================================
# Load results
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Keep only specimen-level evaluation
df = df[df["evaluation_level"].str.lower() == "specimen"].copy()

# ============================================================
# Columns to use
# ============================================================

metrics = [
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_precision_macro",
    "mean_recall_macro",
    "mean_f1_macro",
]

metric_labels = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Macro F1",
]

# ============================================================
# Convert to percentages
# ============================================================

for metric in metrics:
    df[metric] = (df[metric] * 100).round(2)

# ============================================================
# Prepare dissertation table
# ============================================================

table = df[
    [
        "architecture",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "mean_precision_macro",
        "mean_recall_macro",
        "mean_f1_macro",
    ]
].copy()

table.columns = [
    "Architecture",
    "Accuracy (%)",
    "Balanced Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "Macro F1-score (%)",
]

table = table.sort_values(
    by="Accuracy (%)",
    ascending=False,
)

table.to_csv(
    OUTPUT_TABLE,
    index=False,
)

print("\nTable saved as:")
print(OUTPUT_TABLE)

print("\n")
print(table)

# ============================================================
# Figure 7.3
# ============================================================

architectures = table["Architecture"]

x = np.arange(len(architectures))
width = 0.15

plt.figure(figsize=(12,7))

for i, metric in enumerate(table.columns[1:]):

    plt.bar(
        x + (i - 2) * width,
        table[metric],
        width,
        label=metric.replace(" (%)", ""),
    )

plt.xticks(
    x,
    architectures,
    fontsize=12,
)

plt.yticks(fontsize=11)

plt.ylabel(
    "Performance (%)",
    fontsize=13,
)

plt.xlabel(
    "CNN Architecture",
    fontsize=13,
)

plt.title(
    "Comparação das arquiteturas de CNN ao nível de espécime",
    fontsize=15,
    weight="bold",
)

plt.ylim(0,100)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nFigure saved as:")
print(OUTPUT_FIGURE)

print("\nDone.")
