from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = Path("results/evaluation/view_comparison.csv")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

# Use specimen-level evaluation
df = df[df["evaluation_level"] == "specimen"].copy()

# Convert to percentages
metrics = {
    "mean_accuracy": "Accuracy",
    "mean_balanced_accuracy": "Balanced Accuracy",
    "mean_precision_macro": "Precision",
    "mean_recall_macro": "Recall",
    "mean_f1_macro": "Macro F1-score",
}

for metric in metrics:
    df[metric] *= 100

# Better labels
view_names = {
    "FLT": "Lateral do corpo (FLT)",
    "FLP": " lateral parcial (FLP)",
    "FFF": "frontal da cabeça (FFF)",
    "FDT": "completa dorsal total (FDT)",
}

df["View"] = df["view_code"].map(view_names)

# Order by accuracy
df = df.sort_values(
    "mean_accuracy",
    ascending=True,
)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.15

y = range(len(df))

offsets = [
    -2 * bar_width,
    -1 * bar_width,
    0,
    1 * bar_width,
    2 * bar_width,
]

for offset, (column, label) in zip(offsets, metrics.items()):

    bars = ax.barh(
        [i + offset for i in y],
        df[column],
        height=bar_width,
        label=label,
    )

    for bar in bars:

        value = bar.get_width()

        ax.text(
            value + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{value:.1f}%",
            va="center",
            fontsize=8,
        )

ax.set_yticks(y)
ax.set_yticklabels(df["View"])

ax.set_xlim(0, 100)

ax.set_xlabel("Performance (%)", fontsize=12)
ax.set_ylabel("Anatomical View", fontsize=12)

ax.set_title(
    "Comparação de Vistas Anatómicas\n"
    "Desempenho de classificação ao nível de espécime",
    fontsize=15,
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.30,
)

ax.legend(
    title="Metric",
    loc="lower right",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "anatomical_view_comparison.png",
    dpi=300,
)

plt.savefig(
    OUTPUT_DIR / "anatomical_view_comparison.pdf",
)

plt.savefig(
    OUTPUT_DIR / "anatomical_view_comparison.svg",
)

plt.show()
