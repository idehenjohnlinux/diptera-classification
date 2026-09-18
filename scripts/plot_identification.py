#!/usr/bin/env python3
#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

INPUT = Path(
    "results/production/hierarchical/identification/identification_summary.json"
)

OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Load JSON
# ----------------------------------------------------------

with open(INPUT) as f:
    summary = json.load(f)

categories = [
    "Processado com sucesso",
    "Requer revisão de perito",
    "Sugestões de género",
    "Identificações confirmadas",
    "Sugestões de família",
    "Identificações parciais",
    "Erros de processamento"
]

values = [
    summary["successful_identifications"],
    summary["specimens_requiring_expert_review"],
    summary["previously_missing_genus_labels_receiving_suggestion"],
    summary["existing_identifications_confirmed"],
    summary["previously_missing_family_labels_receiving_suggestion"],
    summary["partial_identifications_completed"],
    summary["errors"],
]

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(10,6))

bars = ax.barh(categories, values)

ax.invert_yaxis()

ax.set_xlabel("Número de espécimes", fontsize=12)
ax.set_title(
    "Resultados Operacionais do Fluxo de Identificação Automática",
    fontsize=14,
    pad=15,
)

# Add values at end of each bar
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 3,
        bar.get_y() + bar.get_height()/2,
        f"{int(width)}",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(
    OUTPUT / "figure_7_8_operational_summary.png",
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    OUTPUT / "figure_7_8_operational_summary.pdf",
    bbox_inches="tight",
)

plt.savefig(
    OUTPUT / "figure_7_8_operational_summary.svg",
    bbox_inches="tight",
)

plt.close()

print("Figure 7.8 saved successfully.")
