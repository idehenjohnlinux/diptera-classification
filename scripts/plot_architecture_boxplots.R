# ============================================================
# Combined boxplots for CNN architecture comparison
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

input_file <- "results/evaluation/all_results.csv"
output_dir <- "results/plots"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# Read results
# ------------------------------------------------------------

results <- read_csv(input_file, show_col_types = FALSE)

# Check required columns
required_columns <- c(
  "architecture",
  "accuracy",
  "balanced_accuracy",
  "precision_macro",
  "recall_macro",
  "f1_macro"
)

missing_columns <- setdiff(required_columns, names(results))

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

# ------------------------------------------------------------
# Rename architectures
# ------------------------------------------------------------

results <- results %>%
  mutate(
    architecture = recode(
      architecture,
      "efficientnet_b0" = "EfficientNet-B0",
      "mobilenet_v3_large" = "MobileNetV3 Large",
      "resnet18" = "ResNet18"
    ),
    architecture = factor(
      architecture,
      levels = c(
        "EfficientNet-B0",
        "MobileNetV3 Large",
        "ResNet18"
      )
    )
  )

# ------------------------------------------------------------
# Convert metrics from wide to long format
# ------------------------------------------------------------

plot_data <- results %>%
  select(
    architecture,
    accuracy,
    balanced_accuracy,
    precision_macro,
    recall_macro,
    f1_macro
  ) %>%
  pivot_longer(
    cols = -architecture,
    names_to = "metric",
    values_to = "score"
  ) %>%
  mutate(
    metric = recode(
      metric,
      "accuracy" = "Accuracy",
      "balanced_accuracy" = "Accuracy Balanceada",
      "precision_macro" = "Macro Precisão",
      "recall_macro" = "Macro Recall",
      "f1_macro" = "Macro F1-score"
    ),
    metric = factor(
      metric,
      levels = c(
        "Accuracy",
        "Accuracy Balanceada",
        "Macro Precisão",
        "Macro Recall",
        "Macro F1-score"
      )
    ),
    score = score * 100
  ) %>%
  filter(!is.na(score))

# ------------------------------------------------------------
# Calculate means to display as points
# ------------------------------------------------------------

mean_values <- plot_data %>%
  group_by(architecture, metric) %>%
  summarise(
    mean_score = mean(score, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# Create combined figure
# ------------------------------------------------------------

boxplot_figure <- ggplot(
  plot_data,
  aes(
    x = architecture,
    y = score,
    fill = architecture
  )
) +
  geom_boxplot(
    width = 0.65,
    outlier.shape = 21,
    outlier.size = 1.8,
    outlier.alpha = 0.65
  ) +
  geom_point(
    data = mean_values,
    aes(
      x = architecture,
      y = mean_score
    ),
    inherit.aes = FALSE,
    shape = 23,
    size = 2.7,
    fill = "white",
    colour = "black"
  ) +
  facet_wrap(
    ~ metric,
    ncol = 3
  ) +
  scale_y_continuous(
    name = "Desempenho (%)",
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = NULL,
    title = "Desempenho de classificação por arquitetura de CNN",
    subtitle = "Distribuição de métricas de avaliação entre experimentos de validação",
    caption = "As caixas representam o intervalo interquartil; as linhas horizontais indicam as medianas; os losangos brancos indicam as médias."
  ) +
  guides(fill = "none") +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 15,
      hjust = 0.5
    ),
    plot.subtitle = element_text(
      size = 12,
      hjust = 0.5
    ),
    plot.caption = element_text(
      size = 12,
      hjust = 0
    ),
    strip.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.text.x = element_text(
      angle = 25,
      hjust = 1,
      face = "bold"
    ),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(
      face = "bold",
      size = 11
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1, "lines")
  )

print(boxplot_figure)

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

ggsave(
  filename = file.path(
    output_dir,
    "architecture_performance_boxplots.png"
  ),
  plot = boxplot_figure,
  width = 13,
  height = 8,
  units = "in",
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = file.path(
    output_dir,
    "architecture_performance_boxplots.pdf"
  ),
  plot = boxplot_figure,
  width = 13,
  height = 8,
  units = "in",
  bg = "white"
)

cat("\nFigure successfully created:\n")
cat(
  file.path(
    output_dir,
    "architecture_performance_boxplots.png"
  ),
  "\n"
)
cat(
  file.path(
    output_dir,
    "architecture_performance_boxplots.pdf"
  ),
  "\n"
)
