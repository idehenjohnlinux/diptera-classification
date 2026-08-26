# configfile: "config/config.yaml"

# RULE ALL

rule all:
    input:
        # Dataset preparation
        "metadata/preprocessing_summary.json",

        # Cross-validation
        "metadata/cross_validation/cross_validation_summary.json",

        # CNN evaluation
        "results/evaluation/all_results.csv",
        "results/evaluation/architecture_comparison.csv",
        "results/evaluation/view_comparison.csv",
        "results/evaluation/best_models.csv",

        # Hierarchical model
        "results/production/hierarchical/best_model.pt",

        # Final identification
        "results/production/hierarchical/identification/specimen_identifications.csv",
        "results/production/hierarchical/identification/identification_summary.json",

        # Final hierarchical evaluation
        "results/production/hierarchical/evaluation/evaluation_summary.json",
        "results/production/hierarchical/evaluation/family_per_class_metrics.csv",
        "results/production/hierarchical/evaluation/genus_per_class_metrics.csv",

        # Visualisations
        "results/figures/anatomical_view_comparison.png",
        "figure_7_3_architecture_comparison.png",
        "results/figures/family_confusion_matrix_blue.png",
        "results/figures/genus_confusion_matrix_blue.png",
        
        # R visualisation
        "results/plots/architecture_performance_boxplots.png",
        "results/plots/architecture_performance_boxplots.pdf"
        
# ============================================================
# 1. DATASET AUDIT
# ============================================================

rule dataset_audit:
    input:
        metadata="metadata/BaseDadosCol-2026.xlsx"
    output:
        csv="metadata/specimen_image_check.csv",
        json="metadata/dataset_summary.json"
    shell:
        """
        PYTHONPATH=. python -m src.core.dataset_audit
        """
        
# ============================================================
# 2. DATA VALIDATION
# ============================================================

rule validate_dataset:
    input:
        audit="metadata/specimen_image_check.csv"
    output:
        validation="metadata/image_validation_report.csv",
        training="metadata/training_dataset.csv"
    shell:
        """
        PYTHONPATH=. python -m src.core.validator
        """
        
# ============================================================
# 3. DATA PREPARATION
# ============================================================

rule preprocess:
    input:
        validation="metadata/image_validation_report.csv",
        training="metadata/training_dataset.csv"
    output:
        master="metadata/master_dataset.csv",
        supervised="metadata/supervised_dataset.csv",
        identification="metadata/identification/identification_dataset.csv",
        summary_csv="metadata/preprocessing_summary.csv",
        summary_json="metadata/preprocessing_summary.json"
    shell:
        """
        PYTHONPATH=. python -m src.core.preprocess
        """


# ============================================================
# 4. FIVE-FOLD CROSS-VALIDATION(STRACTIFIED)
# ============================================================

rule cross_validation:
    input:
        dataset="metadata/supervised_dataset.csv"
    output:
        family="metadata/cross_validation/family_folds.csv",
        genus="metadata/cross_validation/genus_folds.csv",
        family_assignments="metadata/cross_validation/family_specimen_fold_assignments.csv",
        genus_assignments="metadata/cross_validation/genus_specimen_fold_assignments.csv",
        summary="metadata/cross_validation/cross_validation_summary.json"
    shell:
        """
        PYTHONPATH=. python -m src.core.create_folds
        """
        
# ============================================================
# 5. CNN TRAINING
# ============================================================

rule train_cnn_models:
    input:
        family_folds="metadata/cross_validation/family_folds.csv",
        genus_folds="metadata/cross_validation/genus_folds.csv"
    output:
        status_csv="results/training/all_experiments_status.csv",
        status_json="results/training/all_experiments_status.json"
    threads: 4
    shell:
        """
        PYTHONPATH=. python -m src.ml.train_all \
            --levels family genus \
            --views FDT FFF FLP FLT \
            --folds 0 1 2 3 4 \
            --architectures efficientnet_b0 resnet18 mobilenet_v3_large \
            --epochs 15 \
            --output-root results/training \
            --skip-completed
        """

# ============================================================
# 6. AGGREGATE AND COMPARE CNN RESULTS
# ============================================================

rule aggregate_results:
    input:
        training="results/training/all_experiments_status.csv"
    output:
        all_results="results/evaluation/all_results.csv",
        architecture="results/evaluation/architecture_comparison.csv",
        views="results/evaluation/view_comparison.csv",
        taxonomy="results/evaluation/taxonomic_level_comparison.csv",
        detailed="results/evaluation/detailed_model_comparison.csv",
        rankings="results/evaluation/model_rankings.csv",
        best="results/evaluation/best_models.csv",
        summary="results/evaluation/aggregation_summary.json"
    shell:
        """
        PYTHONPATH=. python -m src.ml.aggregate_results \
            --training-root results/training \
            --output-directory results/evaluation
        """

# ============================================================
# 7. TRAIN FINAL HIERARCHICAL MODEL
# ============================================================

rule train_hierarchical_model:
    input:
        dataset="metadata/identification/identification_dataset.csv",
        model_selection="results/evaluation/best_models.csv",
        taxonomy="metadata/class_mappings/hierarchical/taxonomy_mappings.json"
    output:
        best_model="results/production/hierarchical/best_model.pt",
        last_model="results/production/hierarchical/last_model.pt",
        history_csv="results/production/hierarchical/history.csv",
        history_json="results/production/hierarchical/history.json",
        summary="results/production/hierarchical/training_summary.json"
    shell:
        """
        PYTHONPATH=. python -m src.ml.hierarchical_trainer
        """

# ============================================================
# 8. HIERARCHICAL IDENTIFICATION
# ============================================================

rule hierarchical_identification:
    input:
        dataset="metadata/identification/identification_dataset.csv",
        checkpoint="results/production/hierarchical/best_model.pt"
    output:
        csv="results/production/hierarchical/identification/specimen_identifications.csv",
        json="results/production/hierarchical/identification/specimen_identifications.json",
        txt="results/production/hierarchical/identification/specimen_identifications.txt",
        summary="results/production/hierarchical/identification/identification_summary.json",
        errors="results/production/hierarchical/identification/identification_errors.csv"
    shell:
        """
        PYTHONPATH=. python -m src.ml.hierarchical_identifier \
            --dataset {input.dataset} \
            --checkpoint {input.checkpoint} \
            --output-directory results/production/hierarchical/identification
        """

# ============================================================
# 9. HIERARCHICAL MODEL EVALUATION
# ============================================================

rule evaluate_hierarchical_model:
    input:
        identifications=(
            "results/production/hierarchical/"
            "identification/specimen_identifications.csv"
        )
    output:
        summary_json=(
            "results/production/hierarchical/"
            "evaluation/evaluation_summary.json"
        ),
        summary_txt=(
            "results/production/hierarchical/"
            "evaluation/evaluation_summary.txt"
        ),
        family_metrics=(
            "results/production/hierarchical/"
            "evaluation/family_per_class_metrics.csv"
        ),
        genus_metrics=(
            "results/production/hierarchical/"
            "evaluation/genus_per_class_metrics.csv"
        ),
        family_confusion=(
            "results/production/hierarchical/"
            "evaluation/family_confusion_matrix.csv"
        ),
        genus_confusion=(
            "results/production/hierarchical/"
            "evaluation/genus_confusion_matrix.csv"
        ),
        review=(
            "results/production/hierarchical/"
            "evaluation/review_priority.csv"
        )
    shell:
        """
        PYTHONPATH=. python -m src.ml.hierarchical_evaluator \
            --identifications {input.identifications} \
            --output-directory results/production/hierarchical/evaluation
        """
# ============================================================
# 10. VISUALISATIONS
# ============================================================

rule plot_anatomical_views:
    input:
        "results/evaluation/view_comparison.csv"
    output:
        "results/figures/anatomical_view_comparison.png"
    conda:
        "workflow/envs/python_plot.yml"
    shell:
        """
        MPLBACKEND=Agg PYTHONPATH=. python scripts/plot_anatomy.py
        """

rule plot_architecture_comparison:
    input:
        "results/evaluation/architecture_comparison.csv"
    output:
        "figure_7_3_architecture_comparison.png"
    conda:
        "workflow/envs/python_plot.yml"
    shell:
        """
        MPLBACKEND=Agg PYTHONPATH=. python scripts/plot_architecture_comparison.py
        """

rule plot_confusion_matrices:
    input:
        family="results/production/hierarchical/evaluation/family_confusion_matrix.csv",
        genus="results/production/hierarchical/evaluation/genus_confusion_matrix.csv"
    output:
        family="results/figures/family_confusion_matrix_blue.png",
        genus="results/figures/genus_confusion_matrix_blue.png"
    conda:
        "workflow/envs/python_plot.yml"
    shell:
        """
        MPLBACKEND=Agg PYTHONPATH=. python scripts/plot_confusion_matrices.py
        """
        
rule plot_architecture_boxplots:
    input:
        "results/evaluation/all_results.csv"
    output:
        png="results/plots/architecture_performance_boxplots.png",
        pdf="results/plots/architecture_performance_boxplots.pdf"
    conda:
        "workflow/envs/r_plot.yml"
    shell:
        """
        Rscript scripts/plot_architecture_boxplots.R
        """
