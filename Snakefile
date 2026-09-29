
configfile: "config/config.yaml"


# ============================================================
# CONFIGURATION
# ============================================================

METADATA = "metadata"
RESULTS = "results"

CV_DIR = f"{METADATA}/cross_validation"
TRAINING_DIR = f"{RESULTS}/training"
EVALUATION_DIR = f"{RESULTS}/evaluation"

HIERARCHICAL_DIR = f"{RESULTS}/production/hierarchical"
IDENTIFICATION_DIR = f"{HIERARCHICAL_DIR}/identification"
HIERARCHICAL_EVAL_DIR = f"{HIERARCHICAL_DIR}/evaluation"

FIGURES_DIR = f"{RESULTS}/figures"
PLOTS_DIR = f"{RESULTS}/plots"

VIEWS = config.get(
    "views",
    ["FDT", "FFF", "FLP", "FLT"],
)

FOLDS = config.get(
    "folds",
    [ 1, 2, 3, 4,5],
)

ARCHITECTURES = config.get(
    "architectures",
    [
        "efficientnet_b0",
        "resnet18",
        "mobilenet_v3_large",
    ],
)

EPOCHS = config.get("epochs", 15)
THREADS = config.get("threads", 4)

PYTHON = "PYTHONPATH=. python"
# Global Conda Environment Declaration (Applies to all rules)
conda: "workflow/envs/envs.yaml"

# ============================================================
# FINAL TARGETS
# ============================================================

rule all:
    input:
        f"{METADATA}/dataset_summary.json",
        f"{METADATA}/image_validation_report.csv",
        f"{METADATA}/preprocessing_summary.json",
        f"{CV_DIR}/cross_validation_summary.json",
        f"{TRAINING_DIR}/all_experiments_status.csv",
        f"{TRAINING_DIR}/all_experiments_status.json",
        f"{EVALUATION_DIR}/all_results.csv",
        f"{EVALUATION_DIR}/architecture_comparison.csv",
        f"{EVALUATION_DIR}/view_comparison.csv",
        f"{EVALUATION_DIR}/taxonomic_level_comparison.csv",
        f"{EVALUATION_DIR}/detailed_model_comparison.csv",
        f"{EVALUATION_DIR}/model_rankings.csv",
        f"{EVALUATION_DIR}/best_models.csv",
        f"{EVALUATION_DIR}/aggregation_summary.json",
        # Hierarchical targets
        f"{HIERARCHICAL_DIR}/best_model.pt",
        f"{HIERARCHICAL_DIR}/last_model.pt",
        f"{HIERARCHICAL_DIR}/training_summary.json",
        f"{IDENTIFICATION_DIR}/specimen_identifications.csv",
        f"{IDENTIFICATION_DIR}/identification_errors.csv",
        "results/tables/specimen_prediction_table.csv",
        
        

# ============================================================
# 1. DATASET AUDIT
# ============================================================

rule dataset_audit:
    input:
        metadata=f"{METADATA}/BaseDadosCol-2026.xlsx"

    output:
        csv=f"{METADATA}/specimen_image_check.csv",
        json=f"{METADATA}/dataset_summary.json"

    log:
        f"{METADATA}/logs/dataset_audit.log"

    params:
        metadata_dir=METADATA

    shell:
        """
        mkdir -p {params.metadata_dir}/logs
        {PYTHON} -m src.core.dataset_audit > {log} 2>&1
        """


# ============================================================
# 2. DATA VALIDATION
# ============================================================

rule validate_dataset:
    input:
        audit=f"{METADATA}/specimen_image_check.csv"

    output:
        validation=f"{METADATA}/image_validation_report.csv",
        training=f"{METADATA}/training_dataset.csv"

    log:
        f"{METADATA}/logs/validate_dataset.log"

    params:
        metadata_dir=METADATA

    shell:
        """
        mkdir -p {params.metadata_dir}/logs
        {PYTHON} -m src.core.validator > {log} 2>&1
        """


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

rule preprocess:
    input:
        validation=f"{METADATA}/image_validation_report.csv",
        training=f"{METADATA}/training_dataset.csv"

    output:
        master=f"{METADATA}/master_dataset.csv",
        supervised=f"{METADATA}/supervised_dataset.csv",
        summary_csv=f"{METADATA}/preprocessing_summary.csv",
        summary_json=f"{METADATA}/preprocessing_summary.json"

    log:
        f"{METADATA}/logs/preprocess.log"

    params:
        metadata_dir=METADATA

    shell:
        """
        mkdir -p {params.metadata_dir}/logs
        {PYTHON} -m src.core.preprocess > {log} 2>&1
        """


# ============================================================
# 4. FIVE-FOLD CROSS-VALIDATION
# ============================================================

rule cross_validation:
    input:
        dataset=f"{METADATA}/master_dataset.csv"
    output:
        family=f"{CV_DIR}/family_folds.csv",
        genus=f"{CV_DIR}/genus_folds.csv",
        family_assignments=f"{CV_DIR}/family_specimen_fold_assignments.csv",
        genus_assignments=f"{CV_DIR}/genus_specimen_fold_assignments.csv",
        summary=f"{CV_DIR}/cross_validation_summary.json"
    log:
        f"{METADATA}/logs/cross_validation.log"
    shell:
        """
        mkdir -p {CV_DIR}
        mkdir -p {METADATA}/logs
        PYTHONPATH=. python -m src.core.cross_validation \
            --input {input.dataset} \
            --output-dir {CV_DIR} \
            > {log} 2>&1
        """

# ============================================================
# 5. CNN TRAINING
# ============================================================

rule train_cnn_models:
    input:
        dataset=f"{METADATA}/master_dataset.csv",
        family_folds=f"{CV_DIR}/family_folds.csv",
        genus_folds=f"{CV_DIR}/genus_folds.csv"

    output:
        status_csv=f"{TRAINING_DIR}/all_experiments_status.csv",
        status_json=f"{TRAINING_DIR}/all_experiments_status.json"

    threads:
        THREADS

    params:
        views=" ".join(VIEWS),
        folds=" ".join(map(str, FOLDS)),
        architectures=" ".join(ARCHITECTURES),
        epochs=EPOCHS,
        training_dir=TRAINING_DIR

    log:
        f"{TRAINING_DIR}/training.log"

    shell:
        """
        mkdir -p {params.training_dir}

        {PYTHON} -m src.ml.train_all \
            --levels family genus \
            --views {params.views} \
            --folds {params.folds} \
            --architectures {params.architectures} \
            --epochs {params.epochs} \
            --output-root {params.training_dir} \
            --skip-completed \
            > {log} 2>&1
        """

# ============================================================
# 6. AGGREGATE CNN RESULTS
# ============================================================

rule aggregate_results:
    input:
        training=f"{TRAINING_DIR}/all_experiments_status.csv"

    output:
        all_results=f"{EVALUATION_DIR}/all_results.csv",
        architecture=f"{EVALUATION_DIR}/architecture_comparison.csv",
        views=f"{EVALUATION_DIR}/view_comparison.csv",
        taxonomy=f"{EVALUATION_DIR}/taxonomic_level_comparison.csv",
        detailed=f"{EVALUATION_DIR}/detailed_model_comparison.csv",
        rankings=f"{EVALUATION_DIR}/model_rankings.csv",
        best=f"{EVALUATION_DIR}/best_models.csv",
        summary=f"{EVALUATION_DIR}/aggregation_summary.json"

    log:
        f"{EVALUATION_DIR}/aggregation.log"

    params:
        training_dir=TRAINING_DIR,
        evaluation_dir=EVALUATION_DIR

    shell:
        """
        mkdir -p {params.evaluation_dir}

        {PYTHON} -m src.ml.aggregate_results \
            --training-root {params.training_dir} \
            --output-directory {params.evaluation_dir} \
            > {log} 2>&1
        """

# ============================================================
# HIERARCHICAL MODEL TRAINING
# ============================================================

rule train_hierarchical_model:
    input:
        master_dataset=f"{METADATA}/master_dataset.csv"
    output:
        checkpoint=f"{HIERARCHICAL_DIR}/best_model.pt",
        last_checkpoint=f"{HIERARCHICAL_DIR}/last_model.pt",
        summary=f"{HIERARCHICAL_DIR}/training_summary.json"
    log:
        f"{HIERARCHICAL_DIR}/logs/hierarchical_training.log"
    params:
        out_dir=HIERARCHICAL_DIR,
        view="FLP"
    shell:
        """
        mkdir -p {params.out_dir}
        mkdir -p {HIERARCHICAL_DIR}/logs
        {PYTHON} -m src.ml.hierarchical_trainer \
            --input {input.master_dataset} \
            --view {params.view} \
            --output-directory {params.out_dir} \
            > {log} 2>&1
        """


# ============================================================
# HIERARCHICAL IDENTIFICATION & INFERENCE
# ============================================================


rule hierarchical_identification:
    input:
        master_dataset=f"{METADATA}/master_dataset.csv",
        checkpoint=f"{HIERARCHICAL_DIR}/best_model.pt"
    output:
        identifications=f"{IDENTIFICATION_DIR}/specimen_identifications.csv",
        errors=f"{IDENTIFICATION_DIR}/identification_errors.csv"
    log:
        f"{HIERARCHICAL_DIR}/logs/hierarchical_identification.log"
    params:
        out_dir=IDENTIFICATION_DIR
    shell:
        """
        mkdir -p {params.out_dir}
        {PYTHON} -m src.ml.hierarchical_identifier \
            --dataset {input.master_dataset} \
            --checkpoint {input.checkpoint} \
            --output-directory {params.out_dir} \
            > {log} 2>&1
        """

# ============================================================
# HIERARCHICAL EVALUATION
# ============================================================

rule evaluate_hierarchical_model:
    input:
        identifications=f"{IDENTIFICATION_DIR}/specimen_identifications.csv"
    output:
        review_priority="results/production/hierarchical/evaluation/review_priority.csv"
    log:
        "results/production/hierarchical/logs/hierarchical_evaluation.log"
    params:
        out_dir="results/production/hierarchical/evaluation"
    shell:
        """
        mkdir -p {params.out_dir}
        MPLBACKEND=Agg PYTHONPATH=. {PYTHON} -m src.ml.hierarchical_evaluator \
            --identifications {input.identifications} \
            --output-directory {params.out_dir} \
            > {log} 2>&1
        """
        
        
# ============================================================
# SPECIMEN PREDICTION SUMMARY TABLE
# ============================================================

rule create_specimen_prediction_table:
    input:
        review_priority=f"{HIERARCHICAL_EVAL_DIR}/review_priority.csv"
    output:
        summary_table="results/tables/specimen_prediction_table.csv"
    log:
        "results/tables/logs/create_specimen_prediction_table.log"
    shell:
        """
        mkdir -p results/tables/logs
        {PYTHON} -m src.core.create_family_genus_summary > {log} 2>&1
        """
       
