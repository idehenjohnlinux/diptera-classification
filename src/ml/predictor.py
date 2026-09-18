"""
Prediction CLI
==============

Supports:
- --image
- --folder
- --folders
- --only-identified
- --all-models
- --all-views
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ml.inference import InferenceEngine
from src.utils.config import CONFIG


SUPPORTED_MODELS = {
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
}


def set_active_view(view_code: str) -> None:
    """Update active view in global config."""
    CONFIG.config["experiment"]["view_code"] = view_code


def get_model_name(model_path: str) -> str:
    """Extract model name from checkpoint path."""
    for part in Path(model_path).parts:
        if part in SUPPORTED_MODELS:
            return part

    return "unknown_model"


def get_experiment_context() -> tuple[str, str, str]:
    """Get mode, classification level, and active view from config."""
    cfg = CONFIG.project()

    mode = cfg["experiment"]["mode"]

    classification_level = cfg["training"].get(
        "classification_level",
        "species",
    )

    view_code = cfg["experiment"].get("view_code", "ALL")

    return mode, classification_level, view_code


def discover_model_paths(fold: int) -> list[str]:
    """Discover checkpoints for all configured CNN models."""
    cfg = CONFIG.project()

    mode, classification_level, view_code = get_experiment_context()

    model_paths = []

    for model_name in cfg["training"]["models"]:
        path = (
            Path("models")
            / mode
            / classification_level
            / view_code
            / model_name
            / f"fold_{fold}"
            / "checkpoints"
            / "best_model.pth"
        )

        if path.exists():
            model_paths.append(str(path))
        else:
            print(f"Warning: missing checkpoint: {path}")

    if not model_paths:
        raise FileNotFoundError(
            "No checkpoints found. Check fold, view_code, "
            "classification_level, and model folders."
        )

    return model_paths


def get_true_label_map() -> dict:
    """Map specimen_id to true label using processed_dataset.csv."""
    csv_path = Path("metadata/processed_dataset.csv")

    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        df["label"] = (
            df["genus"].astype(str).str.strip()
            + "_"
            + df["specific_epithet"].astype(str).str.strip()
        )

    return (
        df[["specimen_id", "label"]]
        .drop_duplicates()
        .set_index("specimen_id")["label"]
        .to_dict()
    )


def get_identified_specimen_folders() -> list[Path]:
    """Return folders for specimens present in training_dataset.csv."""
    csv_path = Path("metadata/training_dataset.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            "metadata/training_dataset.csv not found. Run validator first."
        )

    df = pd.read_csv(csv_path)

    specimen_ids = sorted(df["specimen_id"].astype(str).unique())

    folders = []

    for specimen_id in specimen_ids:
        folder = Path("images") / specimen_id

        if folder.exists() and folder.is_dir():
            folders.append(folder)

    return folders


def clean_result_for_json(result: dict) -> dict:
    """Remove tensors before JSON export."""
    cleaned = dict(result)

    cleaned.pop("probabilities", None)

    if "image_results" in cleaned:
        for image_result in cleaned["image_results"]:
            image_result.pop("probabilities", None)

    return cleaned


def save_prediction_outputs(
    result: dict,
    output_dir: Path,
    model_path: str,
    true_label: str | None = None,
) -> dict:
    """Save prediction.json, prediction.csv, and prediction.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = clean_result_for_json(result)

    with open(output_dir / "prediction.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_path": model_path,
                "true_label": true_label,
                **cleaned,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    rows = []

    if "final_predictions" in result:
        for rank, prediction in enumerate(result["final_predictions"], start=1):
            rows.append(
                {
                    "level": "final_prediction",
                    "image_path": None,
                    "rank": rank,
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                }
            )

        for image_result in result["image_results"]:
            for rank, prediction in enumerate(
                image_result["predictions"],
                start=1,
            ):
                rows.append(
                    {
                        "level": "image_prediction",
                        "image_path": image_result["image_path"],
                        "rank": rank,
                        "label": prediction["label"],
                        "confidence": prediction["confidence"],
                    }
                )

        best = result["final_predictions"][0]

    else:
        for rank, prediction in enumerate(result["predictions"], start=1):
            rows.append(
                {
                    "level": "single_image_prediction",
                    "image_path": result["image_path"],
                    "rank": rank,
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                }
            )

        best = result["predictions"][0]

    pd.DataFrame(rows).to_csv(output_dir / "prediction.csv", index=False)

    with open(output_dir / "prediction.txt", "w", encoding="utf-8") as file:
        file.write("Brachycera CNN Prediction Report\n")
        file.write("=" * 40 + "\n")
        file.write(f"Model: {model_path}\n")

        if true_label:
            file.write(f"True label: {true_label}\n")

        file.write("\n")

        if "final_predictions" in result:
            file.write(f"Specimen folder: {result['folder_path']}\n")
            file.write(f"View used: {result.get('view_code')}\n")
            file.write(f"Images analysed: {result['num_images']}\n\n")
            file.write("Final prediction:\n")

            for rank, prediction in enumerate(result["final_predictions"], start=1):
                file.write(
                    f"{rank}. {prediction['label']} "
                    f"({prediction['confidence'] * 100:.2f}%)\n"
                )

            file.write("\nPer-image predictions:\n")

            for image_result in result["image_results"]:
                best_image = image_result["predictions"][0]
                file.write(
                    f"- {image_result['image_path']} -> "
                    f"{best_image['label']} "
                    f"({best_image['confidence'] * 100:.2f}%)\n"
                )

        else:
            file.write(f"Image: {result['image_path']}\n\n")
            file.write("Prediction:\n")

            for rank, prediction in enumerate(result["predictions"], start=1):
                file.write(
                    f"{rank}. {prediction['label']} "
                    f"({prediction['confidence'] * 100:.2f}%)\n"
                )

    return {
        "prediction": best["label"],
        "confidence": best["confidence"],
        "output_dir": str(output_dir),
    }


def predict_single_image(
    model_path: str,
    image_path: str,
    top_k: int,
) -> dict:
    """Predict one image using one model."""
    mode, classification_level, view_code = get_experiment_context()

    engine = InferenceEngine(model_path)
    result = engine.predict_image(image_path, top_k=top_k)

    model_name = get_model_name(model_path)

    output_dir = (
        Path("predictions")
        / mode
        / classification_level
        / view_code
        / model_name
        / Path(image_path).stem
    )

    summary = save_prediction_outputs(
        result=result,
        output_dir=output_dir,
        model_path=model_path,
    )

    return {
        "specimen_id": Path(image_path).stem,
        "model": model_name,
        "classification_level": classification_level,
        "view_code": view_code,
        "prediction": summary["prediction"],
        "confidence": summary["confidence"],
        "output_dir": summary["output_dir"],
    }


def predict_folder(
    model_path: str,
    folder: Path,
    top_k: int,
    true_label: str | None = None,
) -> dict:
    """Predict one specimen folder using one model and current view."""
    mode, classification_level, view_code = get_experiment_context()

    engine = InferenceEngine(model_path)

    result = engine.predict_folder(
        folder_path=folder,
        top_k=top_k,
        view_code=view_code,
    )

    model_name = get_model_name(model_path)
    specimen_id = folder.name

    output_dir = (
        Path("predictions")
        / mode
        / classification_level
        / view_code
        / model_name
        / specimen_id
    )

    summary = save_prediction_outputs(
        result=result,
        output_dir=output_dir,
        model_path=model_path,
        true_label=true_label,
    )

    correct = None

    if true_label is not None:
        correct = summary["prediction"] == true_label

    return {
        "specimen_id": specimen_id,
        "model": model_name,
        "classification_level": classification_level,
        "view_code": view_code,
        "true_label": true_label,
        "prediction": summary["prediction"],
        "confidence": summary["confidence"],
        "correct": correct,
        "num_images": result["num_images"],
        "output_dir": summary["output_dir"],
    }


def run_model_on_folders(
    model_path: str,
    folders: list[Path],
    top_k: int,
    true_label_map: dict,
) -> list[dict]:
    """Predict many folders using one model."""
    rows = []

    for folder in folders:
        try:
            true_label = true_label_map.get(folder.name)

            row = predict_folder(
                model_path=model_path,
                folder=folder,
                top_k=top_k,
                true_label=true_label,
            )

            rows.append(row)

            print(
                f"{row['model']} | {row['specimen_id']} | "
                f"{row['view_code']} -> {row['prediction']} "
                f"({row['confidence'] * 100:.2f}%)"
            )

        except Exception as error:
            mode, classification_level, view_code = get_experiment_context()

            rows.append(
                {
                    "specimen_id": folder.name,
                    "model": get_model_name(model_path),
                    "classification_level": classification_level,
                    "view_code": view_code,
                    "true_label": true_label_map.get(folder.name),
                    "prediction": None,
                    "confidence": None,
                    "correct": None,
                    "num_images": 0,
                    "output_dir": None,
                    "error": str(error),
                }
            )

    mode, classification_level, view_code = get_experiment_context()
    model_name = get_model_name(model_path)

    output_dir = (
        Path("predictions")
        / mode
        / classification_level
        / view_code
        / model_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output_dir / "batch_predictions.csv",
        index=False,
    )

    return rows


def save_model_comparison(all_rows: list[dict]) -> None:
    """Save combined prediction comparison."""
    mode, classification_level, view_code = get_experiment_context()

    output_dir = (
        Path("predictions")
        / mode
        / classification_level
        / view_code
        / "comparison"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "all_model_predictions.csv", index=False)

    if not df.empty and "model" in df.columns:
        summary = (
            df.groupby("model")
            .agg(
                specimens_predicted=("specimen_id", "count"),
                mean_confidence=("confidence", "mean"),
                accuracy=("correct", "mean"),
            )
            .reset_index()
        )

        summary.to_csv(
            output_dir / "model_prediction_summary.csv",
            index=False,
        )


def get_folders_from_args(args) -> list[Path]:
    """Resolve folders from CLI arguments."""
    if args.only_identified:
        return get_identified_specimen_folders()

    if args.folders:
        parent = Path(args.folders)

        if not parent.exists() or not parent.is_dir():
            raise FileNotFoundError(f"Folder not found: {parent}")

        return sorted(folder for folder in parent.iterdir() if folder.is_dir())

    if args.folder:
        return [Path(args.folder)]

    return []


def run_predictions_for_current_view(args) -> list[dict]:
    """Run predictions for the current active view."""
    if args.all_models:
        model_paths = discover_model_paths(args.fold)
    else:
        if not args.model:
            raise ValueError("Use --model PATH or --all-models")

        model_paths = [args.model]

    true_label_map = get_true_label_map()
    all_rows = []

    if args.image:
        for model_path in model_paths:
            row = predict_single_image(
                model_path=model_path,
                image_path=args.image,
                top_k=args.top_k,
            )

            all_rows.append(row)

            print(
                f"{row['model']} | {row['view_code']} | "
                f"{Path(args.image).name} -> {row['prediction']} "
                f"({row['confidence'] * 100:.2f}%)"
            )

    else:
        folders = get_folders_from_args(args)

        for model_path in model_paths:
            rows = run_model_on_folders(
                model_path=model_path,
                folders=folders,
                top_k=args.top_k,
                true_label_map=true_label_map,
            )

            all_rows.extend(rows)

    save_model_comparison(all_rows)

    return all_rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict Brachycera species from image, folder, folders, "
            "or identified specimens."
        )
    )

    parser.add_argument(
        "--model",
        help="Path to one best_model.pth",
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use all CNN models configured in config.yaml",
    )

    parser.add_argument(
        "--all-views",
        action="store_true",
        help="Run prediction for all views configured in config.yaml",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Cross-validation fold number to use",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to save",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--image",
        help="Path to one image",
    )

    group.add_argument(
        "--folder",
        help="Path to one specimen folder",
    )

    group.add_argument(
        "--folders",
        help="Path to parent folder containing specimen folders",
    )

    group.add_argument(
        "--only-identified",
        action="store_true",
        help="Predict only specimens present in metadata/training_dataset.csv",
    )

    args = parser.parse_args()

    if args.all_views:
        views = CONFIG.project()["experiment"].get("views", [])

        if not views:
            raise ValueError("No experiment.views found in config.yaml.")

    else:
        views = [CONFIG.project()["experiment"].get("view_code", "ALL")]

    all_view_rows = []

    for view_code in views:
        set_active_view(view_code)

        print("\n" + "=" * 80)
        print(f"Running predictions for view: {view_code}")
        print("=" * 80)

        rows = run_predictions_for_current_view(args)
        all_view_rows.extend(rows)

    output_dir = Path("predictions") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_view_rows).to_csv(
        output_dir / "all_views_all_models_predictions.csv",
        index=False,
    )

    print("\nPrediction completed.")
    print("Saved outputs in: predictions/")


if __name__ == "__main__":
    main()
