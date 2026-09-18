"""
Inference Engine
================

Loads trained CNN checkpoints and performs inference on:
- single images
- specimen folders
- optional single-view filtering such as FDT, FFF, FLT, FLP
"""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.ml.models import build_model
from src.utils.config import CONFIG


class InferenceEngine:
    """Inference engine for CNN prediction."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, checkpoint_path: str):
        self.cfg = CONFIG.project()

        self.device = self.get_device()

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model_name = self.checkpoint["model_name"]
        self.classes = self.checkpoint["classes"]

        model_cfg = self.cfg["models"][self.model_name]

        self.model = build_model(
            model_name=self.model_name,
            num_classes=len(self.classes),
            pretrained=False,
            freeze_backbone=bool(model_cfg["freeze_backbone"]),
        )

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        image_size = int(self.cfg["preprocessing"]["image_size"])

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def get_device(self) -> torch.device:
        """Select CPU or GPU from config."""
        device_config = self.cfg["training"].get("device", "auto")

        if device_config == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        return torch.device(device_config)

    def extract_view_code(self, image_path: Path) -> str:
        """
        Extract view code from filename.

        Examples:
        FDT-0001.jpg -> FDT
        FFF-0004.jpg -> FFF
        FLT.jpg      -> FLT
        """
        return image_path.stem.upper().split("-")[0]

    def list_images(
        self,
        folder_path: str | Path,
        view_code: str | None = None,
    ) -> list[Path]:
        """
        List valid images in a folder.

        If view_code is provided, only images matching that view are returned.
        """
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        images = sorted(
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if view_code and view_code.upper() != "ALL":
            view_code = view_code.upper()

            images = [
                image
                for image in images
                if self.extract_view_code(image) == view_code
            ]

        if not images:
            raise FileNotFoundError(
                f"No valid images found in folder {folder}"
                + (f" for view_code={view_code}" if view_code else "")
            )

        return images

    def load_image(self, image_path: str | Path):
        """Load and transform one image."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_image(
        self,
        image_path: str | Path,
        top_k: int = 3,
    ) -> dict:
        """Predict one image."""
        image_tensor = self.load_image(image_path)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(top_k, len(self.classes)),
        )

        return {
            "image_path": str(image_path),
            "predictions": [
                {
                    "label": self.classes[index.item()],
                    "confidence": float(prob.item()),
                }
                for prob, index in zip(top_probs, top_indices)
            ],
            "probabilities": probabilities.cpu(),
        }

    def predict_folder(
        self,
        folder_path: str | Path,
        top_k: int = 3,
        view_code: str | None = None,
    ) -> dict:
        """
        Predict a specimen folder.

        If view_code='FDT', only FDT images are used.
        If view_code='ALL' or None, all valid images are used.
        """
        images = self.list_images(
            folder_path=folder_path,
            view_code=view_code,
        )

        image_results = []
        probability_vectors = []

        for image_path in images:
            result = self.predict_image(
                image_path=image_path,
                top_k=top_k,
            )

            image_results.append(result)
            probability_vectors.append(result["probabilities"])

        mean_probabilities = torch.stack(probability_vectors).mean(dim=0)

        top_probs, top_indices = torch.topk(
            mean_probabilities,
            k=min(top_k, len(self.classes)),
        )

        final_predictions = [
            {
                "label": self.classes[index.item()],
                "confidence": float(prob.item()),
            }
            for prob, index in zip(top_probs, top_indices)
        ]

        return {
            "folder_path": str(folder_path),
            "view_code": view_code,
            "num_images": len(images),
            "image_results": image_results,
            "final_predictions": final_predictions,
        }
