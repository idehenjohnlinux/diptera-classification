from src.ml.dataloader import create_dataloader
from src.ml.losses import build_loss_function
from src.ml.models import build_model
from src.ml.metrics import count_trainable_parameters, count_total_parameters
from src.utils.config import CONFIG


def main():
    cfg = CONFIG.project()
    train_csv = "metadata/train.csv"

    train_loader = create_dataloader(
        csv_file=train_csv,
        batch_size=cfg["training"]["batch_size"],
        train=True,
        shuffle=True,
    )

    dataset = train_loader.dataset
    num_classes = len(dataset.classes)

    model = build_model(
        model_name="resnet18",
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
    )

    loss_fn = build_loss_function(
        csv_file=train_csv,
        classes=dataset.classes,
        use_class_weights=cfg["training"]["use_class_weights"],
        label_column=cfg["training"]["label_column"],
    )

    print("Classes:", dataset.classes)
    print("Num classes:", num_classes)
    print("Loss function:", loss_fn)
    print("Total parameters:", count_total_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))


if __name__ == "__main__":
    main()
