from src.ml.models import build_model


def main():
    for model_name in ["resnet18", "efficientnet_b0", "mobilenet_v3_large"]:
        model = build_model(
            model_name=model_name,
            num_classes=5,
            pretrained=True,
        )

        print(model_name, "loaded successfully")


if __name__ == "__main__":
    main()
