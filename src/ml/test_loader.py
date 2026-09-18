from src.ml.dataloader import create_dataloader


def main():
    train_loader = create_dataloader(
        csv_file="metadata/train.csv",
        batch_size=8,
        train=True,
        shuffle=True,
    )

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels:", labels)


if __name__ == "__main__":
    main()
