from pathlib import Path
from src.ml.evaluate import evaluate_checkpoint

training_root = Path("results/training")

checkpoints = sorted(training_root.rglob("best_model.pt"))

print(f"Found {len(checkpoints)} trained models.\n")

successful = 0
failed = 0

for i, checkpoint in enumerate(checkpoints, start=1):

    print("=" * 80)
    print(f"[{i}/{len(checkpoints)}]")
    print(checkpoint)
    print("=" * 80)

    try:
        evaluate_checkpoint(
            checkpoint_path=checkpoint,
            overwrite=False,
        )
        successful += 1

    except Exception as error:
        failed += 1
        print(f"\nERROR: {error}\n")

print("\nFinished")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
