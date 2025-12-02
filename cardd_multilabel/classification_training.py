from cardd_multilabel.data import DataConfig, create_dataloaders
from cardd_multilabel.model import build_resnet50_multilabel
from cardd_multilabel.train import TrainConfig, train_model


def main():
    # 1. Data
    data_cfg = DataConfig(
        root_dir="/Users/prisriva/Desktop/fai_final/data/raw",   # adjust if your data is elsewhere
        num_classes=6,
        batch_size=16,
        num_workers=0,
    )

    train_loader, val_loader, class_names = create_dataloaders(data_cfg)

    # 2. Model
    model = build_resnet50_multilabel(num_classes=data_cfg.num_classes)

    # 3. Train
    train_cfg = TrainConfig(
        num_epochs=10,
        lr=1e-4,
        weight_decay=1e-4,
        device="mps",   # or "cuda" / "cpu"
        checkpoint_path="../weights/resnet50_multilabel_cardd_best.pt",
        # (or "weights/..." if you prefer; just be consistent)
    )

    results = train_model(
        model,
        train_loader,
        val_loader,
        class_names,
        train_cfg,
    )

    # optional: print or log `results` here
    print("Best micro-F1:", results["best_val_micro_f1"])


if __name__ == "__main__":
    main()
