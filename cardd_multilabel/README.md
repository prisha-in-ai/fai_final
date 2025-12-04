# CarDD Multi-Label Classifier (`cardd_multilabel`)

ResNet-50 multi-label classifier for vehicle damage using COCO-format annotations and PyTorch.

## Folder contents
- `03_training_classification.ipynb`: interactive training notebook (can export to `.py`)
- `classification_training.py`: script entry point mirroring the notebook
- `train.py`: training loop, config, evaluation helpers
- `data.py`: COCO data loading and dataloaders
- `model.py`: ResNet-50 head for multi-label logits
- `metrics.py`: F1/report/confusion utilities

## Data layout (COCO)
- Expected relative path: `../data/raw/{train,val}/` each with `images/` and `annotations.json`
- Categories: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
- Example JSON:
```json
{
  "images": [{ "id": 0, "file_name": "image_0.jpg" }],
  "annotations": [{ "image_id": 0, "category_id": 3, "bbox": [x, y, w, h] }],
  "categories": [
    { "id": 1, "name": "dent" }, { "id": 2, "name": "scratch" },
    { "id": 3, "name": "crack" }, { "id": 4, "name": "glass_shatter" },
    { "id": 5, "name": "lamp_broken" }, { "id": 6, "name": "tire_flat" }
  ]
}
```

## Run with the script
1) Check `DataConfig.root_dir` in `classification_training.py` (defaults to `../data/raw`).
2) Train:
```bash
python classification_training.py
```
3) Best checkpoint saves to `../weights/resnet50_multilabel_cardd_best.pt` (set via `TrainConfig.checkpoint_path`).

## Model and training notes
- Architecture: ResNet-50 backbone -> 256-unit head -> logits for 6 labels; sigmoid applied during evaluation.
- Loss: `BCEWithLogitsLoss`; optimizer: `AdamW`; scheduler: `StepLR`.
- Metrics per epoch: train loss, val loss, micro-F1, per-class F1; final classification report and confusion summary printed.

## License
Research and academic use.
