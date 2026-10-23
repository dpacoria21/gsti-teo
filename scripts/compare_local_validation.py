"""Visualize both fixed Chullo validation areas with identical scale/thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.model import MODEL_PATH, load_model  # noqa: E402
from src.training.dataset import BuildingDataset, read_manifest  # noqa: E402
from src.training.runner import TrainingConfig, configure_image_scale  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/chullo_local_v1/manifest.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "completed":
        raise ValueError("Se necesita una corrida completada y seleccionada por validación.")
    config = TrainingConfig(**run["config"])
    dataset = BuildingDataset(read_manifest(args.manifest), "val")
    model, device = load_model(MODEL_PATH, "auto")
    predictions = []
    for checkpoint in (None, args.run_dir / "best_model.pth"):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        rows = []
        with torch.inference_mode():
            for image, target in dataset:
                configure_image_scale(model, image, config)
                output = model([image.to(device)])[0]
                keep = (output["scores"] >= config.score_threshold) & (output["labels"] == 1)
                masks = output["masks"][keep, 0].cpu().numpy() >= config.mask_threshold
                masks &= target["valid_mask"].numpy()[None]
                rows.append(masks[masks.any(axis=(1, 2))])
        predictions.append(rows)
    figure, axes = plt.subplots(len(dataset), 3, figsize=(14, 6 * len(dataset)), dpi=160, squeeze=False)
    titles = ("Referencias visuales provisionales", "Modelo original USA", "Ajuste local seleccionado")
    for row in range(len(dataset)):
        image, target = dataset[row]
        rgb = image.numpy().transpose(1, 2, 0)
        groups = (target["masks"].numpy(), predictions[0][row], predictions[1][row])
        for column, masks in enumerate(groups):
            axis = axes[row, column]
            axis.imshow(rgb)
            for index, mask in enumerate(masks):
                if mask.any() and not mask.all():
                    axis.contour(mask, levels=[0.5], colors=[plt.cm.tab20(index % 20)], linewidths=0.7)
            invalid = ~target["valid_mask"].numpy()
            axis.imshow(invalid, vmin=0, vmax=1, cmap=ListedColormap([(0, 0, 0, 0), (0.65, 0.2, 0.9, 0.3)]))
            axis.set_title(f"{titles[column]}\n{dataset.scenes[row]['scene_id']} · {len(masks)} contornos", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Chullo · comparación en las dos áreas fijas de validación", fontsize=16)
    figure.text(0.5, 0.035,
                "Cada color delimita una instancia. Morado: región ambigua excluida. Referencias sin revisión humana.\n"
                "Estas áreas se usaron para seleccionar la época; no son la prueba reservada. Escala 1.5625; umbral 0.50.\n"
                "Google Earth · Image © 2026 Airbus · captura 2026-03-04 · georreferenciación aproximada.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.105, top=0.91, wspace=0.12, hspace=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
