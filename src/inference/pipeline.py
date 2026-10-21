"""Orchestrate validation, pretrained inference and traceable result artifacts."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from pyproj import Transformer

from .config import InferenceParameters
from .model import MODEL_ID, MODEL_REVISION, MODEL_SHA256, MODEL_URL, ensure_model, load_model, sha256_file
from .postprocess import SOURCE_ID, candidates_from_masks, detection_frame, suppress_duplicates
from .provenance import inspect_image
from .raster import read_tile, save_preview, tile_starts


def detect_buildings(
    image_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    image_manifest: str | Path | None = None,
    device: str = "auto",
    tile_size: int = 512,
    overlap: int = 128,
    mask_threshold: float = 0.5,
    min_area_m2: float = 10.0,
    analysis_crs: str = "EPSG:32719",
    progress: Callable | None = None,
) -> dict:
    parameters = InferenceParameters(
        confidence_threshold=threshold,
        mask_threshold=mask_threshold,
        tile_size=tile_size,
        overlap=overlap,
        min_area_m2=min_area_m2,
        analysis_crs=analysis_crs,
        device=device,
    )
    import torch
    import torchvision

    def report(**event):
        if progress:
            progress(**event)

    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    image_path, output_dir = Path(image_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report(stage="validating", message="Verificando GeoTIFF, coordenadas y resolución.", progress=0.01)
    image = inspect_image(image_path, image_manifest)
    image_id = "img_" + image["sha256"][:16]
    report(stage="loading_model", message="Cargando Mask R-CNN preentrenado y verificando SHA256.", progress=0.05)
    weights = ensure_model(progress=report)
    model, selected_device = load_model(weights, device)
    candidates = []
    warnings = []
    total = 0
    with rasterio.open(image_path) as src:
        rows = tile_starts(src.height, tile_size, overlap)
        columns = tile_starts(src.width, tile_size, overlap)
        total = len(rows) * len(columns)
        if total > 256:
            raise ValueError("La imagen requiere más de 256 mosaicos; recórtela antes de analizar.")
        metric_transformer = Transformer.from_crs(src.crs, parameters.metric_crs, always_xy=True)
        tile_number = 0
        for row in rows:
            for column in columns:
                tile_number += 1
                tile = read_tile(src, column, row, tile_size)
                if not tile.valid.any():
                    report(
                        stage="inference",
                        message=f"Mosaico {tile_number}/{total} sin datos.",
                        progress=0.1 + 0.75 * tile_number / total,
                        tile=tile_number,
                        total=total,
                    )
                    continue
                tensor = torch.from_numpy(tile.pixels).to(selected_device)
                try:
                    with torch.inference_mode():
                        prediction = model([tensor])[0]
                except RuntimeError as error:
                    if device != "auto" or selected_device != "cuda" or "cuda" not in str(error).lower():
                        raise
                    warnings.append("CUDA falló durante inferencia; se continuó usando CPU.")
                    selected_device = "cpu"
                    model = model.to("cpu")
                    torch.cuda.empty_cache()
                    with torch.inference_mode():
                        prediction = model([tensor.to("cpu")])[0]
                scores = prediction["scores"].detach().cpu().numpy()
                labels = prediction["labels"].detach().cpu().numpy()
                chosen = np.flatnonzero((scores >= threshold) & (labels == 1))
                height, width = tile.valid.shape
                masks = prediction["masks"][chosen, 0, :height, :width].detach().cpu().numpy()
                candidates.extend(
                    candidates_from_masks(
                        scores[chosen],
                        masks,
                        tile,
                        tile_number,
                        metric_transformer,
                        mask_threshold,
                        min_area_m2,
                    )
                )
                report(
                    stage="inference",
                    message=f"Mosaico {tile_number}/{total}: {len(chosen)} detecciones antes de depurar solapamientos.",
                    progress=0.1 + 0.75 * tile_number / total,
                    tile=tile_number,
                    total=total,
                    detections_raw=len(candidates),
                    device=selected_device,
                )
                del prediction, tensor
        native_crs = src.crs
    report(
        stage="vectorizing", message="Eliminando duplicados de mosaicos y georreferenciando contornos.", progress=0.88
    )
    kept = suppress_duplicates(candidates)
    gdf = detection_frame(kept, image_id, image["observed_at"], native_crs)
    output_geojson = output_dir / "buildings.geojson"
    output_geojson.write_text(gdf.to_json(drop_id=True, na="null"), encoding="utf-8")
    preview_bounds = save_preview(image_path, output_dir / "preview.png")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "count": len(gdf),
        "model_id": MODEL_ID,
        "image_id": image_id,
        "source_id": SOURCE_ID,
        "bounds_wgs84": image["bounds_wgs84"],
        "preview_bounds_wgs84": preview_bounds,
        "image": image,
        "model": {
            "id": MODEL_ID,
            "architecture": "Mask R-CNN ResNet50-FPN",
            "pretrained": True,
            "trained_locally": False,
            "path": str(weights),
            "revision": MODEL_REVISION,
            "sha256": MODEL_SHA256,
            "url": MODEL_URL,
            "license": "MIT (model repository)",
            "num_classes": 2,
            "min_size": 800,
            "max_size": 1333,
            "normalization": {"input": "RGB uint8 / 255", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "geography": "Publicado como modelo USA; no validado cuantitativamente en Arequipa.",
        },
        "parameters": parameters.as_manifest(),
        "execution": {
            "device": selected_device,
            "device_requested": device,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "tiles": total,
            "raw_candidates": len(candidates),
            "duplicates_removed": len(candidates) - len(kept),
        },
        "outputs": {
            "buildings": "buildings.geojson",
            "preview": "preview.png",
            "buildings_sha256": sha256_file(output_geojson),
        },
        "warnings": warnings,
        "limitations": [
            "Detecciones automáticas sin revisión; pueden existir omisiones y falsos positivos.",
            "La confianza del modelo no es una probabilidad calibrada ni una medida de peligro.",
            "Se detectan edificaciones, no uso residencial ni ocupación.",
            "Los contornos corresponden a máscaras y pueden representar cubiertas; no sustituyen catastro ni levantamiento de campo.",
            "Fecha de captura desconocida salvo declaración explícita de la fuente.",
            "Las instancias que llegan al borde de la imagen pueden quedar incompletas.",
        ],
    }
    temporary = output_dir / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    report(
        stage="completed",
        message=f"Inferencia finalizada: {len(gdf)} contornos detectados.",
        progress=1.0,
        count=len(gdf),
        elapsed_seconds=manifest["elapsed_seconds"],
    )
    return manifest
