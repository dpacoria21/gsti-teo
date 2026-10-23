"""Reference-label scene datasets with explicit, disjoint train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from torch.utils.data import Dataset


def read_manifest(path: str | Path) -> dict:
    """Resolve paths and reject repeated scenes or prediction-derived supervision."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes", [])
    if not scenes or manifest.get("annotation_type") != "reference":
        raise ValueError("El manifiesto debe incluir scenes y annotation_type='reference'.")
    seen = {key: set() for key in ("scene_id", "image", "labels")}
    for scene in scenes:
        if scene.get("split") not in {"train", "val", "test"}:
            raise ValueError("Cada escena necesita split train, val o test explícito.")
        if scene.get("annotation_type", "reference") != "reference" or not scene.get("labels_source"):
            raise ValueError("Se requieren etiquetas de referencia y labels_source; no predicciones del modelo.")
        for key in ("scene_id", "image", "labels"):
            value = scene.get(key)
            if not value:
                raise ValueError(f"Falta {key} en una escena.")
            if key != "scene_id":
                value = str((path.parent / value).resolve())
                if not Path(value).is_file():
                    raise ValueError(f"No existe el archivo {key}: {value}")
                scene[key] = value
            if value in seen[key]:
                raise ValueError(f"Escena/archivo repetido ({key}); no se permite mezclar splits.")
            seen[key].add(value)
    if {scene["split"] for scene in scenes} != {"train", "val", "test"}:
        raise ValueError("El manifiesto debe incluir train, val y test separados.")
    return manifest


def reference_masks(geometries, shape: tuple[int, int], transform, valid=None, min_pixels: int = 4):
    """One mask per labelled feature, preserving polygon holes and MultiPolygons."""
    instances = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
            raise ValueError("Las etiquetas deben ser Polygon/MultiPolygon válidos; revisar anotaciones.")
        mask = rasterize([(geometry, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
        if valid is not None:
            mask[~valid] = 0
        if int(mask.sum()) >= min_pixels:
            instances.append(mask)
    return np.stack(instances) if instances else np.zeros((0, *shape), dtype="uint8")


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Exclusive upper pixel bounds avoid degenerate boxes for narrow footprints."""
    boxes = []
    for mask in masks:
        rows, columns = torch.where(mask > 0)
        if not len(rows):
            raise ValueError("Una máscara de referencia no puede estar vacía.")
        boxes.append(torch.stack((columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)))
    return torch.stack(boxes).float() if boxes else torch.zeros((0, 4), dtype=torch.float32)


def read_rgb(source) -> tuple[np.ndarray, np.ndarray]:
    """Keep prepared uint8 RGB unchanged; stretch raw uint16 using valid pixels only."""
    if source.crs is None or source.count < 3:
        raise ValueError("La imagen debe tener CRS y tres bandas RGB.")
    rgb = source.read((1, 2, 3))
    valid = source.dataset_mask() > 0
    if not valid.any():
        raise ValueError("La imagen no contiene píxeles válidos.")
    if rgb.dtype == np.uint8:
        pixels = rgb.astype("float32") / 255
    elif rgb.dtype == np.uint16:
        pixels = np.zeros(rgb.shape, dtype="float32")
        for channel in range(3):
            low, high = np.percentile(rgb[channel][valid], [2, 98])
            if high > low:
                pixels[channel] = np.clip((rgb[channel].astype("float32") - low) / (high - low), 0, 1)
    else:
        raise ValueError("Solo se aceptan RGB uint8 o uint16 con preprocesamiento explícito.")
    pixels[:, ~valid] = 0
    return pixels, valid


class BuildingDataset(Dataset):
    """Whole scenes, without crops; augmentation changes only training scenes."""

    def __init__(self, manifest: dict, split: str, augment: bool = False):
        self.scenes = [scene for scene in manifest["scenes"] if scene["split"] == split]
        self.augment = augment
        if split != "train" and augment:
            raise ValueError("Las escenas val/test no deben aumentarse.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        with rasterio.open(scene["image"]) as source:
            pixels, valid = read_rgb(source)
            labels = gpd.read_file(scene["labels"])
            if labels.crs is None:
                raise ValueError("Las etiquetas necesitan un CRS explícito.")
            labels = labels.to_crs(source.crs)
            masks = reference_masks(labels.geometry, (source.height, source.width), source.transform, valid)
        image = torch.from_numpy(pixels)
        masks = torch.from_numpy(masks)
        valid_mask = torch.from_numpy(valid.copy())
        if self.augment:
            rotations = int(torch.randint(4, (1,)).item())
            image, masks = torch.rot90(image, rotations, (1, 2)), torch.rot90(masks, rotations, (1, 2))
            valid_mask = torch.rot90(valid_mask, rotations, (0, 1))
            for axis in (1, 2):
                if torch.rand(1).item() < 0.5:
                    image, masks = image.flip(axis), masks.flip(axis)
                    valid_mask = valid_mask.flip(axis - 1)
        boxes = masks_to_boxes(masks)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(masks), dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": masks.flatten(1).sum(1).float(),
            "iscrowd": torch.zeros(len(masks), dtype=torch.int64),
            "valid_mask": valid_mask,
        }
        return image.contiguous(), target


def collate_scenes(batch):
    return tuple(zip(*batch))
