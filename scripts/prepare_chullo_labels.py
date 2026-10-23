"""Prepare provisional assistant-drawn Chullo references, never detector outputs.

The only selection uses annotation completeness and geometry. No model, score,
prediction or test performance is read. Run from the repository root with the ML
environment; the original raster is required and its recorded hash is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import affinity
from shapely.geometry import GeometryCollection, Polygon, box, mapping, shape
from shapely.ops import unary_union

METHOD = "assistant_visual_manual"
ATTRIBUTION = "Google Earth Pro / Airbus imagery; assistant visual provisional roof annotations."
LABEL_SOURCE = "Manual visual delineation by AI assistant; no detector predictions; not human reviewed."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pixel_geometry(record: dict, size: int = 192):
    """Accept polygon points/holes or pixel-space GeoJSON, without repairing it."""
    geometry = shape(record["geometry"]) if "geometry" in record else Polygon(record["points"], record.get("holes"))
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid polygon: correct its source annotation; no automatic repair is allowed.")
    if not box(0, 0, size, size).covers(geometry):
        raise ValueError(f"Annotation outside local pixel coordinates 0..{size}.")
    return geometry


def validate_annotations(document: dict, scene_id: str, size: int = 192) -> tuple[list, list]:
    if document.get("scene_id") != scene_id or document.get("annotation_method") != METHOD:
        raise ValueError(f"{scene_id}: wrong scene or annotation method; prediction labels are forbidden.")
    if document.get("human_reviewed") is not False:
        raise ValueError(f"{scene_id}: provisional annotations must declare human_reviewed=false.")
    records = document.get("polygons", [])
    identifiers = [record.get("id") for record in records]
    if not records or not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{scene_id}: reference IDs must be present and unique.")
    references = [(pixel_geometry(record, size), record) for record in records]
    ignores = [(pixel_geometry(record, size), record) for record in document.get("ignore_polygons", [])]
    overlaps = []
    for index, (first, record) in enumerate(references):
        for second, other in references[index + 1 :]:
            area = first.intersection(second).area
            if area > 1 + 1e-9:
                overlaps.append(f"{record['id']}/{other['id']}={area:.3f}px2")
    if overlaps:
        raise ValueError(f"{scene_id}: roof overlaps >1px2: {', '.join(overlaps)}")
    return references, ignores


def projected_geometry(geometry, transform: Affine):
    return affinity.affine_transform(
        geometry, [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
    )


def validate_split_regions(parents: list[dict], transform: Affine) -> dict:
    """Reject cross-split leakage before augmentation or crop generation."""
    identifiers = [row["scene_id"] for row in parents]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate parent scene ID.")
    if {row["split"] for row in parents} != {"train", "val", "test"}:
        raise ValueError("Explicit train, val and test parents are required.")
    geometries = []
    for row in parents:
        x, y, width, height = row["window"]
        geometries.append(projected_geometry(box(x, y, x + width, y + height), transform))
    gaps = {}
    for index, first in enumerate(parents):
        for other_index in range(index + 1, len(parents)):
            second = parents[other_index]
            if first["split"] == second["split"]:
                continue
            distance = geometries[index].distance(geometries[other_index])
            if distance <= 0:
                raise ValueError(f"Cross-split overlap/touch: {first['scene_id']} / {second['scene_id']}")
            key = "_".join(sorted((first["split"], second["split"]), key=("train", "val", "test").index))
            gaps[key] = min(gaps.get(key, float("inf")), distance)
    return {"minimum_parent_gap_m": gaps, "cross_split_overlap_count": 0}


def polygon_parts(geometry):
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    parts = [part for part in getattr(geometry, "geoms", []) if part.geom_type in {"Polygon", "MultiPolygon"}]
    return unary_union(parts) if parts else GeometryCollection()


def clipped_references(references: list, region) -> list:
    clipped = []
    for geometry, record in references:
        intersection = polygon_parts(geometry.intersection(region))
        if not intersection.is_empty and intersection.area > 0:
            clipped.append((intersection, record))
    return clipped


def usable_reference_count(references: list, x: int, y: int, size: int, valid=None) -> int:
    transform = Affine.translation(x, y)
    count = 0
    for geometry, _ in references:
        mask = rasterize([(geometry, 1)], out_shape=(size, size), transform=transform, dtype="uint8")
        if valid is not None:
            mask[~valid] = 0
        count += int(mask.sum() >= 4)
    return count


def safe_training_chips(
    references: list,
    ignores: list,
    valid: np.ndarray,
    *,
    minimum_references: int = 8,
    max_chips: int = 4,
    ignore_margin: float = 1,
) -> list[dict]:
    """Select complete train areas; never use partially labelled ignore pixels.

    Greedy new-area coverage is deterministic. Overlap is permitted only between
    descendants of the same training parent. The 25% increment prevents copies.
    """
    if valid.shape != (192, 192):
        raise ValueError("Local parent crops must be 192x192 pixels.")
    excluded = unary_union([geometry for geometry, _ in ignores]).buffer(ignore_margin)
    candidates = []
    for size in [192, 128, 96] if not ignores else [128, 96]:
        last = 192 - size
        coordinates = sorted(set(range(0, last + 1, 32)) | set(range(0, last + 1, 48)) | {last})
        for y in coordinates:
            for x in coordinates:
                region = box(x, y, x + size, y + size)
                if (not excluded.is_empty and region.intersects(excluded)) or not valid[
                    y : y + size, x : x + size
                ].all():
                    continue
                count = int(usable_reference_count(clipped_references(references, region), x, y, size))
                if count >= minimum_references:
                    candidates.append({"x": x, "y": y, "size": size, "reference_count": count, "region": region})
    selected, covered = [], GeometryCollection()
    while candidates and len(selected) < max_chips:
        eligible = []
        for candidate in candidates:
            added = candidate["region"].difference(covered).area
            if added >= 0.25 * candidate["size"] ** 2:
                eligible.append((added, candidate))
        if not eligible:
            break
        added, candidate = max(
            eligible,
            key=lambda pair: (pair[0], pair[1]["reference_count"], pair[1]["size"], -pair[1]["y"], -pair[1]["x"]),
        )
        candidate["new_area_pixels"] = added
        selected.append(candidate)
        covered = covered.union(candidate["region"])
        candidates.remove(candidate)
    if not selected:
        raise ValueError("No safe train chip has >=8 usable references outside ignored areas plus margin.")
    return selected


def feature_collection(references: list, transform: Affine, crs, parent_id: str, *, ignored=False, region=None) -> dict:
    features = []
    for index, (geometry, record) in enumerate(references):
        properties = {key: value for key, value in record.items() if key not in {"points", "holes", "geometry"}}
        properties.update(
            {
                "annotation_type": "ignore" if ignored else "reference",
                "annotation_method": METHOD,
                "human_reviewed": False,
                "parent_scene_id": parent_id,
            }
        )
        if region is not None:
            properties["clipped_to_chip"] = True
        features.append(
            {
                "type": "Feature",
                "id": record.get("id", f"ignore_{index + 1:03d}"),
                "properties": properties,
                "geometry": mapping(projected_geometry(geometry, transform)),
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs.to_string()}},
        "features": features,
    }


def write_rgb(path: Path, source, window: Window) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(window.height),
        "width": int(window.width),
        "count": 3,
        "dtype": source.dtypes[0],
        "crs": source.crs,
        "transform": source.window_transform(window),
        "compress": "deflate",
        "photometric": "RGB",
    }
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path, "w", **profile) as output:
        output.write(source.read((1, 2, 3), window=window))
        output.write_mask(source.dataset_mask(window=window))


def overlay(path: Path, image_path: Path, references: list, ignores: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon as Patch

    with rasterio.open(image_path) as source:
        rgb = source.read((1, 2, 3)).transpose(1, 2, 0)
    figure, axes = plt.subplots(figsize=(8, 8), dpi=140)
    axes.imshow(rgb, extent=(0, 192, 192, 0))
    if ignores:
        ignored = rasterize(
            [(geometry, 1) for geometry, _ in ignores], out_shape=(192, 192), transform=Affine.identity()
        )
        axes.imshow(
            ignored,
            extent=(0, 192, 192, 0),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            cmap=ListedColormap([(0, 0, 0, 0), (0.79, 0.38, 1, 0.3)]),
        )
    for items, color in ((references, "#00ffbc"), (ignores, "#c962ff")):
        for geometry, _ in items:
            for polygon in [geometry] if geometry.geom_type == "Polygon" else geometry.geoms:
                axes.add_patch(
                    Patch(
                        polygon.exterior.coords,
                        fill=False,
                        edgecolor=color,
                        facecolor=color,
                        alpha=1,
                        linewidth=0.8,
                    )
                )
                for interior in polygon.interiors:
                    axes.plot(*interior.xy, color="#ffff00", linewidth=0.7)
    axes.set_title(image_path.stem + " | Assistant provisional; not human reviewed")
    axes.set_xlim(0, 192)
    axes.set_ylim(192, 0)
    figure.text(0.5, 0.015, "Google Earth · Image © 2026 Airbus · captura 2026-03-04", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def prepare(selection_path: Path, repo_root: Path, *, render_overlays: bool = True) -> Path:
    selection_path, repo_root = selection_path.resolve(), repo_root.resolve()
    directory = selection_path.parent
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_path = (repo_root / selection["parent_image"]).resolve()
    if sha256(source_path) != selection["source_sha256"]:
        raise ValueError("Original image SHA256 differs from selection.json; do not silently change the source.")
    parents = selection["scenes"]
    if Counter(row["split"] for row in parents) != {"train": 4, "val": 2, "test": 2}:
        raise ValueError("This protocol requires 4 train, 2 validation and 2 test parents.")
    prepared = []
    # Validate every source before writing any derived artifact.
    for row in parents:
        if row.get("excluded_from_training"):
            if row["split"] != "train" or not row.get("exclusion_reason"):
                raise ValueError("An explicit exclusion is only allowed for train and needs an exclusion_reason.")
        label_path = directory / (row["scene_id"] + ".pixels.json")
        document = json.loads(label_path.read_text(encoding="utf-8"))
        references, ignores = validate_annotations(document, row["scene_id"])
        prepared.append((row, document, references, ignores, label_path))
    scenes, parent_inventory = [], []
    unique_train_ids, unique_usable_train_ids = set(), set()
    with rasterio.open(source_path) as source:
        if source.count < 3 or source.crs is None or not source.crs.is_projected:
            raise ValueError("Original must have three RGB bands and a projected CRS in metres.")
        if source.crs.linear_units != "metre":
            raise ValueError("Projected CRS must use metres for split gap reporting.")
        geography = validate_split_regions(parents, source.transform)
        ready = []
        for row, document, references, ignores, label_path in prepared:
            x, y, width, height = row["window"]
            if (
                any(int(value) != value for value in row["window"])
                or width != 192
                or height != 192
                or x < 0
                or y < 0
                or x + width > source.width
                or y + height > source.height
            ):
                raise ValueError(f"Invalid 192x192 parent window: {row['scene_id']}")
            parent_window = Window(x, y, width, height)
            parent_transform = source.window_transform(parent_window)
            valid = source.dataset_mask(window=parent_window) > 0
            if row.get("excluded_from_training"):
                chips = []
            elif row["split"] == "train":
                chips = safe_training_chips(references, ignores, valid)
            else:
                chips = [{"x": 0, "y": 0, "size": 192, "region": box(0, 0, 192, 192)}]
            ready.append(
                (row, document, references, ignores, label_path, parent_window, parent_transform, valid, chips)
            )
        # Do not leave partially prepared data when a later parent has no safe chip.
        for row, document, references, ignores, label_path, parent_window, parent_transform, valid, chips in ready:
            x, y = int(parent_window.col_off), int(parent_window.row_off)
            parent_image = directory / row["image"]
            write_rgb(parent_image, source, parent_window)
            write_json(
                directory / row["labels"], feature_collection(references, parent_transform, source.crs, row["scene_id"])
            )
            if ignores:
                write_json(
                    directory / (row["scene_id"] + ".ignore.geojson"),
                    feature_collection(ignores, parent_transform, source.crs, row["scene_id"], ignored=True),
                )
            if render_overlays:
                overlay(
                    repo_root / "outputs/chullo_annotation" / (row["scene_id"] + "_prepared.png"),
                    parent_image,
                    references,
                    ignores,
                )
            parent_inventory.append(
                {
                    **row,
                    "annotation_file": label_path.name,
                    "annotation_sha256": sha256(label_path),
                    "image_sha256": sha256(parent_image),
                    "labels_sha256": sha256(directory / row["labels"]),
                    "reference_count": len(references),
                    "ignore_count": len(ignores),
                    "review_notes": document.get("review_notes", []),
                    "derived_scene_count": len(chips),
                }
            )
            for chip in chips:
                region = chip["region"]
                is_train = row["split"] == "train"
                scene_id = (
                    f"{row['scene_id']}_chip_{chip['x']}_{chip['y']}_{chip['size']}" if is_train else row["scene_id"]
                )
                image_path, labels_path = directory / (scene_id + ".tif"), directory / (scene_id + ".geojson")
                labels = clipped_references(references, region) if is_train else references
                window = Window(x + chip["x"], y + chip["y"], chip["size"], chip["size"])
                if is_train:
                    write_rgb(image_path, source, window)
                    write_json(
                        labels_path,
                        feature_collection(labels, parent_transform, source.crs, row["scene_id"], region=region),
                    )
                scene = {
                    "scene_id": scene_id,
                    "parent_scene_id": row["scene_id"],
                    "split": row["split"],
                    "image": image_path.name,
                    "labels": labels_path.name,
                    "labels_source": LABEL_SOURCE,
                    "annotation_type": "reference",
                    "annotation_method": METHOD,
                    "human_reviewed": False,
                    "window_in_parent": [chip["x"], chip["y"], chip["size"], chip["size"]],
                    "window_in_original": [
                        int(window.col_off),
                        int(window.row_off),
                        int(window.width),
                        int(window.height),
                    ],
                    "crs": source.crs.to_string(),
                    "bounds": list(rasterio.windows.bounds(window, source.transform)),
                    "image_sha256": sha256(image_path),
                    "labels_sha256": sha256(labels_path),
                    "reference_count": len(labels),
                    "usable_reference_count_before_ignore": int(
                        usable_reference_count(labels, chip["x"], chip["y"], chip["size"])
                    ),
                }
                if is_train:
                    scene["new_area_pixels"] = chip["new_area_pixels"]
                    for geometry, record in labels:
                        qualified_id = f"{row['scene_id']}:{record['id']}"
                        unique_train_ids.add(qualified_id)
                        if usable_reference_count([(geometry, record)], chip["x"], chip["y"], chip["size"]):
                            unique_usable_train_ids.add(qualified_id)
                elif ignores:
                    ignore_path = directory / (row["scene_id"] + ".ignore.geojson")
                    scene.update({"ignore_regions": ignore_path.name, "ignore_sha256": sha256(ignore_path)})
                evaluated = valid[chip["y"] : chip["y"] + chip["size"], chip["x"] : chip["x"] + chip["size"]].copy()
                if ignores and not is_train:
                    ignored = rasterize(
                        [(geometry, 1) for geometry, _ in ignores],
                        out_shape=evaluated.shape,
                        transform=Affine.translation(chip["x"], chip["y"]),
                    ).astype(bool)
                    evaluated &= ~ignored
                scene["evaluable_pixel_fraction"] = float(evaluated.mean())
                scene["usable_reference_count"] = usable_reference_count(
                    labels, chip["x"], chip["y"], chip["size"], evaluated
                )
                scenes.append(scene)
    manifest = {
        "schema_version": 1,
        "dataset_id": "chullo_local_assistant_provisional_v1",
        "annotation_type": "reference",
        "annotation_method": METHOD,
        "human_reviewed": False,
        "annotation_provenance": {
            "method": METHOD,
            "human_reviewed": False,
            "source_capture_date": "2026-03-04",
            "annotation_unit": "visually distinguishable roof, not dwelling, occupancy or cadastral parcel",
        },
        "assessment_caveat": "Evaluación piloto de una sola captura de Chullo con referencias visuales provisionales del asistente, sin revisión humana. La separación espacial no establece independencia de adquisición ni estadística. Los resultados no representan exactitud en toda Chullo ni Arequipa.",
        "source": selection["parent_image"],
        "source_sha256": selection["source_sha256"],
        "selection_sha256": sha256(selection_path),
        "attribution": ATTRIBUTION,
        "image_rights": "Source imagery remains subject to Google Earth and provider terms; no open imagery license asserted.",
        "geographic_validation": geography,
        "parent_split_counts": dict(Counter(row["split"] for row in parents)),
        "used_parent_split_counts": dict(
            Counter(row["split"] for row in parents if not row.get("excluded_from_training"))
        ),
        "excluded_parents": [
            {"scene_id": row["scene_id"], "split": row["split"], "reason": row["exclusion_reason"]}
            for row in parents
            if row.get("excluded_from_training")
        ],
        "split_counts": dict(Counter(row["split"] for row in scenes)),
        "split_reference_counts_with_training_overlap": {
            split: sum(row["reference_count"] for row in scenes if row["split"] == split)
            for split in ("train", "val", "test")
        },
        "unique_train_reference_ids": sorted(unique_train_ids),
        "unique_train_reference_count": len(unique_train_ids),
        "unique_train_usable_reference_ids": sorted(unique_usable_train_ids),
        "unique_train_usable_reference_count": len(unique_usable_train_ids),
        "chip_policy": {
            "sizes": [192, 128, 96],
            "grid_steps": [32, 48],
            "ignore_margin_pixels": 1,
            "minimum_usable_references": 8,
            "max_per_parent": 4,
            "minimum_new_area_fraction": 0.25,
            "same_train_parent_overlap_allowed": True,
            "cross_split_overlap_allowed": False,
        },
        "limitations": [
            "Provisional visual roof references drawn by an AI assistant, not validated by a human or cadastre.",
            "All chips share one original capture; spatial separation is not independent acquisition or human ground truth.",
            "Instance boundaries and roof-versus-patio interpretation remain approximate; no residential-use or occupancy claim.",
            "Ambiguous regions are excluded only from evaluation; train chips contain none of these regions plus a one-pixel margin.",
            "No model predictions or test performance were used for annotation or chip selection; test labels are only converted.",
            "Training chip reference totals may repeat a roof within the training split; original parent areas and explicit exclusions are inventoried separately.",
        ],
        "parents": parent_inventory,
        "scenes": scenes,
    }
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/training/chullo_local_v1/selection.json"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()
    print(prepare(args.selection, args.repo_root, render_overlays=not args.skip_overlays))


if __name__ == "__main__":
    main()
