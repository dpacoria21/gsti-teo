"""Acquire a reproducible spatially separated subset of public SpaceNet 2 data."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from shapely.geometry import box

BASE = "https://spacenet-dataset.s3.amazonaws.com/"
PREFIX = "spacenet/SN2_buildings/train/AOI_5_Khartoum/"
DATASET_ID = "spacenet2_khartoum_reference_subset_v1"
METRIC_CRS = "EPSG:32636"
SOURCE = "https://spacenet.ai/spacenet-buildings-dataset-v2/"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"


def _emit(callback, **event):
    if callback:
        callback(event)


def _fetch(url: str, *, headers: dict | None = None) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if headers and "Range" in headers and response.status != 206:
                    raise ValueError("S3 did not honor the metadata-only HTTP range request.")
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Unreachable download state")


def _list_objects(folder: str) -> dict[str, dict]:
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    token, result = None, {}
    while True:
        params = {"list-type": "2", "prefix": PREFIX + folder + "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(_fetch(BASE + "?" + urllib.parse.urlencode(params)))
        for item in root.findall("s:Contents", namespace):
            key = item.findtext("s:Key", namespaces=namespace)
            match = re.search(r"_img(\d+)\.(?:tif|geojson)$", key or "")
            if match:
                result[match.group(1)] = {
                    "url": BASE + key,
                    "size_bytes": int(item.findtext("s:Size", namespaces=namespace)),
                    "etag": item.findtext("s:ETag", namespaces=namespace).strip('"'),
                }
        token = root.findtext("s:NextContinuationToken", namespaces=namespace)
        if not token:
            return result


def _header(scene_id: str, item: dict, label: dict) -> dict:
    header = _fetch(item["url"], headers={"Range": "bytes=0-4095"})
    with MemoryFile(header) as memory:
        with memory.open() as source:
            bounds = list(source.bounds)
            if source.crs is None or source.count != 3 or (source.width, source.height) != (650, 650):
                raise ValueError(f"Unexpected SpaceNet RGB metadata: {scene_id}")
            return {
                "scene_id": "khartoum_img" + scene_id,
                "image_id": scene_id,
                "source_image": item,
                "source_labels": label,
                "crs": source.crs.to_string(),
                "bounds": bounds,
                "bbox": bounds,
                "bbox_metric": list(transform_bounds(source.crs, METRIC_CRS, *bounds)),
                "width": source.width,
                "height": source.height,
                "source_dtype": source.dtypes[0],
                "transform": list(source.transform)[:6],
                "nodata": source.nodata,
            }


def spatial_split(scenes: list[dict], counts: dict[str, int], seed: int, gap_m: float) -> tuple[list[dict], dict]:
    """Hold out western/eastern geographic strips, separating full tile bounds."""
    if gap_m < 200 or any(value < 1 for value in counts.values()):
        raise ValueError("Require three nonempty splits and at least 200 m geographic separation.")
    centers = np.array([(row["bbox_metric"][0] + row["bbox_metric"][2]) / 2 for row in scenes])
    candidates = []
    # Search stripe positions using geography/counts only, never model outcomes.
    for left_q in (0.15, 0.20, 0.25, 0.30):
        for right_q in (0.70, 0.75, 0.80, 0.85):
            left, right = np.quantile(centers, [left_q, right_q])
            groups = {"train": [], "val": [], "test": []}
            for row in scenes:
                west, _, east, _ = row["bbox_metric"]
                if east <= left - gap_m / 2:
                    groups["val"].append(row)
                elif west >= right + gap_m / 2:
                    groups["test"].append(row)
                elif west >= left + gap_m / 2 and east <= right - gap_m / 2:
                    groups["train"].append(row)
            if all(len(groups[name]) >= counts[name] for name in groups):
                candidates.append((min(len(groups[name]) / counts[name] for name in groups), left, right, groups))
    if not candidates:
        raise ValueError("The requested split counts do not fit spatial holdouts with the required buffer.")
    _, left, right, groups = max(candidates, key=lambda item: item[0])
    rng = np.random.default_rng(seed)
    selected = []
    for name in ("train", "val", "test"):
        rows = sorted(groups[name], key=lambda row: int(row["image_id"]))
        # Random scene sampling occurs only WITHIN separated geographic regions.
        indices = rng.choice(len(rows), counts[name], replace=False)
        selected.extend({**rows[int(index)], "split": name} for index in sorted(indices))
    validation = validate_geographic_split(selected, gap_m)
    validation.update(
        {
            "strategy": "longitude strips in UTM 36N: validation west, training centre, test east",
            "cut_x_m": [float(left), float(right)],
            "candidate_counts": {k: len(v) for k, v in groups.items()},
            "seed": seed,
            "metric_crs": METRIC_CRS,
        }
    )
    return selected, validation


def validate_geographic_split(scenes: list[dict], gap_m: float) -> dict:
    groups = {
        name: [box(*row["bbox_metric"]) for row in scenes if row["split"] == name] for name in ("train", "val", "test")
    }
    if any(not values for values in groups.values()) or len({s["scene_id"] for s in scenes}) != len(scenes):
        raise ValueError("Every geographic split must be nonempty and contain unique scenes.")
    minimums = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        minimum = min(a.distance(b) for a in groups[first] for b in groups[second])
        if minimum + 1e-6 < gap_m:
            raise ValueError(f"Spatial leakage: {first}/{second} gap {minimum} < {gap_m} m.")
        minimums[f"{first}_{second}"] = minimum
    return {
        "required_gap_m": gap_m,
        "minimum_bbox_gap_m": minimums,
        "cross_split_overlap_count": 0,
        "validation": "passed; complete projected tile bounds separated before any augmentation/cropping",
    }


def _download_checked(item: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = path.read_bytes() if path.is_file() else _fetch(item["url"])
    if len(data) != item["size_bytes"] or hashlib.md5(data).hexdigest() != item["etag"]:
        raise ValueError(f"S3 length/ETag mismatch: {item['url']}")
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def _prepare_scene(row: dict, output: Path) -> dict:
    name = row["scene_id"]
    raw = output / "raw" / "images" / (name + ".tif")
    labels = output / "raw" / "labels" / (name + ".geojson")
    image = output / "rgb" / (name + ".tif")
    raw_sha = _download_checked(row["source_image"], raw)
    labels_sha = _download_checked(row["source_labels"], labels)
    annotations = json.loads(labels.read_text(encoding="utf-8"))
    if annotations.get("type") != "FeatureCollection" or "features" not in annotations:
        raise ValueError(f"Invalid reference annotations: {name}")
    declared_label_crs = annotations.get("crs", {}).get("properties", {}).get("name")
    if declared_label_crs not in (None, "urn:ogc:def:crs:OGC:1.3:CRS84", "EPSG:4326"):
        raise ValueError(f"Unexpected label CRS: {name}: {declared_label_crs}")
    with rasterio.open(raw) as source:
        pixels = source.read((1, 2, 3))
        valid = (source.dataset_mask() > 0) & np.any(pixels != 0, axis=0)
        if valid.mean() < 0.5:
            return {
                **row,
                "excluded": True,
                "reason": "valid pixel fraction below 0.5",
                "valid_fraction": float(valid.mean()),
                "raw_image_sha256": raw_sha,
                "labels_sha256": labels_sha,
            }
        normalized = np.zeros(pixels.shape, dtype=np.uint8)
        stretches = []
        for band in range(3):
            low, high = np.percentile(pixels[band][valid], (2, 98))
            if high <= low:
                raise ValueError(f"Constant RGB channel: {name}")
            normalized[band] = np.rint(
                np.clip((pixels[band].astype(np.float32) - low) / (high - low), 0, 1) * 255
            ).astype(np.uint8)
            normalized[band][~valid] = 0
            stretches.append([float(low), float(high)])
        profile = source.profile.copy()
        profile.update(dtype="uint8", nodata=None, compress="deflate", photometric="RGB")
        image.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(image, "w", **profile) as destination:
                destination.write(normalized)
                destination.write_mask(valid.astype(np.uint8) * 255)
                destination.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    with rasterio.open(image) as check:
        if list(check.bounds) != row["bounds"] or check.crs.to_string() != row["crs"]:
            raise ValueError("Image preparation changed georeferencing.")
    return {
        **row,
        "image": image.relative_to(output).as_posix(),
        "labels": labels.relative_to(output).as_posix(),
        "raw_image": raw.relative_to(output).as_posix(),
        "raw_image_sha256": raw_sha,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "labels_sha256": labels_sha,
        "labels_source": "SpaceNet 2 Khartoum reference building annotations",
        "annotation_type": "reference",
        "label_count": len(annotations["features"]),
        "label_crs": "OGC:CRS84",
        "dtype": "uint8",
        "preprocessing": {
            "operation": "independent per-image RGB 2/98 percentile stretch to uint8",
            "channel_percentiles": stretches,
            "valid_fraction": float(valid.mean()),
            "valid_rule": "source dataset mask and at least one nonzero RGB channel",
            "resampling": "none",
            "shape_changed": False,
        },
    }


def prepare_dataset(
    output: Path,
    *,
    train: int = 200,
    val: int = 40,
    test: int = 40,
    seed: int = 20261023,
    gap_m: float = 200,
    workers: int = 8,
    callback=None,
) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_file = output / "source_index.json"
    if index_file.is_file():
        scenes = json.loads(index_file.read_text(encoding="utf-8"))["scenes"]
    else:
        images, labels = _list_objects("PS-RGB"), _list_objects("geojson_buildings")
        if set(images) != set(labels):
            raise ValueError("SpaceNet RGB images and annotation scene IDs do not pair exactly.")
        _emit(callback, phase="metadata", total=len(images), message="Reading TIFF headers only")
        scenes = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_header, key, images[key], labels[key]) for key in sorted(images, key=int)]
            for count, job in enumerate(as_completed(jobs), 1):
                scenes.append(job.result())
                if count % 100 == 0:
                    _emit(callback, phase="metadata", completed=count, total=len(jobs))
        scenes.sort(key=lambda row: int(row["image_id"]))
        index_file.write_text(json.dumps({"source": SOURCE, "scenes": scenes}, indent=2), encoding="utf-8")
    selected, geographic_validation = spatial_split(scenes, {"train": train, "val": val, "test": test}, seed, gap_m)
    size = sum(row["source_image"]["size_bytes"] + row["source_labels"]["size_bytes"] for row in selected)
    if size > 1_000_000_000:
        raise ValueError(f"Selected raw downloads exceed the 1 GB acquisition limit: {size} bytes.")
    selection = {"scenes": selected, "geographic_validation": geographic_validation, "download_bytes": size}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    _emit(
        callback,
        phase="download",
        total=len(selected),
        expected_bytes=size,
        geographic_validation=geographic_validation,
    )
    prepared, rejected = [], []
    used = {row["scene_id"] for row in selected}
    left, right = geographic_validation["cut_x_m"]
    replacements = {"train": [], "val": [], "test": []}
    for row in scenes:
        if row["scene_id"] in used:
            continue
        west, _, east, _ = row["bbox_metric"]
        split = None
        if east <= left - gap_m / 2:
            split = "val"
        elif west >= right + gap_m / 2:
            split = "test"
        elif west >= left + gap_m / 2 and east <= right - gap_m / 2:
            split = "train"
        if split:
            replacements[split].append({**row, "split": split})
    rng = np.random.default_rng(seed + 1)
    for group in replacements.values():
        rng.shuffle(group)
    desired = {"train": train, "val": val, "test": test}
    pending = selected
    while pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_prepare_scene, row, output) for row in pending]
            for count, job in enumerate(as_completed(jobs), 1):
                result = job.result()
                (rejected if result.get("excluded") else prepared).append(result)
                if count % 10 == 0 or count == len(jobs):
                    _emit(
                        callback,
                        phase="download",
                        completed=len(prepared),
                        quality_rejected=len(rejected),
                        total=sum(desired.values()),
                    )
        pending = []
        for split, target_count in desired.items():
            deficit = target_count - sum(row["split"] == split for row in prepared)
            for _ in range(deficit):
                if not replacements[split]:
                    raise ValueError(f"Not enough valid scenes in the fixed {split} region.")
                row = replacements[split].pop()
                size += row["source_image"]["size_bytes"] + row["source_labels"]["size_bytes"]
                if size > 1_000_000_000:
                    raise ValueError(
                        f"Quality replacements would exceed the 1 GB raw download limit; accepted {len(prepared)} scenes."
                    )
                pending.append(row)
        (output / "quality_rejections.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    geographic_validation.update(validate_geographic_split(prepared, gap_m))
    if len({row["image_sha256"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate normalized image content detected in the selected dataset.")
    prepared.sort(key=lambda row: (row["split"], int(row["image_id"])))
    (output / "selection.json").write_text(
        json.dumps(
            {
                "scenes": prepared,
                "geographic_validation": geographic_validation,
                "download_bytes": size,
                "quality_excluded_scene_count": len(rejected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "annotation_type": "reference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "license": "CC-BY-SA-4.0",
        "license_url": LICENSE,
        "attribution": "SpaceNet Dataset by SpaceNet Partners; DigitalGlobe/Maxar WorldView-3 imagery.",
        "citation": "Van Etten, Lindenbaum & Bacastow (2018), SpaceNet: A Remote Sensing Dataset and Challenge Series, arXiv:1807.01232.",
        "imagery_catalog_id": "104001000A6A1E00",
        "capture_date": "2015-04-13T08:18:08Z",
        "split_seed": seed,
        "geographic_validation": geographic_validation,
        "split_counts": {"train": train, "val": val, "test": test},
        "raw_download_bytes": size,
        "selected_raw_bytes": sum(
            row["source_image"]["size_bytes"] + row["source_labels"]["size_bytes"] for row in prepared
        ),
        "quality_excluded_scene_count": len(rejected),
        "quality_rule": "At least 50% pixels valid in source mask and not all-zero RGB; replacement stays inside same fixed split region.",
        "total_reference_buildings": sum(row["label_count"] for row in prepared),
        "limitations": [
            "Khartoum reference data, not Arequipa ground truth.",
            "Unknown full pretraining corpus of the original checkpoint; independence from pretraining is not established.",
            "Improvement on this holdout alone does not establish improvement in Chullo.",
        ],
        "scenes": prepared,
    }
    (output / "LICENSE_AND_ATTRIBUTION.txt").write_text(
        "SpaceNet 2 Khartoum — reference imagery and building labels\nCC BY-SA 4.0\n"
        + LICENSE
        + "\n"
        + SOURCE
        + "\nSpaceNet Dataset by SpaceNet Partners; DigitalGlobe/Maxar.\n"
        "Modifications: selected spatial subset; 2/98 percentile RGB stretch; original raw data preserved.\n"
        "Redistributed adaptations of the dataset must retain attribution and the applicable share-alike terms.\n",
        encoding="utf-8",
    )
    target = output / "manifest.json"
    temporary = output / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)
    _emit(
        callback,
        phase="complete",
        manifest=str(target),
        split_counts=manifest["split_counts"],
        total_reference_buildings=manifest["total_reference_buildings"],
    )
    return target
