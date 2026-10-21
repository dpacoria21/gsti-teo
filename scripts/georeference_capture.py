"""Georeference a complete RGB capture from auditable fit/check control points.

CSV columns: id,column,row,longitude,latitude,role. column and row are zero-based
pixel-centre indices, as returned by PIL/OpenCV: the first pixel centre is (0,0).
The fitted raster transform uses GDAL's pixel-corner convention internally.
This measures consistency with the supplied coordinates, NOT ground accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_crs(value: str) -> CRS:
    crs = CRS.from_user_input(value)
    if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-10 for axis in crs.axis_info[:2]):
        raise ValueError("El CRS de ajuste debe ser proyectado y usar metros.")
    return crs


def read_points(path: str | Path, width: int, height: int, target_crs: str = "EPSG:32719") -> list[dict]:
    transformer = Transformer.from_crs("EPSG:4326", metric_crs(target_crs), always_xy=True)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"id", "column", "row", "longitude", "latitude", "role"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("CSV: se requieren id,column,row,longitude,latitude,role.")
        points = []
        for index, record in enumerate(reader, 2):
            try:
                point = {"id": record["id"].strip(), "role": record["role"].strip().lower()}
                point.update({name: float(record[name]) for name in ("column", "row", "longitude", "latitude")})
            except (ValueError, AttributeError, TypeError) as error:
                raise ValueError(f"CSV: datos no válidos en la fila {index}.") from error
            if not point["id"] or point["role"] not in {"fit", "check"}:
                raise ValueError(f"CSV: id vacío o role distinto de fit/check en fila {index}.")
            if not all(math.isfinite(point[name]) for name in ("column", "row", "longitude", "latitude")):
                raise ValueError(f"CSV: coordenada no finita en fila {index}.")
            if not (0 <= point["column"] <= width - 1 and 0 <= point["row"] <= height - 1):
                raise ValueError(f"CSV: punto {point['id']} fuera de la imagen {width} × {height}.")
            if not (-180 <= point["longitude"] <= 180 and -90 <= point["latitude"] <= 90):
                raise ValueError(f"CSV: coordenadas WGS84 inválidas en {point['id']}.")
            point["x"], point["y"] = transformer.transform(point["longitude"], point["latitude"])
            if not math.isfinite(point["x"]) or not math.isfinite(point["y"]):
                raise ValueError(f"CSV: no se pudo proyectar {point['id']}.")
            points.append(point)
    if len({point["id"] for point in points}) != len(points):
        raise ValueError("CSV: los identificadores deben ser únicos.")
    if len({(point["column"], point["row"]) for point in points}) != len(points):
        raise ValueError("CSV: un mismo píxel no puede ser punto de ajuste y comprobación ni repetirse.")
    if len({(point["longitude"], point["latitude"]) for point in points}) != len(points):
        raise ValueError("CSV: las coordenadas de control deben ser distintas.")
    return points


def fit_affine(points: list[dict], max_error_m: float = 3.0) -> tuple[Affine, list[dict], dict]:
    """Fit only role=fit; retain independent role=check points exclusively for checks."""
    if not math.isfinite(max_error_m) or max_error_m <= 0 or max_error_m > 3:
        raise ValueError("El límite de error debe ser positivo y no superar 3 metros.")
    fit_points = [point for point in points if point["role"] == "fit"]
    check_points = [point for point in points if point["role"] == "check"]
    if len(fit_points) < 4 or len(check_points) < 2:
        raise ValueError("Se requieren al menos 4 puntos fit y 2 puntos check independientes.")
    pixels = np.array([[point["column"] + .5, point["row"] + .5] for point in fit_points], dtype=float)
    world = np.array([[point["x"], point["y"]] for point in fit_points], dtype=float)
    if not np.isfinite(pixels).all() or not np.isfinite(world).all():
        raise ValueError("Los puntos contienen coordenadas no finitas.")
    center, scale = pixels.mean(axis=0), pixels.std(axis=0)
    if (scale < 1e-8).any():
        raise ValueError("Los puntos fit son degenerados: no abarcan ambas dimensiones.")
    design = np.column_stack(((pixels - center) / scale, np.ones(len(pixels))))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design) > 1e6:
        raise ValueError("Los puntos fit son colineales o casi colineales; ajuste degenerado.")
    # Subtract world mean as well, avoiding large UTM offsets in least squares.
    origin = world.mean(axis=0)
    coefficients, _, _, _ = np.linalg.lstsq(design, world - origin, rcond=None)
    slopes = coefficients[:2] / scale[:, None]
    intercept = origin + coefficients[2] - center @ slopes
    affine = Affine(slopes[0, 0], slopes[1, 0], intercept[0], slopes[0, 1], slopes[1, 1], intercept[1])
    linear = np.array([[affine.a, affine.b], [affine.d, affine.e]])
    if abs(affine.determinant) < 1e-12 or np.linalg.cond(linear) > 100:
        raise ValueError("La transformación ajustada es singular o presenta una deformación excesiva.")
    residuals = []
    for point in points:
        fitted_x, fitted_y = affine * (point["column"] + .5, point["row"] + .5)
        dx, dy = fitted_x - point["x"], fitted_y - point["y"]
        residuals.append({**point, "predicted_x": fitted_x, "predicted_y": fitted_y, "residual_x_m": dx, "residual_y_m": dy, "residual_m": math.hypot(dx, dy)})
    statistics = {}
    for role in ("fit", "check"):
        errors = np.array([point["residual_m"] for point in residuals if point["role"] == role])
        statistics[role] = {"count": len(errors), "rmse_m": float(np.sqrt(np.mean(errors ** 2))), "max_error_m": float(errors.max()), "mean_error_m": float(errors.mean())}
        if errors.max() > max_error_m:
            raise ValueError(f"Georreferenciación rechazada: error máximo {role} {errors.max():.3f} m > {max_error_m:.3f} m.")
    statistics.update({"acceptance_limit_m": max_error_m, "design_condition_number": float(np.linalg.cond(design)), "linear_condition_number": float(np.linalg.cond(linear)), "source_pixel_size_m": [math.hypot(affine.a, affine.d), math.hypot(affine.b, affine.e)]})
    return affine, residuals, statistics


def georeference_capture(
    image_path: str | Path, points_path: str | Path, output_path: str | Path,
    attribution: str, observed_at: str | None = None, resolution: float = .5,
    reference_image: str | Path | None = None, target_crs: str = "EPSG:32719",
    max_error_m: float = 3, title: str = "Chullo · captura real de Google Earth Pro",
    source_id: str = "google_earth_chullo_imagery", study_area_id: str = "chullo",
    source_url: str = "https://earth.google.com/", license_url: str | None = None,
    license_text: str = "Imagen exportada de Google Earth Pro para este prototipo académico; conserva créditos del proveedor y condiciones aplicables. No es un dataset abierto.",
) -> dict:
    image_path, points_path, output_path = Path(image_path).resolve(), Path(points_path).resolve(), Path(output_path).resolve()
    if output_path == image_path or output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("La salida debe ser un nuevo archivo .tif/.tiff; no sobrescriba la imagen original.")
    if not attribution.strip():
        raise ValueError("Es obligatorio conservar y declarar la atribución visible en la captura.")
    if observed_at:
        observed_at = date.fromisoformat(observed_at).isoformat()
    if not .05 <= resolution <= 2:
        raise ValueError("Resolución de salida admitida: entre 0.05 y 2 m/píxel.")
    crs = metric_crs(target_crs)
    original_hash = sha256_file(image_path)
    with Image.open(image_path) as original:
        if original.format not in {"JPEG", "PNG", "TIFF"}:
            raise ValueError("La captura debe ser JPG, PNG o TIFF.")
        width, height = original.size
        if max(width, height) > 32768 or width * height > 100_000_000:
            raise ValueError("La captura excede los límites de preparación (100 millones de píxeles).")
        if original.getexif().get(274, 1) != 1:
            raise ValueError("La imagen contiene orientación EXIF. Normalícela y mida los puntos sobre esa versión antes de continuar.")
        rgb = np.array(original.convert("RGB"))
        alpha = np.array(original.getchannel("A")) if "A" in original.getbands() else np.full((height, width), 255, dtype="uint8")
    reference_metadata = None
    if reference_image:
        reference_image = Path(reference_image).resolve()
        with Image.open(reference_image) as reference:
            if reference.size != (width, height):
                raise ValueError("La captura con puntos y la captura RGB deben tener dimensiones idénticas.")
        reference_metadata = {"path": str(reference_image), "sha256": sha256_file(reference_image), "note": "Imagen usada para localizar puntos; requiere la misma cámara y vista que la captura RGB."}
    points = read_points(points_path, width, height, target_crs)
    affine, residuals, stats = fit_affine(points, max_error_m=max_error_m)
    corners = np.array([affine * point for point in ((0, 0), (width, 0), (width, height), (0, height))])
    # Snap outward, preserving every original image edge; no geographic cropping.
    west = math.floor(float(corners[:, 0].min()) / resolution) * resolution
    east = math.ceil(float(corners[:, 0].max()) / resolution) * resolution
    south = math.floor(float(corners[:, 1].min()) / resolution) * resolution
    north = math.ceil(float(corners[:, 1].max()) / resolution) * resolution
    output_width, output_height = round((east - west) / resolution), round((north - south) / resolution)
    if min(output_width, output_height) < 64 or max(output_width, output_height) > 8192 or output_width * output_height > 16_000_000:
        raise ValueError(f"La imagen completa produciría {output_width} × {output_height} píxeles. El detector admite hasta 8192 por lado y 16 millones en total; ajuste la resolución o la vista de captura, sin inventar escala.")
    target_transform = from_origin(west, north, resolution, resolution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".preparing.tif")
    profile = {"driver": "GTiff", "width": output_width, "height": output_height, "count": 3, "dtype": "uint8", "crs": crs, "transform": target_transform, "compress": "deflate", "predictor": 2, "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(temporary, "w", **profile) as destination:
                for index in range(3):
                    reproject(source=rgb[:, :, index], destination=rasterio.band(destination, index + 1), src_transform=affine, src_crs=crs, dst_transform=target_transform, dst_crs=crs, resampling=Resampling.bilinear, num_threads=2)
                mask = np.zeros((output_height, output_width), dtype="uint8")
                reproject(source=alpha, destination=mask, src_transform=affine, src_crs=crs, dst_transform=target_transform, dst_crs=crs, resampling=Resampling.nearest, num_threads=2)
                destination.write_mask(mask)
                destination.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                destination.update_tags(ATTRIBUTION=attribution, SOURCE_IMAGE=str(image_path), SOURCE_IMAGE_SHA256=original_hash, GEOREFERENCE_METHOD="Affine least squares; independent check points", CAPTURE_DATE=observed_at or "Unknown", ACCURACY_NOTE="Residuals measure consistency with supplied coordinates, not independent ground accuracy.")
        # Verify the original has not changed while processing and pixels/georeference exist.
        if sha256_file(image_path) != original_hash:
            raise ValueError("La imagen original cambió durante la preparación; se rechaza el resultado.")
        with rasterio.open(temporary) as result:
            if result.crs != crs or not result.dataset_mask().any():
                raise ValueError("La salida no tiene CRS o píxeles válidos.")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    residual_path = output_path.with_name("georeferencing_residuals.csv")
    with residual_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "role", "column", "row", "longitude", "latitude", "x", "y", "predicted_x", "predicted_y", "residual_x_m", "residual_y_m", "residual_m"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(residuals)
    wgs_bounds = list(transform_bounds(crs, "EPSG:4326", west, south, east, north, densify_pts=21))
    georeferencing = {
        "method": "Affine least squares in metric CRS, fit/check points kept separate",
        "target_crs": crs.to_string(), "source_affine_gdal_corner_convention": list(affine)[:6],
        "pixel_coordinates": "Zero-based pixel-centre indices in input CSV. GDAL transform uses column+0.5,row+0.5.",
        "fit_check_statistics": stats, "control_points_path": str(points_path),
        "control_points_sha256": sha256_file(points_path), "residuals_path": str(residual_path),
        "control_points": residuals, "reference_image": reference_metadata,
        "extent_policy": "Full original image transformed; bounds snapped outward. No cropping; credits preserved in pixels.",
        "accuracy_caveat": "Fit/check residuals describe internal consistency with supplied coordinates, which may come from the same Google Earth view. They do not establish ground accuracy, image-to-OSM registration accuracy, orthorectification quality, or roof-to-ground offsets. Distances remain exploratory.",
    }
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "image_id": "earth_capture_" + original_hash[:16],
        "source_id": source_id, "study_area_id": study_area_id, "title": title,
        "image_kind": "RGB exportada de Google Earth Pro; georreferenciación propia de captura cenital",
        "raster_path": str(output_path), "source": "Google Earth Pro; proveedor según atribución de la imagen",
        "organization": "Google Earth Pro / proveedor acreditado", "url": source_url,
        "original_image": {"path": str(image_path), "sha256": original_hash, "width": width, "height": height, "unmodified": True},
        "observed_at": observed_at, "capture_date": observed_at, "processed_at": created,
        "license": license_text, "license_url": license_url, "attribution": attribution,
        "native_resolution_m": None, "resolution_m": [resolution, resolution],
        "resolution_note": "Output grid spacing, not a claim about the sensor's native resolution or positional accuracy.",
        "crs": crs.to_string(), "bbox": [west, south, east, north], "bbox_wgs84": wgs_bounds,
        "width": output_width, "height": output_height, "bands": 3, "dtype": "uint8",
        "valid_pixel_fraction": float(np.count_nonzero(mask) / mask.size),
        "sha256": sha256_file(output_path), "georeferencing": georeferencing,
        "accuracy_notes": georeferencing["accuracy_caveat"],
        "processing": "Complete original image warped with fitted affine transform; bilinear RGB resampling, nearest-neighbour validity mask. No superresolution, training or pixel content replacement.",
    }
    manifest_path = output_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attribution", required=True, help="Exact visible credits from the exported image.")
    parser.add_argument("--observed-at", help="YYYY-MM-DD only when the imagery date is known; not today's export date.")
    parser.add_argument("--reference-image", help="Same-camera capture with visible control points.")
    parser.add_argument("--resolution", type=float, default=.5)
    parser.add_argument("--target-crs", default="EPSG:32719")
    parser.add_argument("--max-error-m", type=float, default=3)
    parser.add_argument("--title", default="Chullo · captura real de Google Earth Pro")
    parser.add_argument("--source-id", default="google_earth_chullo_imagery")
    parser.add_argument("--study-area-id", default="chullo")
    parser.add_argument("--source-url", default="https://earth.google.com/")
    parser.add_argument("--license-url")
    parser.add_argument("--license-text", default="Imagen exportada de Google Earth Pro para este prototipo académico; conserva créditos del proveedor y condiciones aplicables. No es un dataset abierto.")
    args = parser.parse_args()
    try:
        manifest = georeference_capture(
            args.image, args.points, args.output, attribution=args.attribution,
            observed_at=args.observed_at, resolution=args.resolution,
            reference_image=args.reference_image, target_crs=args.target_crs,
            max_error_m=args.max_error_m, title=args.title, source_id=args.source_id,
            study_area_id=args.study_area_id, source_url=args.source_url,
            license_url=args.license_url, license_text=args.license_text,
        )
        print(json.dumps({"status": "completed", "raster_path": manifest["raster_path"], "size": [manifest["width"], manifest["height"]], "sha256": manifest["sha256"], "statistics": manifest["georeferencing"]["fit_check_statistics"]}, ensure_ascii=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
