"""Register a completed Chullo pilot run without replacing any existing model.

This command copies an already trained checkpoint and its evidence. It never
trains, performs inference, changes the default model, or publishes remotely.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.model import MODEL_ID, MODEL_SHA256, sha256_file  # noqa: E402
from src.model_catalog import LOCAL_VARIANT, model_presentation  # noqa: E402

DATASET_ID = "chullo_local_assistant_provisional_v1"
DATASET_BASE = "data/training/chullo_local_v1"
REPORTS = (
    "run.json",
    "epochs.json",
    "comparison.json",
    "baseline_val.json",
    "best_val.json",
    "best_val_stored_checkpoint.json",
    "baseline_test.json",
    "candidate_test.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def portable_copy(value, *, run_dir: Path, dataset_dir: Path, repo_root: Path = ROOT):
    """Preserve report content while removing machine/user-specific absolute paths."""
    if isinstance(value, dict):
        return {
            key: portable_copy(item, run_dir=run_dir, dataset_dir=dataset_dir, repo_root=repo_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_copy(item, run_dir=run_dir, dataset_dir=dataset_dir, repo_root=repo_root) for item in value]
    if not isinstance(value, str):
        return value
    was_absolute = PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    result = value
    for directory, replacement in ((run_dir, "training_run"), (dataset_dir, DATASET_BASE), (repo_root, ".")):
        variants = {str(directory), str(directory).replace("\\", "/"), str(directory).replace("/", "\\")}
        for variant in sorted(variants, key=len, reverse=True):
            result = re.sub(re.escape(variant.rstrip("/\\")), lambda _: replacement, result, flags=re.IGNORECASE)
    if result.startswith("./") or result.startswith(".\\"):
        result = result[2:]
    if was_absolute or result.startswith(("training_run\\", DATASET_BASE + "\\")):
        result = result.replace("\\", "/")
    if not result.startswith(("https://", "http://")) and (
        PureWindowsPath(result).is_absolute()
        or PurePosixPath(result).is_absolute()
        or re.search(r"[A-Za-z]:[\\/]|\\\\|(?<!\S)/(?:Users|home|tmp|private|mnt)/", result)
    ):
        return "[ruta local omitida]"
    return result


def used_parent_counts(dataset: dict) -> dict:
    parents = {}
    for scene in dataset.get("scenes", []):
        parent, split = scene.get("parent_scene_id"), scene.get("split")
        if not parent or split not in {"train", "val", "test"}:
            raise ValueError("Cada escena necesita padre original y partición explícitos.")
        if parent in parents and parents[parent] != split:
            raise ValueError("Un padre original no puede participar en dos particiones.")
        parents[parent] = split
    excluded = {item["scene_id"] for item in dataset.get("excluded_parents", [])}
    if excluded.intersection(parents):
        raise ValueError("Un padre excluido no puede aportar escenas al entrenamiento.")
    return {split: sum(value == split for value in parents.values()) for split in ("train", "val", "test")}


def metric_limitations(reports: dict) -> list[str]:
    """Describe the observed trade-offs instead of declaring a generic improvement."""
    baseline = reports["baseline_test.json"]["summary"]
    candidate = reports["candidate_test.json"]["summary"]
    selected = reports["best_val.json"]["summary"]["f1"]
    stored = reports["best_val_stored_checkpoint.json"]["summary"]["f1"]
    notes = [
        "Modelo experimental. Comparación en dos recortes test: "
        f"F1 {baseline['f1']:.5f} → {candidate['f1']:.5f}; "
        f"recuperación {baseline['recall']:.5f} → {candidate['recall']:.5f}; "
        f"precisión {baseline['precision']:.5f} → {candidate['precision']:.5f}; "
        f"falsos positivos {baseline['fp']} → {candidate['fp']}.",
    ]
    if candidate["recall"] > baseline["recall"] and candidate["fp"] > baseline["fp"]:
        notes.append("Recupera más referencias, pero también genera más falsos positivos; no es una mejora uniforme.")
    merges, previous_merges = candidate.get("merged_predictions"), baseline.get("merged_predictions")
    affected = candidate.get("references_affected_by_merges")
    previous_affected = baseline.get("references_affected_by_merges")
    if merges is not None and previous_merges is not None:
        notes.append(
            f"Predicciones que unen al menos dos referencias: {previous_merges} → {merges}; "
            f"referencias afectadas: {previous_affected} → {affected}. "
            + (
                "Las uniones no están resueltas; no permite afirmar que se separan techos o viviendas individuales."
                if merges > 0
                else "Cero fusiones observadas en esta prueba no garantiza separación correcta en otras imágenes."
            )
        )
    notes.append(
        f"Validación usada para seleccionar la época: F1={selected:.17g}; "
        f"validación de los pesos almacenados en FP16 y recargados en FP32: F1={stored:.17g}; "
        f"diferencia almacenado menos selección={stored - selected:.17g}. "
        "Las métricas test publicadas corresponden al checkpoint almacenado."
    )
    return notes


def validate_run(run_dir: Path, data_manifest: Path) -> tuple[dict, dict, dict, Path]:
    """Accept only the agreed complete pilot protocol and matching artifacts."""
    dataset = read_json(data_manifest)
    reports = {name: read_json(run_dir / name) for name in REPORTS}
    run, history, comparison = reports["run.json"], reports["epochs.json"], reports["comparison.json"]
    if run.get("status") != "completed":
        raise ValueError("Solo se puede registrar una corrida completada.")
    config = run.get("config", {})
    if config.get("epochs") != 50 or config.get("batch_size") != 1 or config.get("image_scale") != 1.5625:
        raise ValueError("El piloto requiere 50 épocas, batch_size=1 e image_scale=1.5625.")
    if not isinstance(history, list) or [item.get("epoch") for item in history] != list(range(1, 51)):
        raise ValueError("El historial debe acreditar las 50 épocas completas en orden.")
    if dataset.get("dataset_id") != DATASET_ID or run.get("dataset_id") != DATASET_ID:
        raise ValueError("La corrida debe utilizar el manifiesto del piloto local de Chullo.")
    if dataset.get("annotation_method") != "assistant_visual_manual" or dataset.get("human_reviewed") is not False:
        raise ValueError("Las anotaciones deben conservar su procedencia visual provisional sin revisión humana.")
    used = used_parent_counts(dataset)
    annotated = dataset.get("parent_split_counts", {})
    if used != dataset.get("used_parent_split_counts") or used["train"] < 1 or used["val"] != 2 or used["test"] != 2:
        raise ValueError("Los padres utilizados deben coincidir con las escenas: train positivo, dos val y dos test.")
    if annotated.get("train", 0) < used["train"] or annotated.get("val") != 2 or annotated.get("test") != 2:
        raise ValueError("Los conteos de padres anotados no son coherentes con los padres utilizados.")
    geography = dataset.get("geographic_validation", {})
    gaps = geography.get("minimum_parent_gap_m", {})
    if geography.get("cross_split_overlap_count") != 0 or set(gaps) != {"train_val", "train_test", "val_test"}:
        raise ValueError("Falta la validación geográfica de separación entre particiones.")
    if any(not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0 for gap in gaps.values()):
        raise ValueError("Las particiones deben tener separación geográfica positiva.")
    if run.get("dataset_manifest_sha256") != sha256_file(data_manifest):
        raise ValueError("El manifiesto del dataset cambió después de comenzar el entrenamiento.")
    if run.get("initial_checkpoint_sha256") != MODEL_SHA256 or run.get("production_model_modified") is not False:
        raise ValueError("La inicialización debe ser GeoAI USA fijado y el modelo original debe permanecer protegido.")
    if comparison.get("selection_used_test") is not False:
        raise ValueError("La selección de época no debe utilizar test.")
    test_ids = [scene["scene_id"] for scene in dataset.get("scenes", []) if scene.get("split") == "test"]
    if len(test_ids) != 2:
        raise ValueError("El piloto requiere exactamente dos escenas de prueba reservadas.")
    for side, filename in (("baseline", "baseline_test.json"), ("candidate", "candidate_test.json")):
        report = reports[filename]
        if [row.get("scene_id") for row in report.get("per_scene", [])] != test_ids:
            raise ValueError("La comparación debe conservar las mismas dos escenas test del manifiesto.")
        if report.get("summary") != comparison.get(side) or report["summary"].get("images") != 2:
            raise ValueError("Los resúmenes de comparación y evaluación test no coinciden.")
    best_epoch = run.get("best_epoch")
    scores = [item.get("validation", {}).get("f1") for item in history]
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("El historial debe contener F1 de validación finito por época.")
    expected_epoch = scores.index(max(scores)) + 1
    if best_epoch != expected_epoch or reports["best_val.json"]["summary"]["f1"] != max(scores):
        raise ValueError("La época elegida debe maximizar F1 en validación, conservando la primera en empates.")
    checkpoint = run_dir / "best_model.pth"
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size == 0
        or sha256_file(checkpoint) != run.get("checkpoint_sha256")
    ):
        raise ValueError("El checkpoint final no coincide con su SHA256 registrado.")
    return run, dataset, reports, checkpoint


def register(run_dir: Path, data_manifest: Path, output_dir: Path) -> Path:
    run_dir, data_manifest, output_dir = run_dir.resolve(), data_manifest.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("El destino ya existe; el registro nunca reemplaza pesos ni evidencia previos.")
    run, dataset, reports, checkpoint = validate_run(run_dir, data_manifest)
    presentation = model_presentation({"variant": LOCAL_VARIANT})
    used = used_parent_counts(dataset)
    annotated = dataset["parent_split_counts"]
    excluded_count = annotated["train"] - used["train"]
    presentation["description"] = (
        f"Modelo experimental. Ajuste piloto con {used['train']} recortes originales de entrenamiento "
        f"({annotated['train']} anotados; {excluded_count} excluido), "
        f"{used['val']} de validación y {used['test']} de prueba de una sola captura de Chullo, "
        "Google Earth Pro / Airbus, del 2026-03-04."
    )
    baseline, candidate = reports["baseline_test.json"], reports["candidate_test.json"]
    limitations = [
        *presentation["limitations"],
        *metric_limitations(reports),
        "Las distancias usan georreferenciación aproximada; este modelo no evalúa peligro ni autoriza construir.",
        "Las métricas usan dos recortes test y regiones evaluables: no equivalen a exactitud en toda Chullo o Arequipa.",
        "La imagen fuente conserva las condiciones de Google Earth y Airbus; no se declara licencia abierta.",
    ]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.register-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "staged"
        staged.mkdir()
        shutil.copyfile(checkpoint, staged / "weights.pth")
        checksum = sha256_file(staged / "weights.pth")
        if checksum != run["checkpoint_sha256"]:
            raise ValueError("Los pesos cambiaron durante la copia; no se publica el registro.")
        copied = {}
        for name, value in {**reports, "config.json": run["config"], "dataset_manifest.json": dataset}.items():
            public = portable_copy(value, run_dir=run_dir, dataset_dir=data_manifest.parent)
            write_json(staged / name, public)
            copied[name] = sha256_file(staged / name)
        metadata = {
            "id": "chullo/maskrcnn_chullo_local_v1",
            "variant": LOCAL_VARIANT,
            "name": "Mask R-CNN ajustado Chullo (piloto)",
            **presentation,
            "limitations": limitations,
            "filename": "weights.pth",
            "sha256": checksum,
            "revision": checksum,
            "size_bytes": (staged / "weights.pth").stat().st_size,
            "trained_locally": True,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_model_id": MODEL_ID,
            "initial_model_sha256": MODEL_SHA256,
            "training_image_scale": 1.5625,
            "inference_tile_size": 512,
            "storage_dtype": "float16",
            "inference_dtype": "float32",
            "image_source": "Google Earth Pro / Airbus",
            "image_capture_date": "2026-03-04",
            "annotation_method": "assistant_visual_manual",
            "human_reviewed": False,
            "dataset_id": DATASET_ID,
            "parent_split_counts": used,
            "used_parent_split_counts": used,
            "annotated_parent_split_counts": annotated,
            "excluded_parents": portable_copy(
                dataset.get("excluded_parents", []), run_dir=run_dir, dataset_dir=data_manifest.parent
            ),
            "split_counts": dataset.get("split_counts"),
            "best_epoch": run["best_epoch"],
            "evaluation_report": "comparison.json",
            "evaluation_summary": {
                "test_images": 2,
                "reference_instances": sum(row["reference_instances"] for row in candidate["per_scene"]),
                "score_threshold": run["config"]["score_threshold"],
                "mask_threshold": run["config"]["mask_threshold"],
                "mask_iou_threshold": run["config"]["match_iou"],
                "baseline": baseline["summary"],
                "candidate": candidate["summary"],
                "stored_validation": reports["best_val_stored_checkpoint.json"]["summary"],
                "selection_validation": reports["best_val.json"]["summary"],
                "stored_minus_selection_validation_f1": (
                    reports["best_val_stored_checkpoint.json"]["summary"]["f1"]
                    - reports["best_val.json"]["summary"]["f1"]
                ),
                "baseline_validation": reports["baseline_val.json"]["summary"],
                "selection_used_test": False,
                "interpretation": dataset.get(
                    "assessment_caveat", "Evaluación piloto con referencias visuales del asistente sin revisión humana."
                ),
            },
            "geography": "Recortes espacialmente separados de una sola captura de Chullo; evaluación piloto provisional.",
            "license": "Inicialización GeoAI MIT; imagen y derivados sujetos a condiciones Google Earth/Airbus. Véase LICENSE_AND_ATTRIBUTION.txt.",
            "url": "https://www.google.com/help/terms_maps-earth/",
            "dataset_manifest": "dataset_manifest.json",
            "dataset_manifest_paths_base": DATASET_BASE,
            "dataset_source_manifest_sha256": run["dataset_manifest_sha256"],
            "source_run_sha256": sha256_file(run_dir / "run.json"),
            "public_reports_sha256": copied,
            "portable_copy_note": "Las copias públicas omiten rutas personales. Las rutas de imágenes/etiquetas del dataset se resuelven desde dataset_manifest_paths_base; los hashes originales conservan su referencia al archivo fuente.",
        }
        write_json(staged / "manifest.json", metadata)
        (staged / "LICENSE_AND_ATTRIBUTION.txt").write_text(
            "CHULLO · MODELO PILOTO\n\n"
            "Inicialización: GeoAI building_footprints_usa (repositorio del modelo: MIT).\n"
            "https://huggingface.co/giswqs/geoai\n"
            "Imagen: Google Earth Pro / Airbus; fecha de captura declarada 2026-03-04.\n"
            "La imagen fuente conserva sus condiciones y atribución; no se afirma una licencia abierta "
            "ni un permiso general de redistribución o entrenamiento.\n"
            "https://www.google.com/help/terms_maps-earth/\n"
            "https://about.google/brand-resource-center/products-and-services/geo-guidelines/\n\n"
            "Anotaciones: delineación visual provisional del asistente (assistant_visual_manual), "
            "sin revisión humana (human_reviewed=false), sin etiquetas derivadas de predicciones.\n"
            f"Áreas originales utilizadas: train={used['train']}, val={used['val']}, test={used['test']}. "
            f"Áreas train anotadas={annotated['train']}, excluidas={excluded_count}. Una sola captura. "
            "La separación espacial no acredita independencia estadística ni generalización.\n"
            "Registro educativo experimental; detección de cubiertas, no catastro ni uso residencial. "
            "No evalúa peligro, faja marginal o permiso de construcción.\n\n"
            "dataset_manifest.json es una copia de auditoría. Sus rutas relativas de imágenes y etiquetas "
            f"se resuelven desde {DATASET_BASE}, no desde esta carpeta de pesos. "
            "No se incluyen imágenes ni etiquetas en este registro.\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError("El destino apareció durante la preparación; no se reemplaza.")
        staged.rename(output_dir)
    return output_dir / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / LOCAL_VARIANT)
    args = parser.parse_args(argv)
    print(register(args.run_dir, args.data_manifest, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
