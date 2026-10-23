"""Run from .venv-ml; stdout is newline-delimited JSON progress for the UI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def emit(**event):
    print(json.dumps(event, ensure_ascii=True), flush=True)


def _acquire_run_lock(output: Path) -> tuple[Path, bool]:
    """Reuse the UI lock identified by its child environment; CLI owns its lock."""
    lock = output / ".running.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return lock, True
    except FileExistsError:
        # Windows venv launchers may introduce an intermediary parent process.
        owner = os.environ.get("CHULLO_RUN_LOCK_OWNER")
        if owner and lock.read_text(encoding="utf-8").strip() == owner:
            return lock, False
        raise ValueError("Ya hay una detección ejecutándose en esta carpeta. Espere a que termine.") from None


def main(argv: list[str] | None = None, *, detector=None) -> int:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Detección real de edificaciones con Mask R-CNN USA preentrenado.")
    parser.add_argument("--image", "--raster", dest="image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", "--confidence", dest="threshold", type=float, default=0.5)
    parser.add_argument("--image-manifest")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument("--model-variant", choices=("usa", "spacenet2_khartoum_v1"), default="usa")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    lock, owns_lock, attempt = None, False, None
    try:
        from src.repositories.run_publication import publish_run

        output.mkdir(parents=True, exist_ok=True)
        lock, owns_lock = _acquire_run_lock(output)
        attempt = output / ".attempts" / uuid4().hex
        staging = attempt / "staged"
        staging.mkdir(parents=True)
        (attempt / "status.json").write_text(
            json.dumps({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
        completed_event = None

        def progress(**event):
            nonlocal completed_event
            with (attempt / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")
            if event.get("stage") == "completed":
                # The detector finished in staging; the UI must wait for publication.
                completed_event = event
            else:
                emit(**event)

        if detector is None:
            from src.building_detection import detect_buildings

            detector = detect_buildings
        detector(
            image_path=args.image,
            output_dir=staging,
            threshold=args.threshold,
            image_manifest=args.image_manifest,
            device=args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            mask_threshold=args.mask_threshold,
            min_area_m2=args.min_area_m2,
            progress=progress,
            **({"model_variant": args.model_variant} if args.model_variant != "usa" else {}),
        )
        manifest = publish_run(staging, output)
        emit(**(completed_event or {"stage": "completed", "progress": 1.0, "count": manifest["count"]}))
        # Diagnostic writes after a successful commit cannot invalidate that commit.
        try:
            (attempt / "status.json").write_text(
                json.dumps({"status": "completed", "completed_at": manifest.get("completed_at")}), encoding="utf-8"
            )
        except OSError as error:
            print(f"No se pudo actualizar el registro de la tentativa: {error}", file=sys.stderr)
        return 0
    except Exception as error:
        payload = {
            "status": "error",
            "stage": "error",
            "message": str(error),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if attempt is not None:
            try:
                (attempt / "status.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (attempt / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass  # Keep the original failure on stdout/stderr even on a full disk.
        emit(**payload)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if owns_lock and lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
