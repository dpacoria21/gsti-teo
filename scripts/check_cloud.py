"""Opt-in smoke check of the cloud UI, including a real CPU inference.

Run inside the built container: python scripts/check_cloud.py --run-inference
This creates a session result; it never replaces the bundled demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from streamlit.testing.v1 import AppTest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inference", action="store_true")
    args = parser.parse_args()
    demo = ROOT / "outputs/inference/chullo_demo"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in demo.iterdir() if p.is_file()}
    app = AppTest.from_file(str(ROOT / "cloud/app.py"), default_timeout=900).run()
    assert not app.exception, [element.value for element in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert app.metric[0].value == "237"
    report = {"startup": "ok", "saved_contours": 237}
    if args.run_inference:
        start = time.perf_counter()
        button = next(element for element in app.button if element.label == "Detectar en esta imagen")
        assert not button.disabled, "The cloud detector must be available."
        button.click().run()
        assert not app.exception, [element.value for element in app.exception]
        assert not app.error, [element.value for element in app.error]
        assert int(app.metric[0].value) > 0
        results = app.session_state["_session_detection_results"]
        output = Path(results["chullo:bundled"])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        from src.data_loader import load_study_area
        from src.export import csv_bytes, geojson_bytes
        from src.repositories.detections import load_detection
        from src.spatial_analysis import analyze_proximity

        detection = load_detection(output, area_id="chullo")
        data = load_study_area("chullo")
        analysis = analyze_proximity(detection["buildings"], data["rivers"], band_m=25)
        assert len(json.loads(geojson_bytes(analysis["summary"]))["features"]) == len(detection["buildings"])
        assert csv_bytes(analysis["summary"])
        report.update(
            inference="completed",
            elapsed_seconds=round(time.perf_counter() - start, 2),
            contours=int(app.metric[0].value),
            within_25m=int(app.metric[1].value),
            output_dir=str(output),
        )
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in demo.iterdir() if p.is_file()}
    peak = Path("/sys/fs/cgroup/memory.peak")
    if peak.exists():
        report["container_peak_memory_mib"] = round(int(peak.read_text()) / 1024**2, 1)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
