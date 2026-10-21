"""Executable architectural guarantees for reuse outside the Streamlit/ML runtimes."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_importing_entrypoint_does_not_render_or_launch_workers(tmp_path):
    """Importing the entrypoint is safe for tooling; rendering requires execution."""
    pytest.importorskip("streamlit")
    program = textwrap.dedent(
        """
        import sys
        import subprocess
        import streamlit
        from streamlit.delta_generator import DeltaGenerator

        def unexpected_side_effect(*args, **kwargs):
            raise AssertionError('Importing app must not render UI or start processes.')

        streamlit.set_page_config = unexpected_side_effect
        DeltaGenerator._enqueue = unexpected_side_effect
        subprocess.Popen = unexpected_side_effect
        sys.path.insert(0, sys.argv[1])
        import app
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_spatial_pipeline_runs_without_ui_or_model_dependencies(tmp_path):
    """An independent GIS consumer can load, analyze and export without either UI or ML."""
    program = textwrap.dedent(
        """
        import importlib.abc
        import json
        from io import BytesIO
        from pathlib import Path
        import sys

        class NoUiOrModel(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'streamlit', 'streamlit_folium', 'torch', 'torchvision'}:
                    raise AssertionError('Spatial core imported a UI/ML dependency: ' + fullname)
                return None

        sys.meta_path.insert(0, NoUiOrModel())
        project_root = Path(sys.argv[1])
        sys.path.insert(0, str(project_root))
        from src.data_loader import load_study_area
        from src.spatial_analysis import analyze_proximity
        from src.export import csv_bytes, geojson_bytes
        import pandas as pd

        data = load_study_area('chullo', root=project_root)
        result = analyze_proximity(data['buildings'], data['rivers'], band_m=50)
        table = pd.read_csv(BytesIO(csv_bytes(result['summary'])))
        features = json.loads(geojson_bytes(result['summary']))['features']
        assert len(table) == len(features) == 177
        assert int(table.within_band.sum()) == 60
        assert table.band_m.eq(50).all()
        assert table.analysis_crs.eq('EPSG:32719').all()
        assert {'streamlit', 'torch', 'torchvision'}.isdisjoint(sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
