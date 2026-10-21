"""Cloud entry point: prepare verified weights before exposing the detector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CHULLO_ML_PYTHON", sys.executable)
os.environ.setdefault("CHULLO_DEPLOYMENT", "cloud")

import streamlit as st  # noqa: E402

from src.inference.model import ensure_model  # noqa: E402


@st.cache_resource(show_spinner="Preparando el modelo de edificaciones para este servidor…")
def prepare_model() -> str:
    """Download once per process; Streamlit serializes the shared initialization."""
    return str(ensure_model())


def main() -> None:
    try:
        prepare_model()
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"No se pudo preparar el modelo: {error}")
        st.info("Reintenta recargando la página. El servidor necesita descargar los pesos verificados de Hugging Face.")
        st.stop()
    from src.ui.application import main as render_application

    render_application()


if __name__ == "__main__":
    main()
