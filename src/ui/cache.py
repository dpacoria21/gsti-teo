"""Streamlit caching adapters; scientific calculations stay in their services."""

from pathlib import Path

import streamlit as st

from src.data_loader import load_study_area
from src.settings import PROJECT_ROOT as ROOT
from src.spatial_analysis import analyze_proximity
from src.ui.engine import load_detection


@st.cache_data(show_spinner=False)
def cached_load(area_id, fingerprint):
    return load_study_area(area_id=area_id, root=ROOT)


@st.cache_data(show_spinner=False)
def cached_analysis(area_id, fingerprint, band_m, analysis_crs):
    data = cached_load(area_id, fingerprint)
    return analyze_proximity(data["buildings"], data["rivers"], band_m=band_m, analysis_crs=analysis_crs)


@st.cache_data(show_spinner=False)
def cached_detection(output_dir, fingerprint, area_id):
    return load_detection(Path(output_dir), area_id=area_id)


@st.cache_data(show_spinner=False)
def cached_detection_analysis(output_dir, detection_fingerprint, area_id, data_fingerprint, band_m, analysis_crs):
    detection = cached_detection(output_dir, detection_fingerprint, area_id)
    data = cached_load(area_id, data_fingerprint)
    return analyze_proximity(detection["buildings"], data["rivers"], band_m=band_m, analysis_crs=analysis_crs)
