"""Package code and prepared inputs without environments, logs or original downloads."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs" / "entrega" / "chullo-prototipo.zip"


def main():
    files = [ROOT / name for name in (
        "README.md", "app.py", "requirements.in", "requirements.lock.txt",
        "INICIAR.cmd", "INICIAR.ps1", "PREPARAR.ps1", "DETENER.ps1", "data/sources.json",
        "PREPARAR_IA.ps1", "requirements-ml.in", "requirements-ml.lock.txt", "models/manifest.json",
        "data/imagery/acquisition_report.json",
        "pyproject.toml", "requirements-dev.in", "requirements-dev.lock.txt", ".gitignore",
    )]
    for folder in ("src", "scripts", "tests", ".streamlit", "config", "docs", "data/processed", "outputs/presentacion", "outputs/google_earth"):
        files.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    for folder in ("data/imagery/chullo", "data/imagery/mariano_melgar", "outputs/inference/chullo_demo", "outputs/inference/mariano_melgar_demo"):
        files.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "source" not in path.parts and ".attempts" not in path.parts
                     and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".csv", ".json", ".geojson", ".md"}
                     and path.name not in {"full_preview.png", "ign_cenepred_print_response.json", "ign_print_response.json"})
    weights = ROOT / "models/building_footprints_usa.pth"
    if weights.is_file():
        files.append(weights)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(DESTINATION, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, arcname=Path("chullo-prototipo") / path.relative_to(ROOT))
    with ZipFile(DESTINATION) as archive:
        assert archive.testzip() is None
        assert "chullo-prototipo/data/processed/chullo/buildings.geojson" in archive.namelist()
    print(f"{DESTINATION}\n{len(files)} archivos; {DESTINATION.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
