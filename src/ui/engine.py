"""Optional adapter to the isolated detection worker."""

try:
    from src.detection_client import load_detection, ml_status, run_detection
except ImportError:
    load_detection = run_detection = None

    def ml_status():
        return {"available": False, "reason": "El motor de detección aún no está instalado."}
