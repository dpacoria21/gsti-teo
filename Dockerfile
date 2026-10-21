FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHULLO_DEPLOYMENT=cloud \
    CHULLO_ML_PYTHON=/usr/local/bin/python \
    PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY cloud/requirements.txt cloud/requirements.txt
RUN python -m pip install -r cloud/requirements.txt

COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser .streamlit /app/.streamlit
COPY --chown=appuser:appuser cloud /app/cloud
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser data /app/data
COPY --chown=appuser:appuser models/manifest.json /app/models/manifest.json
COPY --chown=appuser:appuser outputs/inference /app/outputs/inference
COPY --chown=appuser:appuser tests /app/tests
COPY --chown=appuser:appuser pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/.runtime /app/data/uploads /app/outputs/inference/sessions \
    && chown -R appuser:appuser /app
USER appuser
RUN python -c "from src.inference.model import ensure_model; ensure_model()"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8501')+'/_stcore/health',timeout=4)"
CMD ["sh", "-c", "exec python -m streamlit run cloud/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
