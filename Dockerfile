FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    IMPUTEAUDIT_RUNS_DIR=/app/runs IMPUTEAUDIT_LOG_DIR=/app/logs
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY imputeaudit/ imputeaudit/
COPY data/samples/ data/samples/
COPY app.py research_runner.py ./
RUN useradd --create-home appuser && mkdir -p runs logs && chown -R appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.maxUploadSize=50"]
