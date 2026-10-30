FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

COPY requirements.txt .
# CPU wheels keep inference independent of a CUDA installation on the host.
# These torch/torchvision versions were used for the local code checks.
RUN python -m pip install --no-cache-dir \
    torch==2.15.0 torchvision==0.29.2 \
    --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY helper_lib/ ./helper_lib/
COPY api.py .
# Build from the final Colab export, which contains trained weights.
COPY outputs/generator.pt ./outputs/generator.pt
RUN python -c "from api import load_generator; load_generator(); print('Generator weights loaded successfully')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); assert r['model_file_present']"
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
