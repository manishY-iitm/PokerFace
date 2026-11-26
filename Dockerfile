FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# Run with:
#   docker build -t raksha-pokerface .
#   docker run --rm --network none --cpus=2 --memory=4g \
#     -e RAKSHA_SECRET=release-secret \
#     -v /abs/path/to/input:/data/input:ro \
#     -v /abs/path/to/output:/data/output \
#     raksha-pokerface --input /data/input --output /data/output
ENTRYPOINT ["python", "main.py"]
