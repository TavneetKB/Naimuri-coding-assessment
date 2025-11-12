FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY pyproject.toml /app/pyproject.toml

COPY README.md /app/README.md

COPY src/ /app/src/


RUN python -m pip install --upgrade pip \
 && pip install .


COPY model/ /app/model/
COPY data/time_series_data /app/data/time_series_data

# Set environment variables
ENV MODEL_PATH=/app/model/lstm_covid_model.h5 \
    SCALER_PATH=/app/model/scaler.pkl \
    DATA_PATH=/app/data/time_series_data

# Expose API port 3000
EXPOSE 3000

CMD ["uvicorn", "lstm_covid_model.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "3000"]

