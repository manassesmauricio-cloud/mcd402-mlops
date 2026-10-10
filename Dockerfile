# Entorno Linux para ejecutar el proyecto en Windows sin WSL
# (Airflow no corre nativo en Windows y Smart App Control puede bloquear las DLLs de pip).
# El código se monta como volumen; la imagen solo contiene las dependencias.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt notebook

ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/app/dags

WORKDIR /app
EXPOSE 8080 5000 8888
