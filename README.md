# MLOps — Orquestación con Airflow y MLflow

**MCD-402 · 2026-2**

Pipeline completo de Machine Learning con gestión de experimentos (MLflow), entrenamiento paralelo con soporte GPU, y orquestación programada con Apache Airflow. En Linux/macOS (o WSL2) corre sin Docker; en Windows nativo se ejecuta dentro de un contenedor (ver [Ejecución en Windows con Docker](#ejecución-en-windows-con-docker)).

---

## Estructura del proyecto

```
mcd402-mlops/
├── config/
│   └── pipeline_config.yml   ← parámetros del pipeline y cron de Airflow
├── dags/
│   └── training_dag.py       ← DAG de Airflow (lee el config al iniciar)
├── src/
│   ├── preprocessing.py      ← carga y limpieza de datos
│   ├── training.py           ← HPO + paralelismo CPU + detección GPU
│   └── registry.py           ← registro de experimentos en MLflow
├── notebooks/
│   └── mlflow_experimentos.ipynb  ← notebook principal de la sesión
├── data/
│   ├── raw/                  ← colocar los CSV de entrada aquí (incluye .gitkeep)
│   └── processed/            ← se genera automáticamente
├── models/                   ← modelos de save_model() (el DAG actual no lo usa)
├── mlflow/                   ← base de datos SQLite de experimentos (se genera)
├── mlruns/                   ← artefactos de MLflow: modelos, curvas ROC (se genera)
├── Dockerfile                ← entorno Linux para ejecutar en Windows
├── Makefile                  ← comandos de un solo paso
└── requirements.txt
```

---

## Requisitos

- Python 3.10 – 3.12 (Airflow 2.x no soporta versiones más nuevas)
- Sistema operativo: Linux, macOS o Windows con WSL2 recomendado para Airflow
- Windows sin WSL2: Docker Desktop (ver sección más abajo)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/manassesmauricio-cloud/mcd402-mlops.git
cd mcd402-mlops
```

### 2. Crear y activar un entorno virtual

```bash
# Crear entorno
python -m venv .venv

# Activar en Linux/macOS
source .venv/bin/activate

# Activar en Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias principales son:

| Librería | Versión mínima | Propósito |
|---|---|---|
| `mlflow` | 3.0 | Gestión de experimentos |
| `apache-airflow` | 2.9 | Orquestación de pipelines |
| `scikit-learn` | 1.4 | Modelos base y métricas |
| `xgboost` | 2.0 | Gradient boosting con soporte GPU |
| `lightgbm` | 4.3 | Gradient boosting eficiente |
| `catboost` | 1.2 | Gradient boosting para variables categóricas |
| `joblib` | 1.3 | Paralelismo de procesos |
| `pandas` | 2.1 | Manejo de datos |
| `pyyaml` | 6.0 | Lectura del archivo de configuración |

> **GPU (opcional):** Para activar entrenamiento en GPU, instalar las dependencias CUDA
> correspondientes a tu hardware. El proyecto detecta GPU automáticamente en tiempo de ejecución.
> Si no hay GPU disponible, corre en CPU sin cambios en el código.

---

## Inicio rápido — Notebook

La forma más directa de explorar el proyecto es el notebook:

```bash
pip install notebook   # Jupyter no está en requirements.txt
jupyter notebook notebooks/mlflow_experimentos.ipynb
```

En Windows con Docker, ver el comando de notebook en [Ejecución en Windows con Docker](#ejecución-en-windows-con-docker).

El notebook guía el flujo completo:
1. Cargar o generar datos
2. Iniciar el tracking de MLflow
3. Ejecutar HPO con entrenamiento paralelo
4. Comparar experimentos por API
5. Cargar el modelo campeón
6. Abrir la UI de MLflow en `http://localhost:5000`

---

## Configurar el pipeline

Todos los parámetros están centralizados en `config/pipeline_config.yml`:

```yaml
airflow:
  cron_schedule: "0 */2 * * *"   # cada 2 horas — modificar aquí

training:
  n_iter_hpo: 15     # iteraciones de búsqueda por modelo
  cv_folds: 3        # folds de validación cruzada
  n_jobs: -1         # -1 = todos los cores disponibles
  use_gpu: auto      # auto | true | false
```

**Ejemplos de cron:**

| Expresión | Frecuencia |
|---|---|
| `0 */2 * * *` | cada 2 horas (default) |
| `0 6 * * *` | todos los días a las 6 am |
| `0 0 * * 1` | cada lunes a medianoche |
| `@daily` | una vez al día |

---

## Iniciar Airflow (sin Docker)

> Los comandos `make` requieren Linux, macOS o WSL2 (usan `make` y sintaxis bash).

```bash
# Solo la primera vez — inicializa la base de datos
make airflow-init

# Iniciar el scheduler + webserver
make airflow-up
```

Abrir `http://localhost:8080` (usuario: `admin` / contraseña: `admin`).

Para probar el DAG sin esperar el cron:

```bash
make dag-test    # corre el DAG una vez con fecha de hoy
make dag-run     # dispara el DAG manualmente
```

---

## Abrir MLflow UI

```bash
make mlflow-ui
```

Luego abrir `http://localhost:5000` para ver y comparar todos los experimentos.

---

## Ejecución en Windows con Docker

Airflow no corre nativo en Windows y, si **Smart App Control** está activo, Windows bloquea las DLLs
de los paquetes instalados con pip (`DLL load failed ... Control de aplicaciones bloqueó este archivo`).
En ese caso se usa el `Dockerfile` del repo como entorno Linux; el código se monta como volumen.

```powershell
# Construir la imagen (una sola vez, o al cambiar requirements.txt)
docker build -t mcd402-mlops .

# Ejecutar el DAG completo una vez (equivale a make airflow-init + make dag-test)
docker run --rm -v "${PWD}:/app" mcd402-mlops bash -c "airflow db migrate && airflow dags test automl_training_pipeline"

# UI de MLflow → http://localhost:5000
docker run --rm -p 5000:5000 -v "${PWD}:/app" mcd402-mlops mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 0.0.0.0 --port 5000

# Notebook → abrir la URL con token que imprime la consola
docker run --rm -p 8888:8888 -v "${PWD}:/app" mcd402-mlops jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```

El DAG completo tarda ~12 minutos en CPU con la configuración por defecto
(3 modelos × 15 iteraciones × 3 folds, entrenados dos veces para medir el speedup secuencial vs. paralelo).

> **Nota sobre MLflow 3.x:** desde MLflow 3 los modelos sklearn se serializan con `skops` por defecto,
> que rechaza XGBoost. Por eso `src/registry.py` usa `serialization_format="cloudpickle"` en `log_model`.

---

## Datos propios

Colocar los archivos CSV en `data/raw/`. El pipeline espera que la primera columna sea el target binario (`0`/`1`). Si la carpeta está vacía, el pipeline genera un dataset sintético automáticamente para que el flujo sea ejecutable desde el primer momento.
