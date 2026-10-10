# MLOps — Orquestación con Airflow y MLflow

**MCD-402 · 2026-2**

Pipeline completo de Machine Learning con gestión de experimentos (MLflow), entrenamiento paralelo con soporte GPU, y orquestación programada con Apache Airflow — sin Docker.

---

## Estructura del proyecto

```
mlops-airflow/
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
│   ├── raw/                  ← colocar los CSV de entrada aquí
│   └── processed/            ← se genera automáticamente
├── models/                   ← modelos entrenados (organizados por timestamp)
├── mlflow/                   ← base de datos SQLite de experimentos
├── Makefile                  ← comandos de un solo paso
└── requirements.txt
```

---

## Requisitos

- Python 3.10 o superior
- Sistema operativo: Linux, macOS o Windows con WSL2 recomendado para Airflow

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
| `mlflow` | 2.12 | Gestión de experimentos |
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
jupyter notebook notebooks/mlflow_experimentos.ipynb
```

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

## Datos propios

Colocar los archivos CSV en `data/raw/`. El pipeline espera que la primera columna sea el target binario (`0`/`1`). Si la carpeta está vacía, el pipeline genera un dataset sintético automáticamente para que el flujo sea ejecutable desde el primer momento.
