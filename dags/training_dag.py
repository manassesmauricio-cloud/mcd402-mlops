import os
import sys
import json
import pickle
from datetime import datetime, timedelta

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator


# ── Ruta raíz del proyecto (dos niveles arriba de dags/) ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_cfg = _load_config()

CRON        = _cfg["airflow"]["cron_schedule"]
DAG_OWNER   = _cfg["airflow"]["dag_owner"]
RETRIES     = _cfg["airflow"]["retries"]
RETRY_MIN   = _cfg["airflow"]["retry_delay_minutes"]

MLFLOW_URI  = os.path.join(PROJECT_ROOT, _cfg["mlflow"]["tracking_uri"].replace("sqlite:///", ""))
ARTIFACT_RT = os.path.join(PROJECT_ROOT, _cfg["mlflow"]["artifact_root"])
EXP_NAME    = _cfg["mlflow"]["experiment_name"]
TAGS        = _cfg["mlflow"].get("tags", {})

TRAIN_PATH  = os.path.join(PROJECT_ROOT, _cfg["data"]["train_path"])
TEST_PATH   = os.path.join(PROJECT_ROOT, _cfg["data"]["test_path"])
MODELS_DIR  = os.path.join(PROJECT_ROOT, _cfg["data"]["models_dir"])

N_ITER      = _cfg["training"]["n_iter_hpo"]
CV          = _cfg["training"]["cv_folds"]
N_JOBS      = _cfg["training"]["n_jobs"]
RANDOM_ST   = _cfg["training"]["random_state"]
DECAY_THR   = _cfg["training"]["decay_threshold"]
USE_GPU     = _cfg["training"]["use_gpu"]

default_args = {
    "owner":           DAG_OWNER,
    "depends_on_past": False,
    "start_date":      datetime(2026, 1, 1),
    "retries":         RETRIES,
    "retry_delay":     timedelta(minutes=RETRY_MIN),
    "email_on_failure": False,
}


# ── TASKS ─────────────────────────────────────────────────────────────────────

def task_preprocess(**ctx):
    from src.preprocessing import main as preprocess_main
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    if not csv_files:
        print("Sin datos reales — generando dataset sintético para demo")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=20_000, n_features=25, n_informative=15,
            random_state=42, class_sep=0.8
        )
        df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
        df.insert(0, "target", y)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
        print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
    else:
        preprocess_main(
            dir_rawdata=raw_dir,
            dir_processed=proc_dir,
            target_col="target",
        )

    ctx["ti"].xcom_push(key="train_path", value=TRAIN_PATH)
    ctx["ti"].xcom_push(key="test_path",  value=TEST_PATH)


def task_train(**ctx):
    from src.training import run_parallel_hpo, save_model
    import pandas as pd

    ti = ctx["ti"]
    train_path = ti.xcom_pull(key="train_path", task_ids="preprocesar_datos")
    test_path  = ti.xcom_pull(key="test_path",  task_ids="preprocesar_datos")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    target   = train_df.columns[0]
    X_train  = train_df.drop(columns=[target])
    y_train  = train_df[target]
    X_test   = test_df.drop(columns=[target]).reindex(columns=X_train.columns, fill_value=0)
    y_test   = test_df[target]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results, champion, par_meta = run_parallel_hpo(
        X_train, y_train, X_test, y_test,
        n_iter=N_ITER, cv=CV, n_jobs=N_JOBS,
        random_state=RANDOM_ST, decay_threshold=DECAY_THR,
        use_gpu=USE_GPU,
    )

    # serializar resultados para XCom (sin modelos, son demasiado grandes)
    results_meta = [
        {k: v for k, v in r.items() if k != "model"} for r in results
    ]

    tmp_dir = "/tmp/airflow_mlops"
    os.makedirs(tmp_dir, exist_ok=True)
    results_path = f"{tmp_dir}/results_{ts}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"results": results, "champion": champion, "par_meta": par_meta}, f)

    ti.xcom_push(key="results_path",  value=results_path)
    ti.xcom_push(key="results_meta",  value=json.dumps(results_meta))
    ti.xcom_push(key="par_meta",      value=json.dumps(par_meta))
    ti.xcom_push(key="champion_name", value=champion["name"] if champion else None)
    ti.xcom_push(key="run_timestamp", value=ts)
    ti.xcom_push(key="test_path",     value=test_path)


def task_log_mlflow(**ctx):
    import pandas as pd
    import mlflow
    from src.registry import setup_mlflow, log_all_runs, get_best_run

    ti = ctx["ti"]
    results_path = ti.xcom_pull(key="results_path", task_ids="entrenar_modelos")
    test_path    = ti.xcom_pull(key="test_path",     task_ids="entrenar_modelos")

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    results  = data["results"]
    par_meta = data["par_meta"]

    test_df = pd.read_csv(test_path)
    target  = test_df.columns[0]
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target]

    mlflow_db = os.path.join(PROJECT_ROOT, "mlflow", "mlflow.db")
    setup_mlflow(
        tracking_uri=f"sqlite:///{mlflow_db}",
        artifact_root=ARTIFACT_RT,
        experiment_name=EXP_NAME,
    )

    run_ids = log_all_runs(results, par_meta, X_test, y_test, extra_tags=TAGS)

    best = get_best_run(EXP_NAME, metric="auc_test")
    if best:
        ti.xcom_push(key="best_run_id",   value=best.info.run_id)
        ti.xcom_push(key="best_run_name", value=best.info.run_name)
        ti.xcom_push(key="best_auc",      value=best.data.metrics.get("auc_test"))


def task_report(**ctx):
    ti = ctx["ti"]
    results_meta  = json.loads(ti.xcom_pull(key="results_meta",  task_ids="entrenar_modelos"))
    champion_name = ti.xcom_pull(key="champion_name", task_ids="entrenar_modelos")
    par_meta      = json.loads(ti.xcom_pull(key="par_meta",      task_ids="entrenar_modelos"))
    best_run_id   = ti.xcom_pull(key="best_run_id",   task_ids="registrar_mlflow")

    lines = ["=" * 50, "REPORTE PIPELINE — AutoML", "=" * 50]
    lines.append(f"{'Modelo':<10} {'AUC CV':>8} {'AUC Test':>10} {'Decay':>8} {'Tiempo':>8}")
    lines.append("-" * 50)
    for r in sorted(results_meta, key=lambda x: x["auc_test"], reverse=True):
        lines.append(
            f"{r['name']:<10} {r['auc_cv']:>8.4f} {r['auc_test']:>10.4f} "
            f"{r['decay_pct']:>7.1f}% {r['elapsed_s']:>7.1f}s"
        )
    lines.append("=" * 50)
    lines.append(f"Campeón: {champion_name}")
    lines.append(f"Speedup CPU: {par_meta.get('speedup', '?')}×  |  GPU: {par_meta.get('gpu_used')}")
    lines.append(f"MLflow run_id: {best_run_id[:8] if best_run_id else 'N/A'}")
    print("\n".join(lines))


# ── DAG ──────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="automl_training_pipeline",
    default_args=default_args,
    description="Pipeline AutoML: preprocesar → HPO paralelo + GPU → MLflow → reporte",
    schedule_interval=CRON,
    catchup=False,
    tags=["mlops", "urp", "automl"],
) as dag:

    t1 = PythonOperator(task_id="preprocesar_datos",   python_callable=task_preprocess,  provide_context=True)
    t2 = PythonOperator(task_id="entrenar_modelos",    python_callable=task_train,       provide_context=True)
    t3 = PythonOperator(task_id="registrar_mlflow",    python_callable=task_log_mlflow,  provide_context=True)
    t4 = PythonOperator(task_id="generar_reporte",     python_callable=task_report,      provide_context=True)

    t1 >> t2 >> t3 >> t4
