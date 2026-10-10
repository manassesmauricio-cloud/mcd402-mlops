AIRFLOW_HOME := $(CURDIR)/airflow_home
PYTHON       := python

.PHONY: setup airflow-init airflow-up mlflow-ui dag-test dag-run clean

setup:
	pip install -r requirements.txt

airflow-init:
	AIRFLOW_HOME=$(AIRFLOW_HOME) \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	airflow db migrate
	AIRFLOW_HOME=$(AIRFLOW_HOME) \
	airflow users create \
		--username admin --password admin \
		--firstname MLOps --lastname URP \
		--role Admin --email admin@urp.edu.pe

airflow-up:
	AIRFLOW_HOME=$(AIRFLOW_HOME) \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	airflow standalone

mlflow-ui:
	mlflow ui \
		--backend-store-uri sqlite:///mlflow/mlflow.db \
		--default-artifact-root mlflow/artifacts \
		--host 0.0.0.0 --port 5000

dag-test:
	AIRFLOW_HOME=$(AIRFLOW_HOME) \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	airflow dags test automl_training_pipeline $$(date +%Y-%m-%d)

dag-run:
	AIRFLOW_HOME=$(AIRFLOW_HOME) \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	airflow dags trigger automl_training_pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf models/*/
	rm -f mlflow/mlflow.db
	rm -rf mlflow/artifacts/
