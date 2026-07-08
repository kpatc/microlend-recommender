.PHONY: setup generate train register mlflow-ui dashboard api test clean

setup:
	python3 -m venv venv && \
	venv/bin/pip install --upgrade pip && \
	venv/bin/pip install -r requirements.txt
	@echo "Setup complete — microlend-recommender ready"

generate:
	venv/bin/python src/data/synthetic_generator.py

train:
	venv/bin/python src/pipeline.py

register:
	venv/bin/python -c "\
import yaml; \
from src.tracking import setup_mlflow, set_production, list_registered_models; \
config = yaml.safe_load(open('configs/config.yaml')); \
setup_mlflow(config); \
model_name = config['mlflow']['registry']['model_name']; \
promoted = set_production(model_name, metric='rmse'); \
[print(r) for r in list_registered_models()]; \
"

mlflow-ui:
	venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

dashboard:
	venv/bin/streamlit run dashboard/app.py

ui:
	PYTHONPATH=. venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api:
	PYTHONPATH=. venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	venv/bin/pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "Cleaned build artifacts"
