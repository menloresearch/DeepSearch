.PHONY: style quality install tensorboard clean fix update-worklog test data download-musique prepare-musique-jsonl extract-musique-paragraphs build-musique-index prepare-musique-index prepare-all-musique check-data prepare-dev-data ensure-unzip download-all-models serve-retriever serve-generator run-evaluation download-flashrag-data download-flashrag-index download-retriever-model download-generator-model serve-all run-full-evaluation evaluation-download-models prepare-serving serve-background stop-serving

# make sure to test the local checkout in scripts and not the pre-installed one
export PYTHONPATH = src 

check_dirs := . src notebooks scripts tests

# Run tests
test:
	pytest -v tests/

# Development dependencies
install:
	pip install -e . && pip install -e third_party/FlashRAG

# Code quality and style
style:
	ruff format --line-length 119 --target-version py311 $(check_dirs)
	isort $(check_dirs)

quality:
	ruff check --line-length 119 --target-version py311 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 --max-line-length 119 $(check_dirs)

# Auto-fix issues
fix:
	ruff check --fix --line-length 119 --target-version py311 $(check_dirs)
	isort $(check_dirs)

# TensorBoard
tensorboard:
	tensorboard --logdir=trainer_output_*_runs --port=6006

# List available run directories
list-runs:
	@echo "Available run directories:"
	@ls -d trainer_output_*_runs 2>/dev/null || echo "No run directories found"

# Ensure unzip is available
ensure-unzip:
	@which unzip > /dev/null || (echo "Installing unzip..." && sudo apt-get update && sudo apt-get install -y unzip)
	@echo "✓ unzip is available"

# Data Preparation - One command to rule them all
data: download-musique prepare-musique-jsonl extract-musique-paragraphs build-musique-index prepare-dev-data check-data
	@echo "✨ All data preparation complete! ✨"

# Index Preparation
prepare-musique-index: build-musique-index
	@echo "Musique index preparation complete."

download-musique: ensure-unzip
	@echo "Downloading Musique dataset..."
	bash scripts/train_data/download_data_musique.sh
	@echo "Musique dataset ready in ./data/raw/"

prepare-musique-jsonl: download-musique
	@echo "Preparing Musique data (JSONL)..."
	python scripts/train_data/prepare_musique_jsonl.py
	@echo "Processed Musique JSONL ready in ./data/processed/questions.jsonl"

extract-musique-paragraphs: download-musique
	@echo "Extracting unique paragraphs from raw Musique data..."
	python scripts/train_data/extract_musique_paragraphs.py
	@echo "Musique paragraphs extracted to ./data/processed/paragraphs.csv"

build-musique-index: extract-musique-paragraphs
	@echo "Building Musique FAISS index from paragraphs..."
	python scripts/train_data/build_musique_index.py
	@echo "Musique FAISS index files saved to ./data/processed/"

# Combined Preparation
prepare-all-musique: data prepare-musique-index
	@echo "All Musique data and index preparation complete."

# Check Data
check-data:
	@echo "Checking generated data files..."
	python scripts/check_data.py

# Prepare Dev Data
prepare-dev-data: download-musique
	@echo "Preparing Musique DEV data (JSONL)..."
	python scripts/train_data/prepare_musique_dev_jsonl.py
	@echo "Processed Musique DEV JSONL ready in ./data/processed/questions_dev.jsonl"

# ======= SERVING COMMANDS =======

# Prepare everything needed for serving (download models and data)
prepare-serving: download-all-models
	@echo "✨ All models and data for serving prepared! ✨"
	@echo "You can now run services with:"
	@echo "  make serve-retriever"
	@echo "  make serve-generator"
	@echo "  or both with separate terminals"

# Download all required models and data for serving
download-all-models: download-flashrag-data download-flashrag-index download-retriever-model download-generator-model
	@echo "✨ All models and data downloaded! ✨"

# Download FlashRAG datasets
download-flashrag-data:
	@echo "Downloading FlashRAG datasets..."
	python scripts/serving/download_flashrag_datasets.py
	@echo "FlashRAG datasets downloaded!"

# Download FlashRAG index
download-flashrag-index:
	@echo "Downloading FlashRAG index..."
	python scripts/serving/download_flashrag_index.py
	@echo "FlashRAG index downloaded!"

# Download retriever model
download-retriever-model:
	@echo "Downloading retriever model..."
	python scripts/serving/download_retriever_model.py
	@echo "Retriever model downloaded!"

# Download generator model
download-generator-model:
	@echo "Downloading generator model..."
	python scripts/serving/download_generator_model.py
	@echo "Generator model downloaded!"

# Serve retriever
serve-retriever: download-retriever-model download-flashrag-index download-flashrag-data
	@echo "Starting retriever service..."
	python scripts/serving/serve_retriever.py --config scripts/serving/retriever_config.yaml

# Serve generator
serve-generator: download-generator-model
	@echo "Starting generator service..."
	python scripts/serving/serve_generator.py

# Start both services (retriever and generator) in the background
serve-background: prepare-serving
	@echo "Starting both retriever and generator services in background..."
	@mkdir -p logs
	@echo "Starting retriever in background..."
	@nohup python scripts/serving/serve_retriever.py --config scripts/serving/retriever_config.yaml > logs/retriever.log 2>&1 &
	@echo "Retriever started! PID: $$!"
	@echo "Starting generator in background..."
	@nohup python scripts/serving/serve_generator.py > logs/generator.log 2>&1 &
	@echo "Generator started! PID: $$!"
	@echo "✨ Both services running in background! ✨"
	@echo "Check logs in logs/retriever.log and logs/generator.log"
	@echo "To stop services: make stop-serving"

# Stop all serving processes
stop-serving:
	@echo "Stopping all serving processes..."
	@pkill -f 'python scripts/serving/serve_' || echo "No serving processes found"
	@echo "✅ All services stopped!"

# Serve all components
serve-all: download-all-models
	@echo "Starting all services..."
	@echo "Please run these commands in separate terminals:"
	@echo "  make serve-retriever"
	@echo "  make serve-generator"
	@echo ""
	@echo "Or run both in background with one command:"
	@echo "  make serve-background"
	@echo ""
	@echo "To stop background services:"
	@echo "  make stop-serving"

# ======= EVALUATION COMMANDS =======

# Download models needed for evaluation
evaluation-download-models: download-all-models
	@echo "✨ All models for evaluation downloaded! ✨"

# Run evaluation script
run-evaluation:
	@echo "Running evaluation..."
	python scripts/evaluation/run_eval.py --config scripts/evaluation/eval_config.yaml
	@echo "Evaluation complete! Results in scripts/evaluation/output_logs/"

# Run complete evaluation pipeline
run-full-evaluation: evaluation-download-models run-evaluation
	@echo "✨ Full evaluation pipeline complete! ✨"

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
	rm -rf ./data/raw ./data/processed # Clean raw and processed data
	# Clean up the old faiss_index directory if it exists
	rm -rf ./data/processed/faiss_index

# Update worklog in GitHub issue
update-worklog:
	gh api -X PATCH /repos/menloresearch/DeepSearch/issues/comments/2743047160 \
		-f body="$$(cat docs/00_worklog.md)" | cat && kill -9 $$PPID 