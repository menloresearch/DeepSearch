.PHONY: style quality install tensorboard clean fix update-worklog test data download-musique prepare-musique-jsonl extract-musique-paragraphs build-musique-index prepare-musique-index prepare-all-musique check-data prepare-dev-data

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

# Data Preparation
data: prepare-musique-jsonl
	@echo "Data preparation complete."

# Index Preparation
prepare-musique-index: build-musique-index
	@echo "Musique index preparation complete."

download-musique:
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
check-data: prepare-all-musique prepare-dev-data
	@echo "Checking generated data files..."
	python scripts/check_data.py

# Prepare Dev Data
prepare-dev-data: download-musique
	@echo "Preparing Musique DEV data (JSONL)..."
	python scripts/train_data/prepare_musique_dev_jsonl.py
	@echo "Processed Musique DEV JSONL ready in ./data/processed/questions_dev.jsonl"

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