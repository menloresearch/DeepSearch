.PHONY: style quality install clean fix test check-data simple-qa generate-data eval-base eval-lora download-checkpoint upload-checkpoint save-merged-16bit

# make sure to test the local checkout in scripts and not the pre-installed one
export PYTHONPATH = src 

check_dirs := . src notebooks scripts tests

# Run tests
test:
	pytest -v tests/

# Development dependencies
install:
	pip install -e .

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

# Check Data
check-data:
	@echo "Checking generated data files..."
	python scripts/check_data.py

# Simple QA
simple-qa:
	python scripts/simple_qa.py

# Generate data
generate-data:
	python scripts/generate_data.py

# Evaluate base model
eval-base:
	python scripts/eval_base.py

# Evaluate LoRA model
eval-lora:
	python scripts/eval_lora.py

# Download checkpoint
download-checkpoint:
	python scripts/download_checkpoint.py

# Upload checkpoint
upload-checkpoint:
	python scripts/upload_checkpoint.py

# Save merged 16bit model
save-merged-16bit:
	python scripts/save_merged_16bit.py

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