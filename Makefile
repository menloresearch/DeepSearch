.PHONY: style quality install tensorboard clean fix update-worklog test

# make sure to test the local checkout in scripts and not the pre-installed one
export PYTHONPATH = src 

check_dirs := . src notebooks scripts tests

# Run tests
test:
	pytest -v tests/

# Development dependencies
install:
	python -m venv venv && . venv/bin/activate && pip install --upgrade pip
	pip install -r requirements.txt

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

# Update worklog in GitHub issue
update-worklog:
	gh api -X PATCH /repos/menloresearch/DeepSearch/issues/comments/2743047160 \
		-f body="$$(cat docs/00_worklog.md)" | cat && kill -9 $$PPID 