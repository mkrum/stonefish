# 🪨🐟 stonefish: Transformers For Chess

The code for the efficient RL based finetuning for language models can now be
found [here](https://github.com/mkrum/rl4ft).

## Installation

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mkrum/stonefish.git
cd stonefish

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode with all dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Docker

```bash
docker build -t stonefish .
docker run -it stonefish
```

## Testing

```bash
pytest
```

## Development Tools

This project uses modern Python development tools:

- **Black**: Code formatter
- **isort**: Import sorter
- **Ruff**: Fast linter
- **mypy**: Static type checking
- **pre-commit**: Automated code quality checks

These tools are configured in `pyproject.toml` and `.pre-commit-config.yaml`.