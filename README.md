# Testing_1_Python - CSV Cleaner

Validates and cleans the `house_prices.csv` dataset: checks schema, coerces
types, enforces value constraints, writes a cleaned CSV, performs a
train/test split, and evaluates a baseline linear regression model.

**Requires Python 3.10+**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic run
python m1_1.py --input house_prices.csv

# Custom output path + drop duplicates + verbose logging
python m1_1.py --input house_prices.csv --output cleaned.csv --drop-duplicates --verbose

# Custom seed and test size
python m1_1.py --input house_prices.csv --seed 123 --test-size 0.3
```

### Options

| Flag | Description |
|---|---|
| `--input PATH` | **(required)** Path to input CSV |
| `--output PATH` | Output CSV path (default: `output/house_prices_cleaned.csv`) |
| `--drop-duplicates` | Remove duplicate rows |
| `--verbose` | Enable debug-level logging |
| `--seed INT` | Random seed for train/test split (default: 42) |
| `--test-size FLOAT` | Fraction of data for the test set (default: 0.2) |

## Project Structure

```
.
├── m1_1.py                 # Main script: validate, clean, split, evaluate
├── requirements.txt        # Python dependencies
├── tests/
│   └── test_m1_1.py        # Unit tests
├── house_prices.csv        # Input data (not tracked in git)
├── output/                 # Generated cleaned/split CSVs
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest

# Verbose output with individual test names
pytest tests/ -v
```
