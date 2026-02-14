# Testing_1_Python - CSV Cleaner

Validates and cleans the `house_prices.csv` dataset: checks schema, coerces
types, enforces value constraints, and writes a cleaned CSV.

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
```

### Options

| Flag | Description |
|---|---|
| `--input PATH` | **(required)** Path to input CSV |
| `--output PATH` | Output CSV path (default: `output/house_prices_cleaned.csv`) |
| `--drop-duplicates` | Remove duplicate rows |
| `--verbose` | Enable debug-level logging |

## Running Tests

```bash
pytest
```
