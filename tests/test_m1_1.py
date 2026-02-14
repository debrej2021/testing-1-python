from __future__ import annotations

import pandas as pd
import pytest

from m1_1 import (
    EXPECTED_COLUMNS,
    Config,
    clean_data,
    parse_args,
    validate_input_file,
    validate_schema,
    write_cleaned_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, **overrides):
    defaults = {
        "input_path": tmp_path / "input.csv",
        "output_path": tmp_path / "output.csv",
        "drop_duplicates": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _valid_df():
    return pd.DataFrame(
        {
            "SizeSqFt": [1000, 1500],
            "Bedrooms": [2, 3],
            "AgeYears": [10, 5],
            "Price": [80000, 120000],
        }
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_input(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self):
        cfg = parse_args(["--input", "data.csv"])
        assert str(cfg.input_path) == "data.csv"
        assert str(cfg.output_path) == "output/house_prices_cleaned.csv"
        assert cfg.drop_duplicates is False
        assert cfg.verbose is False

    def test_all_flags(self):
        cfg = parse_args([
            "--input", "in.csv",
            "--output", "out.csv",
            "--drop-duplicates",
            "--verbose",
        ])
        assert str(cfg.input_path) == "in.csv"
        assert str(cfg.output_path) == "out.csv"
        assert cfg.drop_duplicates is True
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# validate_input_file
# ---------------------------------------------------------------------------

class TestValidateInputFile:
    def test_missing_file(self, tmp_path):
        cfg = _make_config(tmp_path, input_path=tmp_path / "no_such.csv")
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_input_file(cfg)

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a,b\n1,2\n")
        cfg = _make_config(tmp_path, input_path=f)
        with pytest.raises(ValueError, match="Only .csv"):
            validate_input_file(cfg)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        cfg = _make_config(tmp_path, input_path=f)
        with pytest.raises(ValueError, match="empty"):
            validate_input_file(cfg)

    def test_valid_file(self, tmp_path):
        f = tmp_path / "ok.csv"
        f.write_text("a\n1\n")
        cfg = _make_config(tmp_path, input_path=f)
        validate_input_file(cfg)  # should not raise


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_schema(self):
        validate_schema(_valid_df())  # no exception

    def test_missing_column(self):
        df = _valid_df().drop(columns=["Price"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_extra_columns_warns(self):
        df = _valid_df()
        df["Extra"] = 1
        validate_schema(df)  # warns but does not raise


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_valid_data_unchanged(self, tmp_path):
        cfg = _make_config(tmp_path)
        result = clean_data(_valid_df(), cfg)
        assert len(result) == 2
        assert list(result.columns) == EXPECTED_COLUMNS

    def test_coerces_non_numeric(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = pd.DataFrame(
            {
                "SizeSqFt": [1000, 1500],
                "Bedrooms": [2, 3],
                "AgeYears": [10, 5],
                "Price": ["not_a_number", 120000],  # string in numeric column
            }
        )
        result = clean_data(df, cfg)
        assert len(result) == 1  # bad row dropped

    def test_negative_price_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "Price"] = -100
        with pytest.raises(ValueError, match="Price must be > 0"):
            clean_data(df, cfg)

    def test_zero_bedrooms_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "Bedrooms"] = 0
        with pytest.raises(ValueError, match="Bedrooms must be > 0"):
            clean_data(df, cfg)

    def test_negative_age_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "AgeYears"] = -1
        with pytest.raises(ValueError, match="AgeYears must be >= 0"):
            clean_data(df, cfg)

    def test_zero_sqft_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "SizeSqFt"] = 0
        with pytest.raises(ValueError, match="SizeSqFt must be > 0"):
            clean_data(df, cfg)

    def test_drop_duplicates(self, tmp_path):
        cfg = _make_config(tmp_path, drop_duplicates=True)
        df = pd.concat([_valid_df(), _valid_df()], ignore_index=True)
        result = clean_data(df, cfg)
        assert len(result) == 2

    def test_deterministic_order(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df().iloc[::-1].reset_index(drop=True)  # reversed
        r1 = clean_data(df, cfg)
        r2 = clean_data(_valid_df(), cfg)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# write_cleaned_csv
# ---------------------------------------------------------------------------

class TestWriteCleanedCSV:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "sub" / "out.csv"
        cfg = _make_config(tmp_path, output_path=out)
        write_cleaned_csv(_valid_df(), cfg)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "out.csv"
        cfg = _make_config(tmp_path, output_path=out)
        write_cleaned_csv(_valid_df(), cfg)
        assert out.exists()
