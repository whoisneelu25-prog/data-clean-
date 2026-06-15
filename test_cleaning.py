"""
Unit tests for the data cleaning pipeline.
Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import pytest
from data_cleaning import (
    remove_duplicates,
    standardise_text,
    clean_emails,
    clean_dates,
    clean_numeric,
    clean_phones,
    remove_near_duplicates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id":         [1, 2, 2, 3],
        "name":       ["alice johnson", "BOB SMITH", "BOB SMITH", "Carol White"],
        "email":      ["alice@example.com", "bob@example.com", "bob@example.com", "bad-email"],
        "age":        [29, -5, -5, 300],
        "salary":     [55000, 62000, 62000, 999999],
        "join_date":  ["2021-03-15", "15/04/2020", "15/04/2020", "March 10 2018"],
        "department": ["engineering", "MARKETING", "MARKETING", "hr"],
        "phone":      ["5551234", "5559876543", "5559876543", None],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_remove_duplicates(sample_df):
    result = remove_duplicates(sample_df)
    assert len(result) == 3, "Should remove 1 exact duplicate row"


def test_standardise_text(sample_df):
    result = standardise_text(sample_df)
    assert result["name"].iloc[0] == "Alice Johnson"
    assert result["email"].iloc[0] == "alice@example.com"
    assert result["department"].iloc[1] == "Marketing"


def test_clean_emails_invalid(sample_df):
    result = clean_emails(sample_df)
    carol_email = result.loc[result["name"].str.strip().str.lower() == "carol white", "email"]
    assert pd.isna(carol_email.values[0]), "Invalid email should become NaN"


def test_clean_emails_valid(sample_df):
    result = clean_emails(sample_df)
    assert result.loc[result["email"] == "alice@example.com", "email"].values[0] == "alice@example.com"


def test_clean_dates(sample_df):
    result = clean_dates(sample_df, "join_date")
    parsed = result["join_date"].apply(lambda x: isinstance(x, (pd.Timestamp, type(pd.NaT))))
    assert parsed.all(), "All dates should be Timestamps or NaT"


def test_clean_numeric_age(sample_df):
    result = clean_numeric(sample_df)
    assert pd.isna(result.loc[result["id"] == 2, "age"].values[0]), "Negative age → NaN"
    assert pd.isna(result.loc[result["id"] == 3, "age"].values[0]), "Age 300 → NaN"


def test_clean_numeric_salary(sample_df):
    result = clean_numeric(sample_df)
    assert pd.isna(result.loc[result["id"] == 3, "salary"].values[0]), "Extreme salary → NaN"


def test_clean_phones(sample_df):
    result = clean_phones(sample_df, "phone")
    assert result["phone"].iloc[0] == "555-1234"        # 7-digit
    assert result["phone"].iloc[1] == "(555) 987-6543"  # 10-digit
    assert pd.isna(result["phone"].iloc[3])             # None → NaN


def test_remove_near_duplicates():
    df = pd.DataFrame({
        "email":  ["a@b.com", "a@b.com", "c@d.com"],
        "name":   ["Alice", "Alice", "Carol"],
        "salary": [50000, np.nan, 60000],
    })
    result = remove_near_duplicates(df, subset=["email"])
    assert len(result) == 2, "Should collapse duplicate emails to 1 row each"
