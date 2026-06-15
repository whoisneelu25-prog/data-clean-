"""
Data Cleaning Pipeline
======================
Transforms raw, unstructured data into a reliable format for analysis.
Handles: duplicate entries, missing values, and inconsistent formats.

Usage:
    python data_cleaning.py --input raw_data.csv --output cleaned_data.csv
    python data_cleaning.py  # uses built-in sample dataset
"""

import pandas as pd
import numpy as np
import re
import argparse
import logging
from datetime import datetime
from io import StringIO

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Sample dirty dataset (embedded for demo)
# ─────────────────────────────────────────────
SAMPLE_CSV = """id,name,email,age,salary,join_date,department,phone
1,Alice Johnson,alice@example.com,29,55000,2021-03-15,Engineering,555-1234
2,BOB SMITH,bob@EXAMPLE.COM,,62000,15/04/2020,engineering,5559876
3,Alice Johnson,alice@example.com,29,55000,2021-03-15,Engineering,555-1234
4,Carol White,carol.white@example.com,35,N/A,2019-07-01,Marketing,555-4321
5,dave brown,dave@example,42,48000,2022-01-30,HR,
6,,eve@example.com,31,71000,2020-11-22,Engineering,555-6789
7,Frank Green,frank@example.com,-5,85000,March 10 2018,Finance,555-2468
8,Grace Hall,grace@example.com,27,,2023-05-19,Marketing,555-1357
9,Hank Lee,hank@example.com,300,52000,2021-08-08,HR,555-8642
10,Ivy Chen,ivy@example.com,26,61000,2024/02/14,Engineering,555-9753
11,BOB SMITH,bob@example.com,,62000,2020-04-15,Engineering,5559876
12,Jack Kim,jack@example.com,38,999999,2017-12-01,Finance,555-3579
13,Karen Patel,karen@example.com,44,58000,2018-06-30,Marketing,
14,Leo Torres,leo@,55,47000,2022-09-10,HR,555-7531
15,Mia Nguyen,mia@example.com,33,63000,2021-11-25,Engineering,555-8024
"""


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def load_data(path: str | None) -> pd.DataFrame:
    """Load CSV from file path or fall back to embedded sample."""
    if path:
        df = pd.read_csv(path)
        log.info(f"Loaded {len(df)} rows from '{path}'")
    else:
        df = pd.read_csv(StringIO(SAMPLE_CSV))
        log.info(f"Using built-in sample dataset ({len(df)} rows)")
    return df


def report(label: str, before: int, after: int) -> None:
    removed = before - after
    log.info(f"  [{label}] {before} → {after} rows  (removed {removed})")


# ─────────────────────────────────────────────
# Step 1 – Initial audit
# ─────────────────────────────────────────────

def audit(df: pd.DataFrame) -> dict:
    """Collect a snapshot of data quality issues before cleaning."""
    issues = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_per_col": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return issues


# ─────────────────────────────────────────────
# Step 2 – Remove exact duplicates
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    report("exact duplicates", before, len(df))
    return df


# ─────────────────────────────────────────────
# Step 3 – Standardise text columns
# ─────────────────────────────────────────────

def standardise_text(df: pd.DataFrame) -> pd.DataFrame:
    """Title-case names; lower-case emails and departments."""
    if "name" in df.columns:
        df["name"] = df["name"].str.strip().str.title()

    if "email" in df.columns:
        df["email"] = df["email"].str.strip().str.lower()

    if "department" in df.columns:
        df["department"] = df["department"].str.strip().str.title()

    log.info("  [text] Names title-cased, emails/departments lower-cased")
    return df


# ─────────────────────────────────────────────
# Step 4 – Validate & clean emails
# ─────────────────────────────────────────────

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$", re.IGNORECASE)

def clean_emails(df: pd.DataFrame) -> pd.DataFrame:
    if "email" not in df.columns:
        return df
    mask_invalid = ~df["email"].str.match(EMAIL_RE, na=False)
    n_invalid = mask_invalid.sum()
    df.loc[mask_invalid, "email"] = np.nan
    log.info(f"  [email] {n_invalid} invalid email(s) set to NaN")
    return df


# ─────────────────────────────────────────────
# Step 5 – Parse & standardise dates
# ─────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y/%m/%d", "%B %d %Y", "%b %d %Y",
]

def parse_date(val) -> pd.Timestamp | None:
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    return pd.NaT  # unparseable → NaT


def clean_dates(df: pd.DataFrame, col: str = "join_date") -> pd.DataFrame:
    if col not in df.columns:
        return df
    before_na = df[col].isna().sum()
    df[col] = df[col].apply(parse_date)
    after_na = df[col].isna().sum()
    log.info(
        f"  [dates] '{col}' standardised; "
        f"{after_na - before_na} unparseable value(s) set to NaT"
    )
    return df


# ─────────────────────────────────────────────
# Step 6 – Handle numeric columns
# ─────────────────────────────────────────────

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    age  : coerce non-numeric, flag biologically implausible values (< 16 or > 100)
    salary: coerce non-numeric; flag extreme outliers (> 3 IQR rule)
    """
    # --- age ---
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        mask_bad_age = ~df["age"].between(16, 100, inclusive="both")
        n_bad = mask_bad_age.sum()
        df.loc[mask_bad_age, "age"] = np.nan
        log.info(f"  [age] {n_bad} out-of-range value(s) set to NaN")

    # --- salary ---
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
        q1 = df["salary"].quantile(0.25)
        q3 = df["salary"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        lower = q1 - 3 * iqr
        mask_outlier = ~df["salary"].between(lower, upper, inclusive="both")
        n_out = mask_outlier.sum()
        df.loc[mask_outlier, "salary"] = np.nan
        log.info(
            f"  [salary] IQR range [{lower:.0f}, {upper:.0f}]; "
            f"{n_out} outlier(s) set to NaN"
        )

    return df


# ─────────────────────────────────────────────
# Step 7 – Standardise phone numbers
# ─────────────────────────────────────────────

def clean_phones(df: pd.DataFrame, col: str = "phone") -> pd.DataFrame:
    if col not in df.columns:
        return df

    def normalise(val):
        if pd.isna(val):
            return np.nan
        digits = re.sub(r"\D", "", str(val))
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if len(digits) == 7:
            return f"{digits[:3]}-{digits[3:]}"
        return np.nan  # can't interpret

    df[col] = df[col].apply(normalise)
    log.info(f"  [phone] '{col}' standardised to (NNN) NNN-NNNN format")
    return df


# ─────────────────────────────────────────────
# Step 8 – Drop near-duplicates on key fields
# ─────────────────────────────────────────────

def remove_near_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Drop rows that share the same email (or custom subset),
    keeping the record with fewer missing values.
    """
    if subset is None:
        subset = ["email"]
    subset = [c for c in subset if c in df.columns]
    if not subset:
        return df

    before = len(df)
    df = df.assign(_missing=df.isnull().sum(axis=1))
    df = (
        df.sort_values("_missing")
          .drop_duplicates(subset=subset, keep="first")
          .drop(columns="_missing")
    )
    report("near-duplicates (by email)", before, len(df))
    return df


# ─────────────────────────────────────────────
# Step 9 – Fill / impute missing values
# ─────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      - Numeric columns  → median (robust to outliers)
      - Categorical cols → mode
      - Dates            → left as NaT (cannot safely impute)
    """
    for col in df.columns:
        if df[col].dtype == "object":
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                n = df[col].isna().sum()
                df[col] = df[col].fillna(mode[0])
                if n:
                    log.info(f"  [impute] '{col}': {n} NaN(s) → mode '{mode[0]}'")
        elif pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            n = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            if n:
                log.info(f"  [impute] '{col}': {n} NaN(s) → median {med:.2f}")
        # datetime cols: leave NaT — imputing a hire date is dangerous

    return df


# ─────────────────────────────────────────────
# Step 10 – Final type enforcement
# ─────────────────────────────────────────────

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:
        df["age"] = df["age"].astype("Int64")   # nullable integer
    if "salary" in df.columns:
        df["salary"] = df["salary"].round(2)
    if "id" in df.columns:
        df["id"] = df["id"].astype("Int64")
    log.info("  [types] Dtypes enforced")
    return df


# ─────────────────────────────────────────────
# Step 11 – Add metadata column
# ─────────────────────────────────────────────

def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["cleaned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return df


# ─────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── Starting cleaning pipeline ──────────────────────")

    log.info("Step 1 → Remove exact duplicates")
    df = remove_duplicates(df)

    log.info("Step 2 → Standardise text fields")
    df = standardise_text(df)

    log.info("Step 3 → Validate emails")
    df = clean_emails(df)

    log.info("Step 4 → Parse dates")
    df = clean_dates(df, "join_date")

    log.info("Step 5 → Clean numeric columns")
    df = clean_numeric(df)

    log.info("Step 6 → Standardise phone numbers")
    df = clean_phones(df, "phone")

    log.info("Step 7 → Remove near-duplicates (by email)")
    df = remove_near_duplicates(df, subset=["email"])

    log.info("Step 8 → Impute remaining missing values")
    df = impute_missing(df)

    log.info("Step 9 → Enforce final data types")
    df = enforce_types(df)

    log.info("Step 10 → Add metadata")
    df = add_metadata(df)

    log.info("── Pipeline complete ────────────────────────────────")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────

def quality_report(before: dict, df_clean: pd.DataFrame) -> None:
    print("\n" + "═" * 55)
    print("  DATA QUALITY REPORT")
    print("═" * 55)
    print(f"  Rows   : {before['total_rows']:>4}  →  {len(df_clean):>4}")
    print(f"  Columns: {before['total_cols']:>4}  →  {len(df_clean.columns):>4}  (+cleaned_at)")
    print()

    print("  Missing values BEFORE cleaning:")
    for col, n in before["missing_per_col"].items():
        if n:
            pct = before["missing_pct"][col]
            print(f"    {col:<15} {n:>3} ({pct:.1f}%)")

    after_missing = df_clean.isnull().sum()
    remaining = after_missing[after_missing > 0]
    if remaining.empty:
        print("\n  Missing values AFTER cleaning: none ✓")
    else:
        print("\n  Missing values AFTER cleaning:")
        for col, n in remaining.items():
            print(f"    {col:<15} {n:>3}  (dates — intentionally preserved)")

    print()
    print("  Final dtypes:")
    for col, dtype in df_clean.dtypes.items():
        print(f"    {col:<15} {str(dtype)}")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--input",  "-i", default=None,
                        help="Path to raw CSV (omit to use built-in sample)")
    parser.add_argument("--output", "-o", default="cleaned_data.csv",
                        help="Path for cleaned CSV output")
    args = parser.parse_args()

    # Load
    df_raw = load_data(args.input)
    before = audit(df_raw)

    # Clean
    df_clean = clean(df_raw.copy())

    # Report
    quality_report(before, df_clean)

    # Save
    df_clean.to_csv(args.output, index=False)
    log.info(f"Cleaned data saved → '{args.output}'")

    # Preview
    print(df_clean.to_string(index=False))


if __name__ == "__main__":
    main()
