# 🧹 Data Cleaning Pipeline

A complete data cleaning project that transforms raw, messy datasets into reliable, analysis-ready formats.

Covers the core concepts from the [Data Cleaning Essentials tutorial](https://youtu.be/jxq4-KSB_OA) — applied in Python with pandas.

---

## 📁 Project Structure

```
data-cleaning-project/
│
├── data/
│   ├── raw/
│   │   └── employees_raw.csv            # Original dirty dataset
│   └── cleaned/
│       └── employees_cleaned.csv        # Output after cleaning
│
├── notebooks/
│   └── data_cleaning_walkthrough.ipynb  # Step-by-step notebook
│
├── scripts/
│   └── data_cleaning.py                 # Reusable cleaning pipeline
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚨 Problems in the Raw Data

| Issue | Example |
|---|---|
| Duplicate rows | Row 1 and Row 3 are identical |
| Inconsistent casing | `BOB SMITH`, `dave brown`, `EXAMPLE.COM` |
| Invalid emails | `dave@example`, `leo@` |
| Mixed date formats | `15/04/2020`, `March 10 2018`, `2024/02/14` |
| Out-of-range values | Age `-5`, Age `300` |
| Salary outliers | `999999`, `N/A` |
| Missing values | Blank names, salaries, phones |
| Malformed phones | `5559876` (7 digits, no formatting) |

---

## ✅ Cleaning Steps

| Step | What It Does |
|---|---|
| 1 | Remove exact duplicate rows |
| 2 | Standardise text — title-case names, lowercase emails/departments |
| 3 | Validate emails with regex, null out malformed ones |
| 4 | Parse dates from 6+ formats to ISO YYYY-MM-DD |
| 5 | Flag out-of-range ages (< 16 or > 100) to NaN |
| 6 | Detect salary outliers via IQR rule to NaN |
| 7 | Standardise phone numbers to (555) 123-4567 format |
| 8 | Remove near-duplicates by email (keep most complete record) |
| 9 | Impute missing values — median for numbers, mode for categories |
| 10 | Enforce correct dtypes + add cleaned_at timestamp |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/data-cleaning-project.git
cd data-cleaning-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the cleaning script
```bash
# Use the built-in sample dataset
python scripts/data_cleaning.py

# Use your own CSV
python scripts/data_cleaning.py --input data/raw/employees_raw.csv --output data/cleaned/employees_cleaned.csv
```

### 4. Or open the notebook
```bash
jupyter notebook notebooks/data_cleaning_walkthrough.ipynb
```

---

## 🛠 Tech Stack

- **Python 3.11+**
- **pandas** — data manipulation
- **numpy** — numeric operations
- **Jupyter** — interactive walkthrough

---

## 📚 Reference

Tutorial: [Master Data Cleaning Essentials on Excel in Just 10 Minutes – Kenji Explains](https://youtu.be/jxq4-KSB_OA)
