# 🌱 AgriPipe

**From messy agricultural Excel files to validated PyTorch tensors. Three steps, reproducible, traceable.**

> 🇮🇹 Versione italiana: [README.md](README.md)

---

## 🎯 Who this project is for

AgriPipe fills a concrete gap in digital agriculture: the distance between **raw field data** (sensors, digitised paper logs, hand-filled spreadsheets) and the **rigid format required by Machine Learning models**.

It's designed for three user profiles:

- **👨‍🔬 Agritech data scientists and researchers** who receive agronomic Excel files of varying quality and need to turn them into ML-ready datasets reproducibly.
- **🎓 Students in agronomy, environmental sciences, and sustainable agriculture** who want to bring a real dataset to a PyTorch model without writing cleaning code from scratch.
- **🌾 Agritech operators and developers at companies** (such as X Farm) who need a predictable, auditable pipeline to feed their yield-prediction models.

You don't need to be a PyTorch expert to use it: the Streamlit UI covers the entire flow with a few clicks.

---

## 💡 The problem it solves

A typical agronomic Excel file is a minefield:

- Dates in *Excel serial format* (`45123` instead of `2024-01-15`).
- Humidity recorded as `150%` (physically impossible).
- Three or four rows of company header before the actual header.
- Duplicate rows caused by sensor sync errors.
- Decimal separator `,` instead of `.` (common in Italian datasets).
- NaN values scattered everywhere, sometimes marked as `-`, `n.d.`, or empty cells.

Training Machine Learning models on such data requires **hours of manual cleanup** and introduces silent, hard-to-trace bugs. AgriPipe automates the whole process in a transparent 3-step pipeline, producing a self-documenting `.zip` bundle with PyTorch tensors, JSON metadata, and scaler parameters ready for training or inference.

---

## 🚀 How to use it

AgriPipe offers two fully supported usage modes:

### 🖥 Streamlit UI (recommended for exploration)

```bash
streamlit run app.py
```

A 3-step web app opens: upload the file, configure the cleaning, download the `.zip` bundle. Zero lines of code.

![AgriPipe UI — 3 steps](docs/screenshots/agripipe_ui.png)

### ⚙️ CLI (recommended for automated pipelines)

```bash
# Clean + tensorize with a regional preset
agripipe run --input data.xlsx --preset ulivo_pugliese --output model_input.pt

# Export full ML bundle (.pt + .json + .zip)
agripipe run -i data.xlsx -p vite_piemontese -e ./export/

# Generate synthetic test data
agripipe generate --rows 1000 --output data/synthetic.xlsx
```

Run `agripipe --help` for the full command list.

---

## 📦 What it produces

At the end of the pipeline you get a **`<name>.zip`** archive containing:

| File | Contents |
|------|----------|
| `<name>.pt` *(or `<name>_train.pt`, `_val.pt`, `_test.pt` if split is enabled)* | PyTorch bundle with `features`, `target`, `feature_names`, `scaler_mean`, `scaler_scale`, `metadata` |
| `<name>.json` | Full manifest: schema, units, per-column statistics, correlations, cleaning diagnostics, PyTorch usage example |

Everything is traceable: `metadata.json` includes the source file's SHA-256 hash and a `schema_lock_hash` that lets you detect when a dataset's shape changes.

### Loading into PyTorch (5 lines)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

bundle = torch.load("agripipe_export.pt", weights_only=False)
dataset = TensorDataset(bundle["features"], bundle["target"])
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

The PyTorch model is ready to train with no further transformations.

---

## 🏗 How it works: the 3 steps

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ Excel / CSV │──▶│  1. LOADER  │──▶│  2. CLEANER  │──▶│ 3. TENSORIZER  │──▶ .pt + .json + .zip
└─────────────┘   └─────────────┘   └──────────────┘   └────────────────┘
                    schema          imputation          scaling
                    validation      outliers (IQR/Z)    cat. encoding
                    SHA-256 hash    physical bounds     train/val/test
```

1. **Loader** — reads Excel (`.xlsx`/`.xls`) or CSV, detects dirty headers, normalises dates (including Excel serial format), validates the minimum schema (`date`, `field_id`, `temp`, `humidity`, `ph`, `yield`), and computes a SHA-256 fingerprint for traceability.

2. **Cleaner** — applies, in order: type coercion → configurable physical bounds → outlier detection (IQR or Z-score) → missing value imputation (mean, median, forward-fill, time interpolation) → deduplication. Every operation is counted and reported in the diagnostics.

3. **Tensorizer** — scales numeric features (`StandardScaler` or `RobustScaler`), encodes categoricals (`LabelEncoder` or `OneHotEncoder`), builds the PyTorch tensor, and optionally splits into train/val/test.

Each step produces a queryable, auditable output: this is not a black box.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/francesco5252/agripipe.git
cd agripipe

# Install in development mode (includes test dependencies)
pip install -e ".[dev]"
```

Requirements: **Python 3.10+**, any operating system (tested on Windows, Linux, macOS).

---

## 🧪 Development and testing

The project follows a disciplined TDD approach:

```bash
pytest                          # 38 tests, ~82% coverage
ruff check src tests app.py      # linting
black --check src tests app.py   # formatting
```

GitHub Actions CI automatically runs tests + lint on Python 3.10, 3.11, and 3.12 for every push.

---

## ⚠️ Known limits (intellectual honesty)

Knowing a tool's limits is part of its quality. AgriPipe **does NOT do**:

- **Fuzzy matching of column names** — the minimum schema (`date`, `field_id`, `temp`, `humidity`, `ph`, `yield`) is mandatory. If your Excel uses `Temperatura_C`, you must rename it first.
- **Unit conversion** — no Fahrenheit → Celsius, no inches → mm. Data is assumed to already be in canonical units (SI where possible).
- **Batch folder loading** — one file at a time. Merging multiple files is an external workflow choice.
- **Interpretive agronomic models** — no sustainability indices, no "green/yellow/red" scorecards. AgriPipe produces clean data, not agronomic judgements. This is an intentional design choice: separating data preparation from interpretation.
- **ML-based imputation (KNN, MICE)** — sticks to classical statistical methods for transparency and reproducibility.

These exclusions are **intentional**: they keep the pipeline predictable, debuggable, and scientifically verifiable.


## 📄 License

Distributed under the **MIT** license. See [`LICENSE`](LICENSE) for details.

---

<sub>Project built with a rigorous ML-Ops approach. For the full step-by-step development journey, see [`DOCUMENTAZIONE_LOG.md`](DOCUMENTAZIONE_LOG.md).</sub>
