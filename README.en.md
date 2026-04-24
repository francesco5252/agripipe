# AgriPipe: Data-to-Tensor

> 🇮🇹 **[Leggi questo in italiano →](README.md)**

AgriPipe is THE prototype of an **MLOps** platform designed to ferry raw agronomic data (coming from Excel, drones, weather stations, or field operators) to the state of armored vector ecosystems (`.pt` PyTorch tensors) for predictive Deep Learning training.

---

## 1. Who is this project for

* **Data Scientists and AI Engineers in the AgriTech landscape**: who need to iterate rapidly and scale heavy models without getting lost in the jungle of "ad-hoc" preprocessing every time a new Excel arrives or the sensors change.
* **Digital Agronomists & Research Centers**: who need a usable and guaranteed interface ensuring the historicization of the test that automatically validates dirty data by applying agronomic cleaning rules.
* **Operations Teams**: aiming to standardize a perfect bridge between those who collect in the field and those who develop the inference layers in Python.

---

## 2. The problem it solves

Raw agricultural data is chaotic: humidity is meant as a ratio (`0.2`) and the following year expressed as a percentage (`20%`), pH often presents stratospheric typos (`pH 45.0` instead of `4.5`), and they are filled with _Not-a-Number (NaN)_.

Handling these bugs manually and separately between laboratories and the Machine Learning team slows down algorithmic scaling or leads models to overfit statistical noise.
**AgriPipe solves the manual preprocessing bottleneck**: it guarantees a type-safe and modular pipeline that performs massive data-cleaning while preserving the integrity of agronomic meaning and validating files to create safe and immutable splits (train, val, test) via real-time `scikit-learn` standardization.

---

## 3. How to use it

The repository is conceived around a solid User Interface via **Streamlit** (multipage architecture).
From the project folder, just launch:

```bash
streamlit run app.py
```

The browser will guide you through three pages, arranged in rigid steps to prevent misconfigurations:
1. `1_📥_Ingestion`: upload your CSV / Excel `(xls/xlsx)`. Explore the raw visual preview and verify the file's ingress hash.
2. `2_🧹_Refinery`: run the dynamic wizards by choosing outlier detection criteria (IQR, Z-Score, or None), imputation for missing values, and customized physical tolerance limits to prevent Deep Learning from interpreting biologically absurd numbers.
3. `3_📦_Tensorizer`: here you slice the dataset into the canonical sets (Training, Validation, and Testing), decide the desired Scaler (`Standard` / `Robust`), and apply categorical/numeric Tensorizers to them. Finally, the interface will extract and let you download a self-contained `.zip` file.

---

## 4. What it produces

At the end of the processing cycle, you generate the **Machine Learning Bundle**.
It is an exact `.zip` that you can ship to model development with the following outputs:
* **Multiple Tensors** (`_train.pt`, `_val.pt`, `_test.pt`): Serialized PyTorch objects composed of `features` vectors, the `target` label, and even `scaler_mean / scaler_scale`. Everything computed live ensuring zero Data Leakage (test values are computed on train estimators!).
* **Manifest `metadata.json`**: An indelible snapshot of the algorithm you ran. It shows in plain sight the dropouts, the selected features, the algorithmic presets originally applied.
* **MLflow Logging (Behind the scenes)**: Agripipe tacitly evaluates the mathematical integrity of the final package by computing a safety benchmark (Ridge Regression baseline) and tracking records via a side MLOps runtime if `mlflow` is active.

---

## 5. How the 3 Engines work (Loader, Cleaner, Tensorizer)

Behind the friendly interface, Agripipe implements an enterprise architecture rigorously controlled by Pydantic:

* **Engine I: Loader (`src/agripipe/loader.py`)**
  The ingestion demon. It leverages fuzzy matching and deep lexical intelligence to recognize synonyms (e.g., if the file reads `data_raccolta`, the Loader internally coerces the column to `date`). It supports **batch loading** from entire directories (`--input-dir`) and **automatic SI unit conversion** (Fahrenheit → Celsius, inch → mm, lb/acre → kg/ha) via the `--auto-units` flag. It handles cache hashing.

* **Engine II: Cleaner (`src/agripipe/cleaner.py` / `transformers.py`)**
  It is the statistical colossus disassembled into a scikit-learn pipeline. Under the hood there are **11 decoupled serial modules** ("Transformers"), in formal order: date coercion, Auto-Unit conversion, `pydantic` bound checker on physical scales of pH/Temp/Hum, outlier detection (IQR or Z-score), numeric imputation (`median` / `mean` / `ffill` / time interpolation), categorical imputation via mode, deduplication. It returns the sanitized and ready DataFrame. It injects the dynamic computation of Accumulated Growing Degree Days (GDD) if the biological base is set.

* **Engine III: Tensorizer (`src/agripipe/tensorizer.py` & `export.py`)**
  The hinge between Pandas Analytics and PyTorch AI. It breaks down and replaces the categorical string/integer pipeline into normalized matrices. It scales everything by encapsulating the `StandardScaler` or `RobustScaler` model. The generated instance is an encapsulated and safely exportable module. After tensorization, `tracking.py` computes a **safety benchmark** (Ridge Regression baseline) with optional **MLflow** logging: if the baseline predicts nothing, the problem is in the data, not in the model.

---

## 6. Installation instructions

Agripipe embraces `pyproject.toml` for native dependency installation. Use Python 3.11+.

```bash
# 1. Clone the code
git clone https://github.com/francesco5252/agripipe.git
cd agripipe

# 2. Set up a clean environment
python -m venv venv

# On macOS/Linux:    source venv/bin/activate
# On Windows:        venv\Scripts\activate

# 3. Install the repository in editable mode (includes all app deps)
pip install -e "."

# 4. Build (optional if you are a backend developer of the platform, in that case run `pip install -e ".[dev]"`)

# 5. Launch the Data-to-Tensor terminal
streamlit run app.py
```
