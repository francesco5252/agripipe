# AgriPipe Pro Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform AgriPipe from a functional prototype into a polished, professional data pipeline for Agritech X Farm — with Clean & Nature UI, time-series aware cleaning, Sustainability Score Card, and ML-ready bundle exports (.pt + metadata.json).

**Architecture:** Modular Python package (`src/agripipe/`) with isolated responsibilities: `cleaner` handles data transformations and raises `CleanerDiagnostics`; `sustainability` converts diagnostics to semaphore badges (pure function); `export` bundles training-ready `.pt` + `metadata.json` + `.zip`; `ui/theme` and `ui/components` are the single source of visual truth for the Streamlit app. The CLI keeps all existing commands but now auto-writes metadata.

**Tech Stack:** Python 3.10+, pandas, PyTorch, Streamlit, pytest. Existing stack; no new runtime dependencies.

**Reference spec:** [docs/superpowers/specs/2026-04-17-agripipe-pro-upgrade-design.md](../specs/2026-04-17-agripipe-pro-upgrade-design.md)

---

## File Structure

| Path | Responsibility | Status |
|---|---|---|
| `src/agripipe/cleaner.py` | Pipeline pulizia + diagnostics + time imputation | Modify |
| `src/agripipe/indices.py` | Indici agronomici (GDD, Huglin, ecc.) — solo docstring | Modify |
| `src/agripipe/loader.py` | Lettura Excel + validazione schema — solo docstring | Modify |
| `src/agripipe/tensorizer.py` | DataFrame → Tensor — solo docstring | Modify |
| `src/agripipe/dataset.py` | PyTorch Dataset wrapper — solo docstring | Modify |
| `src/agripipe/sustainability.py` | Compute scorecard (pura, no I/O) | **Create** |
| `src/agripipe/metadata.py` | Build + save metadata.json | **Create** |
| `src/agripipe/export.py` | Orchestrazione bundle ML (.pt + .json + .zip) | **Create** |
| `src/agripipe/ui/__init__.py` | Package marker | **Create** |
| `src/agripipe/ui/theme.py` | Palette + inject_css | **Create** |
| `src/agripipe/ui/components.py` | Componenti Streamlit riusabili | **Create** |
| `src/agripipe/app.py` | Composizione UI | Rewrite |
| `src/agripipe/cli.py` | CLI + metadata.json sidecar | Modify |
| `configs/agri_knowledge.yaml` | +region +crop_display +7 preset +regole colture | Modify |
| `tests/test_cleaner_diagnostics.py` | Test diagnostics counting | **Create** |
| `tests/test_cleaner_time_imputation.py` | Test strategia "time" + fallback | **Create** |
| `tests/test_sustainability.py` | Test soglie badge + overall_message | **Create** |
| `tests/test_metadata.py` | Test metadata.json schema | **Create** |
| `tests/test_export.py` | Test bundle (.pt + json + zip) | **Create** |
| `tests/test_e2e.py` | +test_exported_bundle_is_training_ready | Modify |

---

## Task 1: Add `CleanerDiagnostics` dataclass

**Files:**
- Modify: `src/agripipe/cleaner.py`
- Test: `tests/test_cleaner_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cleaner_diagnostics.py`:

```python
"""Test che CleanerDiagnostics sia istanziabile e collegato a AgriCleaner."""

from agripipe.cleaner import AgriCleaner, CleanerConfig, CleanerDiagnostics


def test_cleaner_has_diagnostics_attribute():
    config = CleanerConfig(numeric_columns=["temp"])
    cleaner = AgriCleaner(config)
    assert isinstance(cleaner.diagnostics, CleanerDiagnostics)
    assert cleaner.diagnostics.total_rows == 0


def test_diagnostics_has_all_expected_fields():
    d = CleanerDiagnostics()
    expected_fields = {
        "total_rows", "imputation_strategy_used", "values_imputed",
        "outliers_removed", "out_of_bounds_removed",
        "nitrogen_violations", "peronospora_events",
        "irrigation_inefficient", "soil_organic_low",
        "heat_stress_flowering", "late_frost_events",
    }
    for f in expected_fields:
        assert hasattr(d, f), f"Missing field: {f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cleaner_diagnostics.py -v`
Expected: FAIL with `ImportError: cannot import name 'CleanerDiagnostics'`

- [ ] **Step 3: Add the dataclass and wire it into AgriCleaner**

Modify `src/agripipe/cleaner.py`. After the `CleanerConfig` dataclass (around line 33), add:

```python
@dataclass
class CleanerDiagnostics:
    """Conteggi raccolti durante la pulizia per la Sustainability Score Card.
    
    Popolato da ``AgriCleaner.clean()`` e consumato da
    ``sustainability.compute_scorecard`` e ``metadata.build_metadata``.
    Non influenza il comportamento del cleaner: è solo esposizione dati.
    """
    total_rows: int = 0
    imputation_strategy_used: str = ""
    values_imputed: int = 0
    outliers_removed: int = 0
    out_of_bounds_removed: int = 0
    nitrogen_violations: int = 0
    peronospora_events: int = 0
    irrigation_inefficient: int = 0
    soil_organic_low: int = 0
    heat_stress_flowering: int = 0
    late_frost_events: int = 0
```

Update `AgriCleaner.__init__` to initialize `self.diagnostics`:

```python
def __init__(self, config: CleanerConfig):
    self.config = config
    self.knowledge = self._load_knowledge()
    self.diagnostics = CleanerDiagnostics()
```

Update `AgriCleaner.clean` to reset diagnostics at the start:

```python
def clean(self, df: pd.DataFrame) -> pd.DataFrame:
    """Applica l'intera pipeline di pulizia e calcola gli indici agronomici."""
    self.diagnostics = CleanerDiagnostics(total_rows=len(df))
    logger.info("Avvio pulizia intelligente su %d righe", len(df))
    df = df.copy()
    df = self._coerce_types(df)
    df = self._drop_sparse_columns(df)
    df = self._apply_agronomic_rules(df)
    df = self._apply_physical_bounds(df)
    df = self._handle_outliers(df)
    df = self._impute_missing(df)
    df = self._deduplicate(df)
    df = compute_agronomic_indices(df, self.knowledge)
    logger.info("Indici agronomici calcolati con successo.")
    logger.info("Pulizia completata: %d righe finali", len(df))
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cleaner_diagnostics.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Run full test suite to verify no regression**

Run: `pytest -x`
Expected: all existing tests PASS (diagnostics is additive).

- [ ] **Step 6: Commit**

```bash
git add src/agripipe/cleaner.py tests/test_cleaner_diagnostics.py
git commit -m "feat(cleaner): add CleanerDiagnostics dataclass for scorecard input"
```

---

## Task 2: Wire diagnostics counting into agronomic rules

**Files:**
- Modify: `src/agripipe/cleaner.py:79-170` (the `_apply_agronomic_rules` method)
- Modify: `src/agripipe/cleaner.py` (`_apply_physical_bounds`, `_handle_outliers`)
- Test: `tests/test_cleaner_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cleaner_diagnostics.py`:

```python
import pandas as pd
import numpy as np


def _synth_violations_df() -> pd.DataFrame:
    """DataFrame costruito ad hoc per generare violazioni note."""
    return pd.DataFrame({
        "date": pd.date_range("2024-05-01", periods=5, freq="D"),
        "field_id": ["F1"] * 5,
        "crop_type": ["wine_grape_docg"] * 5,
        "temp": [11.0, 35.0, 20.0, 20.0, 20.0],   # riga 1 = Regola Tre 10 se pioggia>10
        "rainfall": [12.0, 0.0, 0.0, 0.0, 0.0],    # riga 0 = Peronospora
        "humidity": [30.0, 50.0, 50.0, 50.0, 50.0],
        "soil_moisture": [12.0, 50.0, 90.0, 50.0, 50.0],
        "irrigation": [0.0, 0.0, 10.0, 0.0, 0.0],  # riga 2 = irrigazione su suolo saturo
        "n": [15.0, 0.0, 0.0, 0.0, 0.0],           # riga 0 = azoto su suolo secco
        "organic_matter": [1.0, 2.0, 2.0, 2.0, 2.0],  # riga 0 = suolo povero
        "ph": [7.0] * 5,
        "yield": [5.0] * 5,
    })


def test_diagnostics_counts_peronospora():
    from agripipe.cleaner import AgriCleaner, CleanerConfig
    config = CleanerConfig(
        numeric_columns=["temp", "rainfall", "humidity", "soil_moisture",
                         "irrigation", "n", "organic_matter", "ph", "yield"],
        date_columns=["date"],
    )
    cleaner = AgriCleaner(config)
    cleaner.clean(_synth_violations_df())
    assert cleaner.diagnostics.peronospora_events >= 1
    assert cleaner.diagnostics.nitrogen_violations >= 1
    assert cleaner.diagnostics.irrigation_inefficient >= 1
    assert cleaner.diagnostics.soil_organic_low >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cleaner_diagnostics.py::test_diagnostics_counts_peronospora -v`
Expected: FAIL with `assert 0 >= 1` (counters remain at 0).

- [ ] **Step 3: Add counting to agronomic rules**

In `src/agripipe/cleaner.py`, modify `_apply_agronomic_rules` — next to every `logger.warning(...)` that describes a violation, add a diagnostics increment. Full updated blocks:

```python
# --- C. CONCIMAZIONE SOSTENIBILE ---
if n_col and (rain_col or soil_moist_col):
    dry_soil = df[soil_moist_col] < 15 if soil_moist_col else True
    no_rain = df[rain_col] < 2 if rain_col else True
    mask = (df[n_col] > 10) & dry_soil & no_rain
    if mask.any():
        count = int(mask.sum())
        logger.warning("Sostenibilità: %d concimazioni su terreno troppo secco.", count)
        self.diagnostics.nitrogen_violations += count

# --- E. EFFICIENZA IDRICA ---
if irrig_col and soil_moist_col:
    mask = (df[irrig_col] > 5) & (df[soil_moist_col] > 85)
    if mask.any():
        count = int(mask.sum())
        logger.warning("Efficienza: %d irrigazioni inutili su suolo saturo.", count)
        self.diagnostics.irrigation_inefficient += count

# --- F. SALUTE SUOLO ---
if som_col:
    min_som = self.knowledge.get("general", {}).get("min_organic_matter", 1.5)
    mask = df[som_col] < min_som
    if mask.any():
        count = int(mask.sum())
        logger.warning("Salute: %d lotti con sostanza organica degradata (<%s%%).", count, min_som)
        self.diagnostics.soil_organic_low += count
```

For per-crop rules (inside the `for crop_name, rules in self.knowledge["crops"].items():` loop):

```python
# Regola dei Tre 10 (Peronospora Vite)
if crop_name == "wine_grape_docg" and temp_col and rain_col:
    inf_mask = mask & (df[temp_col] > rules.get("rule_10_temp", 10)) & (df[rain_col] > rules.get("rule_10_rain", 10))
    if inf_mask.any():
        count = int(inf_mask.sum())
        logger.warning("Malattia [It-Vite]: %d eventi a rischio Peronospora.", count)
        self.diagnostics.peronospora_events += count

# Colpo di calore in fioritura
if date_col and temp_col and "flowering_months" in rules:
    m = pd.to_datetime(df[date_col]).dt.month
    ct = rules.get("critical_temp_flowering", 35)
    h_mask = mask & m.isin(rules["flowering_months"]) & (df[temp_col] > ct)
    if h_mask.any():
        count = int(h_mask.sum())
        logger.warning("Stress [It-%s]: %d colpi di calore in fioritura.", crop_name, count)
        self.diagnostics.heat_stress_flowering += count

# Gelo tardivo
if date_col and temp_col and "frost_danger_months" in rules:
    m = pd.to_datetime(df[date_col]).dt.month
    f_mask = mask & m.isin(rules["frost_danger_months"]) & (df[temp_col] < 0)
    if f_mask.any():
        count = int(f_mask.sum())
        logger.warning("Gelo [It-%s]: %d gelate tardive rilevate.", crop_name, count)
        self.diagnostics.late_frost_events += count
```

Also in `_apply_physical_bounds`:

```python
def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in self.config.physical_bounds.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                count = int(mask.sum())
                logger.warning("%s: %d fuori range fisico [%s, %s] → NaN", col, count, lo, hi)
                df.loc[mask, col] = np.nan
                self.diagnostics.out_of_bounds_removed += count
    return df
```

And in `_handle_outliers`:

```python
def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    if self.config.outlier_method == "none": return df
    for col in self.config.numeric_columns:
        if col not in df.columns: continue
        s = df[col]
        if self.config.outlier_method == "iqr":
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - self.config.outlier_iqr_multiplier * iqr, q3 + self.config.outlier_iqr_multiplier * iqr
        else:
            mu, sigma = s.mean(), s.std()
            lo, hi = mu - 3 * sigma, mu + 3 * sigma
        mask = (s < lo) | (s > hi)
        if mask.any():
            count = int(mask.sum())
            logger.info("%s: %d outlier → NaN", col, count)
            df.loc[mask, col] = np.nan
            self.diagnostics.outliers_removed += count
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cleaner_diagnostics.py -v`
Expected: all tests PASS.

Run: `pytest -x`
Expected: all existing tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agripipe/cleaner.py tests/test_cleaner_diagnostics.py
git commit -m "feat(cleaner): count agronomic violations into diagnostics"
```

---

## Task 3: Add `"time"` imputation strategy with fallback

**Files:**
- Modify: `src/agripipe/cleaner.py` (Literal, `_impute_missing`)
- Test: `tests/test_cleaner_time_imputation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cleaner_time_imputation.py`:

```python
"""Test per la strategia di imputazione time-series."""

import numpy as np
import pandas as pd
import pytest

from agripipe.cleaner import AgriCleaner, CleanerConfig


def _ts_df_with_gap() -> pd.DataFrame:
    """Campo F1: temperatura 10 al giorno 1, NaN al giorno 2, 20 al giorno 3."""
    return pd.DataFrame({
        "date": pd.date_range("2024-03-01", periods=5, freq="D"),
        "field_id": ["F1"] * 5,
        "temp": [10.0, np.nan, 20.0, 22.0, 24.0],
        "humidity": [60.0, 65.0, np.nan, 70.0, 72.0],
    })


def test_time_imputation_interpolates_inside_field():
    config = CleanerConfig(
        numeric_columns=["temp", "humidity"],
        date_columns=["date"],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(_ts_df_with_gap())
    # Il NaN del giorno 2 (temp) dovrebbe essere ~15 (interpolazione lineare temporale)
    assert df_clean.loc[df_clean["date"] == "2024-03-02", "temp"].iloc[0] == pytest.approx(15.0, abs=0.5)
    assert cleaner.diagnostics.imputation_strategy_used == "time"


def test_time_imputation_falls_back_to_median_when_no_date(caplog):
    df = pd.DataFrame({
        "field_id": ["F1"] * 5,
        "temp": [10.0, np.nan, 20.0, 22.0, 24.0],
    })
    config = CleanerConfig(
        numeric_columns=["temp"],
        date_columns=[],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    import logging
    with caplog.at_level(logging.WARNING):
        df_clean = cleaner.clean(df)
    # Fallback: nessun NaN residuo
    assert df_clean["temp"].isna().sum() == 0
    assert cleaner.diagnostics.imputation_strategy_used == "median"
    assert any("fallback" in r.message.lower() or "median" in r.message.lower() for r in caplog.records)


def test_time_imputation_respects_field_boundaries():
    """Il NaN di F1 non deve essere riempito con valori di F2."""
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-03-01", "2024-03-02", "2024-03-03",   # F1
            "2024-03-01", "2024-03-02", "2024-03-03",   # F2
        ]),
        "field_id": ["F1", "F1", "F1", "F2", "F2", "F2"],
        "temp": [10.0, np.nan, 20.0, 100.0, 105.0, 110.0],
    })
    config = CleanerConfig(
        numeric_columns=["temp"],
        date_columns=["date"],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    f1_middle = df_clean[(df_clean["field_id"] == "F1") & (df_clean["date"] == "2024-03-02")]["temp"].iloc[0]
    # Deve essere ~15 (media di 10 e 20), non ~100 (valori di F2)
    assert f1_middle == pytest.approx(15.0, abs=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cleaner_time_imputation.py -v`
Expected: FAIL (strategy "time" not recognized or not implemented).

- [ ] **Step 3: Update the `ImputationStrategy` type**

In `src/agripipe/cleaner.py` around line 17:

```python
ImputationStrategy = Literal["mean", "median", "ffill", "drop", "time"]
```

- [ ] **Step 4: Implement the time imputation in `_impute_missing`**

Replace the entire `_impute_missing` method in `src/agripipe/cleaner.py`:

```python
def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
    """Imputa i valori mancanti secondo la strategia configurata.
    
    Per ``strategy="time"`` richiede una colonna ``date``. Se manca, effettua
    fallback a ``median`` con log di warning (scelta di design D1: robustezza
    sopra la rigorosità).
    """
    strat = self.config.missing_strategy
    
    if strat == "time":
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if not date_col or len(df) < 3:
            logger.warning(
                "Strategia 'time' richiede colonna date e >=3 righe; "
                "fallback automatico a 'median'."
            )
            strat = "median"
            self.diagnostics.imputation_strategy_used = "median"
        else:
            return self._impute_time(df, date_col)
    
    self.diagnostics.imputation_strategy_used = strat
    
    if strat == "drop":
        return df.dropna()
    
    before_na = int(df[self.config.numeric_columns].isna().sum().sum()) \
        if self.config.numeric_columns else 0
    for col in self.config.numeric_columns:
        if col not in df.columns: continue
        if strat == "mean": df[col] = df[col].fillna(df[col].mean())
        elif strat == "median": df[col] = df[col].fillna(df[col].median())
        elif strat == "ffill": df[col] = df[col].ffill().bfill()
    after_na = int(df[self.config.numeric_columns].isna().sum().sum()) \
        if self.config.numeric_columns else 0
    self.diagnostics.values_imputed += (before_na - after_na)
    return df


def _impute_time(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Interpolazione temporale per-campo.
    
    Ordina per (field, date), interpola ``method="time"`` con ``limit=3``,
    chiude i bordi con ``ffill().bfill()``. Raggruppa per ``field_id`` per
    non mischiare campi diversi.
    """
    field_col = next((c for c in ["field_id", "campo", "lotto"] if c in df.columns), None)
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    sort_cols = [field_col, date_col] if field_col else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    before_na = int(df[self.config.numeric_columns].isna().sum().sum())
    
    original_index = df.index
    df_indexed = df.set_index(date_col)
    
    for col in self.config.numeric_columns:
        if col not in df_indexed.columns: continue
        if field_col:
            df_indexed[col] = df_indexed.groupby(field_col)[col].transform(
                lambda s: s.interpolate(method="time", limit=3).ffill().bfill()
            )
        else:
            df_indexed[col] = df_indexed[col].interpolate(method="time", limit=3).ffill().bfill()
    
    df = df_indexed.reset_index()
    df.index = original_index
    
    after_na = int(df[self.config.numeric_columns].isna().sum().sum())
    self.diagnostics.values_imputed += (before_na - after_na)
    self.diagnostics.imputation_strategy_used = "time"
    
    if not field_col:
        logger.warning("Imputazione 'time' senza field_id: interpolazione globale.")
    
    return df
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cleaner_time_imputation.py -v`
Expected: 3 tests PASS.

Run: `pytest -x`
Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agripipe/cleaner.py tests/test_cleaner_time_imputation.py
git commit -m "feat(cleaner): add time-series imputation strategy with safe fallback"
```

---

## Task 4: Create `sustainability.py` with Badge + compute_scorecard

**Files:**
- Create: `src/agripipe/sustainability.py`
- Test: `tests/test_sustainability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sustainability.py`:

```python
"""Test delle soglie semaforo per la Sustainability Score Card."""

import pytest

from agripipe.cleaner import CleanerDiagnostics
from agripipe.sustainability import Badge, compute_scorecard, overall_message


def _diag(**overrides) -> CleanerDiagnostics:
    d = CleanerDiagnostics(total_rows=100)
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def test_nitrogen_green_at_zero_violations():
    badges = compute_scorecard(_diag(nitrogen_violations=0), total_rows=100)
    assert badges["nitrogen"].color == "green"


def test_nitrogen_yellow_under_five_percent():
    badges = compute_scorecard(_diag(nitrogen_violations=3), total_rows=100)
    assert badges["nitrogen"].color == "yellow"


def test_nitrogen_red_over_five_percent():
    badges = compute_scorecard(_diag(nitrogen_violations=10), total_rows=100)
    assert badges["nitrogen"].color == "red"


def test_peronospora_thresholds():
    assert compute_scorecard(_diag(peronospora_events=0), 100)["peronospora"].color == "green"
    assert compute_scorecard(_diag(peronospora_events=2), 100)["peronospora"].color == "yellow"
    assert compute_scorecard(_diag(peronospora_events=5), 100)["peronospora"].color == "red"


def test_irrigation_thresholds():
    assert compute_scorecard(_diag(irrigation_inefficient=0), 100)["irrigation"].color == "green"
    assert compute_scorecard(_diag(irrigation_inefficient=5), 100)["irrigation"].color == "yellow"
    assert compute_scorecard(_diag(irrigation_inefficient=20), 100)["irrigation"].color == "red"


def test_soil_thresholds():
    assert compute_scorecard(_diag(soil_organic_low=0), 100)["soil"].color == "green"
    assert compute_scorecard(_diag(soil_organic_low=10), 100)["soil"].color == "yellow"
    assert compute_scorecard(_diag(soil_organic_low=20), 100)["soil"].color == "red"


def test_scorecard_returns_four_badges_in_fixed_order():
    badges = compute_scorecard(_diag(), total_rows=100)
    assert list(badges.keys()) == ["nitrogen", "peronospora", "irrigation", "soil"]
    for b in badges.values():
        assert isinstance(b, Badge)


def test_overall_message_four_greens():
    badges = compute_scorecard(_diag(), total_rows=100)
    msg = overall_message(badges)
    assert "esemplare" in msg.lower()


def test_overall_message_with_one_red():
    badges = compute_scorecard(_diag(peronospora_events=5), total_rows=100)
    msg = overall_message(badges)
    assert any(word in msg.lower() for word in ["buona", "margine", "rivedi"])


def test_total_rows_zero_does_not_crash():
    """Edge case: dataframe vuoto non deve causare ZeroDivisionError."""
    badges = compute_scorecard(_diag(total_rows=0), total_rows=0)
    assert badges["nitrogen"].color in ("green", "yellow", "red")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sustainability.py -v`
Expected: FAIL with `ModuleNotFoundError: agripipe.sustainability`.

- [ ] **Step 3: Implement `sustainability.py`**

Create `src/agripipe/sustainability.py`:

```python
"""Sustainability Score Card: traduce CleanerDiagnostics in badge semaforici.

Modulo puro (zero I/O, zero Streamlit). Consumato dalla UI e dal metadata.
Le soglie sono scelte pragmatiche (spec D2): 0% è raramente raggiungibile su
dati reali, quindi il giallo rappresenta "accettabile con margine".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agripipe.cleaner import CleanerDiagnostics

BadgeColor = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class Badge:
    """Badge di sostenibilità per la Score Card.
    
    Attributes:
        name: Titolo visibile (es. "Azoto").
        icon: Emoji Unicode (es. "💧").
        color: Stato semaforo ("green" | "yellow" | "red").
        headline: Frase breve con il numero della violazione.
        tip: Consiglio agronomico in 1 riga.
    """
    name: str
    icon: str
    color: BadgeColor
    headline: str
    tip: str


def _pct(count: int, total: int) -> float:
    return (count / total * 100) if total > 0 else 0.0


def _nitrogen_badge(d: CleanerDiagnostics, total: int) -> Badge:
    pct = _pct(d.nitrogen_violations, total)
    if pct == 0:
        return Badge("Azoto", "💧", "green",
                    "0% concimazioni fuori standard",
                    "Direttiva Nitrati rispettata.")
    if pct <= 5:
        return Badge("Azoto", "💧", "yellow",
                    f"{pct:.1f}% concimazioni su suolo secco",
                    "Sincronizza la concimazione con la pioggia.")
    return Badge("Azoto", "💧", "red",
                f"{pct:.1f}% concimazioni fuori standard",
                "Rivedi il piano di fertilizzazione: rischio perdite.")


def _peronospora_badge(d: CleanerDiagnostics, total: int) -> Badge:
    events = d.peronospora_events
    if events == 0:
        return Badge("Peronospora", "🍇", "green",
                    "Nessun evento Tre 10",
                    "Condizioni climatiche favorevoli.")
    if events <= 3:
        return Badge("Peronospora", "🍇", "yellow",
                    f"{events} eventi Tre 10 rilevati",
                    "Monitora il microclima, valuta trattamenti mirati.")
    return Badge("Peronospora", "🍇", "red",
                f"{events} eventi Tre 10 critici",
                "Alto rischio: trattamento antifungino consigliato.")


def _irrigation_badge(d: CleanerDiagnostics, total: int) -> Badge:
    pct = _pct(d.irrigation_inefficient, total)
    if pct == 0:
        return Badge("Irrigazione", "🚿", "green",
                    "100% irrigazioni efficienti",
                    "Gestione idrica ottimale.")
    if pct <= 10:
        return Badge("Irrigazione", "🚿", "yellow",
                    f"{pct:.1f}% irrigazioni su suolo saturo",
                    "Verifica i sensori di umidità del suolo.")
    return Badge("Irrigazione", "🚿", "red",
                f"{pct:.1f}% irrigazioni sprecate",
                "Spreco idrico significativo: rivedi la programmazione.")


def _soil_badge(d: CleanerDiagnostics, total: int) -> Badge:
    pct = _pct(d.soil_organic_low, total)
    if pct == 0:
        return Badge("Suolo", "🌰", "green",
                    "Sostanza organica in range",
                    "Suolo vitale e strutturato.")
    if pct <= 15:
        return Badge("Suolo", "🌰", "yellow",
                    f"{pct:.1f}% lotti con SOM bassa",
                    "Valuta sovescio e ammendanti organici.")
    return Badge("Suolo", "🌰", "red",
                f"{pct:.1f}% lotti in degrado",
                "Rischio desertificazione: intervento urgente.")


def compute_scorecard(
    diagnostics: CleanerDiagnostics,
    total_rows: int,
) -> dict[str, Badge]:
    """Calcola i 4 badge di sostenibilità dal conteggio violazioni.
    
    Args:
        diagnostics: Popolato da ``AgriCleaner.clean()``.
        total_rows: Righe del DataFrame pulito, per calcolare le percentuali.
    
    Returns:
        Dizionario ordinato ``{"nitrogen", "peronospora", "irrigation", "soil"}``.
        L'ordine è stabile per rendering nella griglia 2×2.
    
    Example:
        >>> cleaner = AgriCleaner(config); cleaner.clean(df)
        >>> badges = compute_scorecard(cleaner.diagnostics, len(df))
        >>> badges["nitrogen"].color
        'green'
    """
    return {
        "nitrogen":    _nitrogen_badge(diagnostics, total_rows),
        "peronospora": _peronospora_badge(diagnostics, total_rows),
        "irrigation":  _irrigation_badge(diagnostics, total_rows),
        "soil":        _soil_badge(diagnostics, total_rows),
    }


def overall_message(badges: dict[str, Badge]) -> str:
    """Messaggio di sintesi dinamico in base al numero di badge verdi.
    
    Args:
        badges: Output di ``compute_scorecard``.
    
    Returns:
        Frase unica pronta da mostrare sotto la griglia dei badge.
    """
    greens = sum(1 for b in badges.values() if b.color == "green")
    if greens == 4:
        return "🌱 Gestione esemplare: pratiche agricole pienamente sostenibili."
    if greens >= 2:
        return "👍 Buona gestione con margine di miglioramento su alcune aree."
    return "⚠️ Rivedi le aree critiche per allinearti agli standard di sostenibilità."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sustainability.py -v`
Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agripipe/sustainability.py tests/test_sustainability.py
git commit -m "feat: add sustainability scorecard (4 badges, pure function)"
```

---

## Task 5: Create `metadata.py` (build + save metadata.json)

**Files:**
- Create: `src/agripipe/metadata.py`
- Test: `tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metadata.py`:

```python
"""Test della costruzione e serializzazione del metadata.json."""

import json
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.dataset import AgriDataset
from agripipe.metadata import build_metadata, save_metadata_json


def _prepare_dataset():
    df = pd.DataFrame({
        "temp": [20.0, 22.0, 24.0],
        "humidity": [60.0, 65.0, 70.0],
        "yield": [5.0, 6.0, 7.0],
    })
    ds = AgriDataset(df=df, numeric_columns=["temp", "humidity"], target="yield")
    return df, ds


def test_build_metadata_has_schema_version_and_timestamp():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    cleaner_diag_dict = {"values_imputed": 0, "outliers_removed": 0}
    meta = build_metadata(ds, preset, cleaner_diag_dict, target="yield")
    assert meta["schema_version"] == 1
    assert "generated_at" in meta
    assert meta["dataset_info"]["target"] == "yield"


def test_build_metadata_describes_all_columns():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    meta = build_metadata(ds, preset, {}, target="yield")
    col_names = [c["name"] for c in meta["columns"]]
    assert "temp" in col_names
    assert "humidity" in col_names
    for col in meta["columns"]:
        assert "index" in col
        assert "description" in col
        assert "normalized" in col


def test_save_metadata_json_writes_valid_json(tmp_path: Path):
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    meta = build_metadata(ds, preset, {}, target="yield")
    out = tmp_path / "metadata.json"
    save_metadata_json(meta, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1


def test_metadata_includes_agronomic_context():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    meta = build_metadata(ds, preset, {}, target="yield")
    assert meta["agronomic_context"]["crop"] == "olive"
    assert meta["agronomic_context"]["region"] == "Puglia"


def test_metadata_has_pytorch_example_code():
    df, ds = _prepare_dataset()
    meta = build_metadata(ds, {}, {}, target="yield")
    assert "example_code" in meta["pytorch_usage"]
    assert "torch.load" in meta["pytorch_usage"]["example_code"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: agripipe.metadata`.

- [ ] **Step 3: Implement `metadata.py`**

Create `src/agripipe/metadata.py`:

```python
"""Costruzione e salvataggio del file metadata.json che accompagna il tensor .pt.

Serve da "manuale d'uso" del dataset per il team Data Science di X Farm:
spiega ogni colonna, il contesto agronomico, e mostra un esempio PyTorch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agripipe.dataset import AgriDataset

SCHEMA_VERSION = 1

_COLUMN_DESCRIPTIONS = {
    "temp":        ("°C",       "Temperatura media giornaliera"),
    "temperatura": ("°C",       "Temperatura media giornaliera"),
    "humidity":    ("%",        "Umidità relativa aria"),
    "umidità":     ("%",        "Umidità relativa aria"),
    "rainfall":    ("mm",       "Precipitazione giornaliera"),
    "pioggia":     ("mm",       "Precipitazione giornaliera"),
    "ph":          ("pH",       "Acidità del suolo"),
    "yield":       ("t/ha",     "Resa colturale"),
    "resa":        ("t/ha",     "Resa colturale"),
    "n":           ("kg/ha",    "Concimazione azotata"),
    "azoto":       ("kg/ha",    "Concimazione azotata"),
    "soil_moisture": ("%",      "Umidità del suolo"),
    "irrigation":  ("mm",       "Irrigazione applicata"),
    "organic_matter": ("%",     "Sostanza organica del suolo"),
    "gdd_daily":       ("°C·d", "Gradi Giorno giornalieri"),
    "gdd_accumulated": ("°C·d", "Gradi Giorno cumulati"),
    "huglin_index":    ("idx",  "Indice di Huglin (qualità vitivinicola)"),
    "huglin_daily":    ("idx",  "Contributo Huglin giornaliero"),
    "daily_wb":        ("mm",   "Bilancio idrico giornaliero"),
    "drought_7d_score":("mm",   "Indice di siccità cumulata a 7 giorni"),
    "n_efficiency":    ("t/kg", "Efficienza dell'azoto (NUE)"),
}


def _describe_column(name: str, index: int) -> dict:
    unit, description = _COLUMN_DESCRIPTIONS.get(
        name.lower(), ("",  f"Colonna {name}")
    )
    return {
        "name": name,
        "index": index,
        "unit": unit,
        "description": description,
        "normalized": True,  # Tensorizer applica StandardScaler di default
    }


def build_metadata(
    dataset: AgriDataset,
    preset: dict,
    cleaner_diagnostics: dict,
    target: str | None = None,
    name: str = "agripipe_export",
) -> dict:
    """Costruisce il dizionario metadata dal dataset e dal contesto agronomico.
    
    Args:
        dataset: ``AgriDataset`` già fit (features + feature_names).
        preset: Entry di ``regional_presets`` del YAML (region, crop, zona, ...).
        cleaner_diagnostics: ``dataclasses.asdict(cleaner.diagnostics)``.
        target: Nome della colonna target (es. "yield").
        name: Identificatore del dataset (diventa ``dataset_info.name``).
    
    Returns:
        Dizionario pronto per JSON con: schema_version, generated_at,
        dataset_info, columns, agronomic_context, cleaning_stats,
        pytorch_usage.
    """
    n_rows = dataset.features.shape[0]
    n_features = dataset.features.shape[1]
    
    columns = [_describe_column(n, i) for i, n in enumerate(dataset.feature_names)]
    
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_info": {
            "name": name,
            "rows": int(n_rows),
            "features": int(n_features),
            "target": target,
            "target_unit": _COLUMN_DESCRIPTIONS.get((target or "").lower(), ("", ""))[0],
            "task": "regression" if target else "unsupervised",
        },
        "columns": columns,
        "agronomic_context": {
            "crop": preset.get("crop", "unknown"),
            "crop_display": preset.get("crop_display", ""),
            "region": preset.get("region", "unknown"),
            "zona": preset.get("zona", ""),
            "cleaning_rules": [
                "Direttiva Nitrati (soglia azoto)",
                "Regola Tre 10 (peronospora vite)",
                "Coerenza pioggia/umidità",
                "Imputazione time-series",
            ],
        },
        "cleaning_stats": cleaner_diagnostics,
        "pytorch_usage": {
            "example_code": (
                "import torch\n"
                "from torch.utils.data import TensorDataset, DataLoader\n\n"
                "bundle = torch.load('agripipe_export.pt')\n"
                "features, target = bundle['features'], bundle['target']\n"
                "loader = DataLoader(TensorDataset(features, target), batch_size=32, shuffle=True)\n"
                "# features è già normalizzato (StandardScaler): passa direttamente alla rete."
            ),
        },
    }


def save_metadata_json(metadata: dict, path: str | Path) -> Path:
    """Scrive il dict metadata su disco come JSON UTF-8 indentato.
    
    Args:
        metadata: Output di ``build_metadata``.
        path: Destinazione (cartelle create se mancanti).
    
    Returns:
        Path al file scritto.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metadata.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agripipe/metadata.py tests/test_metadata.py
git commit -m "feat: add metadata.json builder for ML bundle self-documentation"
```

---

## Task 6: Create `export.py` (bundle `.pt` + `.json` + `.zip`)

**Files:**
- Create: `src/agripipe/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_export.py`:

```python
"""Test del bundling ML-ready: .pt + metadata.json + .zip."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.export import export_ml_bundle


def _clean_df_and_cleaner():
    df = pd.DataFrame({
        "temp": [20.0, 22.0, 24.0, 26.0, 28.0],
        "humidity": [60.0, 65.0, 70.0, 75.0, 80.0],
        "yield": [5.0, 6.0, 7.0, 8.0, 9.0],
    })
    config = CleanerConfig(
        numeric_columns=["temp", "humidity", "yield"],
        outlier_method="none",
        missing_strategy="median",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    return df_clean, cleaner


def test_export_creates_three_files(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    paths = export_ml_bundle(df, cleaner, preset, tmp_path, name="test")
    assert paths["pt"].exists()
    assert paths["json"].exists()
    assert paths["zip"].exists()


def test_exported_pt_contains_all_bundle_keys(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {}, tmp_path, name="test")
    bundle = torch.load(paths["pt"], weights_only=False)
    assert "features" in bundle
    assert "feature_names" in bundle
    assert "scaler_mean" in bundle
    assert "scaler_scale" in bundle
    assert isinstance(bundle["features"], torch.Tensor)


def test_exported_zip_contains_pt_and_json(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {}, tmp_path, name="test")
    with zipfile.ZipFile(paths["zip"]) as z:
        names = z.namelist()
    assert any(n.endswith(".pt") for n in names)
    assert any(n.endswith(".json") for n in names)


def test_exported_metadata_is_valid_json(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {"region": "X", "crop": "y"}, tmp_path, name="t")
    meta = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["dataset_info"]["rows"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: agripipe.export`.

- [ ] **Step 3: Implement `export.py`**

Create `src/agripipe/export.py`:

```python
"""Orchestrazione dell'esportazione ML-ready: .pt + metadata.json + .zip.

Questo modulo è la "uscita" di AgriPipe verso il team Data Science di X Farm.
Produce un bundle completo: features normalizzate, target, nomi colonne,
parametri dello scaler, più un manuale d'uso in JSON.
"""

from __future__ import annotations

import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.metadata import build_metadata, save_metadata_json


def export_ml_bundle(
    df_clean: pd.DataFrame,
    cleaner: AgriCleaner,
    preset: dict,
    output_dir: str | Path,
    name: str = "agripipe_export",
    target: str = "yield",
) -> dict[str, Path]:
    """Esporta un bundle completo per training PyTorch.
    
    Crea nella ``output_dir``:
        - ``{name}.pt``   : bundle tensoriale (features, target, feature_names, scaler).
        - ``{name}.json`` : metadata auto-documentato (build_metadata output).
        - ``{name}.zip``  : zip di entrambi, per download singolo dall'UI.
    
    Args:
        df_clean: DataFrame già pulito da ``AgriCleaner.clean``.
        cleaner: Istanza usata per la pulizia (per accedere a diagnostics).
        preset: Entry di ``regional_presets`` selezionata dall'utente.
        output_dir: Cartella destinazione (creata se mancante).
        name: Prefisso per i file generati.
        target: Colonna target (passa ``None`` per bundle unsupervised).
    
    Returns:
        Dict con i Path dei 3 file: ``{"pt", "json", "zip"}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = [c for c in df_clean.select_dtypes(include=["number"]).columns
                    if c != target]
    target_col = target if target and target in df_clean.columns else None
    
    ds = AgriDataset(
        df=df_clean,
        numeric_columns=numeric_cols,
        target=target_col,
    )
    
    pt_path = output_dir / f"{name}.pt"
    json_path = output_dir / f"{name}.json"
    zip_path = output_dir / f"{name}.zip"
    
    bundle = {
        "features": ds.features,
        "target": ds.target,
        "feature_names": ds.feature_names,
        "scaler_mean": torch.tensor(ds.tensorizer.scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(ds.tensorizer.scaler.scale_, dtype=torch.float32),
    }
    torch.save(bundle, pt_path)
    
    metadata = build_metadata(
        dataset=ds,
        preset=preset,
        cleaner_diagnostics=asdict(cleaner.diagnostics),
        target=target_col,
        name=name,
    )
    save_metadata_json(metadata, json_path)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pt_path, arcname=pt_path.name)
        zf.write(json_path, arcname=json_path.name)
    
    return {"pt": pt_path, "json": json_path, "zip": zip_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agripipe/export.py tests/test_export.py
git commit -m "feat: add ML bundle exporter (.pt + metadata.json + .zip)"
```

---

## Task 7: Expand `agri_knowledge.yaml` — add `region` and `crop_display` to existing 5 presets

**Files:**
- Modify: `configs/agri_knowledge.yaml`

- [ ] **Step 1: Read current YAML structure**

Run: `cat configs/agri_knowledge.yaml | head -80`
Expected: See 5 existing presets under `regional_presets`.

- [ ] **Step 2: Add `region` and `crop_display` to each existing preset**

In `configs/agri_knowledge.yaml`, update each existing entry. For `ulivo_ligure`:

```yaml
  ulivo_ligure:
    region: "Liguria"
    crop: "olive"
    crop_display: "Olivo DOP Taggiasca"
    zona: "Liguria (Riviera)"
    suolo_tessitura: "Medio impasto / Scheletro"
    max_yield: 5.0
    temp_range: [-5, 38]
    ideal_ph: [6.5, 7.5]
    note: "Focus su terreni terrazzati e bassa meccanizzazione."
```

For `ulivo_pugliese`:

```yaml
  ulivo_pugliese:
    region: "Puglia"
    crop: "olive"
    crop_display: "Olivo Intensivo (Salento)"
    zona: "Puglia (Salento/Bari)"
    suolo_tessitura: "Terra rossa / Calcareo"
    max_yield: 12.0
    temp_range: [-2, 45]
    ideal_ph: [7.0, 8.5]
    salinity_tolerance: 4.0
    note: "Pianure assolate, alta tolleranza al calore."
```

For `grano_siciliano`:

```yaml
  grano_siciliano:
    region: "Sicilia"
    crop: "durum_wheat"
    crop_display: "Grano Duro Antico Siciliano"
    zona: "Sicilia (Entroterra)"
    suolo_tessitura: "Argilloso (Vertisuoli)"
    max_yield: 4.5
    temp_range: [-5, 45]
    harvest_months: [5, 6]
    ideal_ph: [7.5, 8.5]
    note: "Grano duro antico, suoli pesanti e fessurati in estate."
```

For `grano_emiliano`:

```yaml
  grano_emiliano:
    region: "Emilia-Romagna"
    crop: "soft_wheat"
    crop_display: "Grano Tenero Emiliano"
    zona: "Emilia-Romagna (Pianura Padana)"
    suolo_tessitura: "Limoso / Alluvionale"
    max_yield: 10.0
    temp_range: [-15, 35]
    harvest_months: [6, 7]
    ideal_ph: [6.0, 7.5]
    note: "Grano tenero per panificazione, suoli fertili e freschi."
```

For `vite_piemontese`:

```yaml
  vite_piemontese:
    region: "Piemonte"
    crop: "wine_grape_docg"
    crop_display: "Vite Nebbiolo (Barolo DOCG)"
    zona: "Langhe / Roero"
    suolo_tessitura: "Marne / Argille calcaree"
    max_yield: 8.0
    temp_range: [-15, 36]
    harvest_months: [9, 10]
    ideal_ph: [7.0, 8.0]
    note: "Grandi vini rossi (DOCG), focus su microclimi di collina."
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `pytest -x`
Expected: all tests PASS (YAML changes are additive).

- [ ] **Step 4: Commit**

```bash
git add configs/agri_knowledge.yaml
git commit -m "feat(config): add region and crop_display to existing presets"
```

---

## Task 8: Expand `agri_knowledge.yaml` — add 7 new presets + biological rules

**Files:**
- Modify: `configs/agri_knowledge.yaml`

- [ ] **Step 1: Append the 7 new presets under `regional_presets:`**

Add in `configs/agri_knowledge.yaml` (after `vite_piemontese`):

```yaml
  # --- LOMBARDIA ---
  riso_lombardo:
    region: "Lombardia"
    crop: "rice"
    crop_display: "Riso Carnaroli"
    zona: "Lomellina (Pavia)"
    suolo_tessitura: "Sommerso / Franco-limoso"
    max_yield: 7.0
    temp_range: [-8, 35]
    harvest_months: [9, 10]
    ideal_ph: [5.5, 7.0]
    note: "Risaie allagate, richiede gestione idrica precisa."

  # --- VENETO ---
  prosecco_veneto:
    region: "Veneto"
    crop: "wine_grape_docg"
    crop_display: "Vite Prosecco DOCG"
    zona: "Valdobbiadene / Conegliano"
    suolo_tessitura: "Vulcanico / Marnoso-arenaceo"
    max_yield: 13.5
    temp_range: [-10, 35]
    harvest_months: [8, 9]
    ideal_ph: [6.5, 7.5]
    note: "Colline UNESCO, microclimi ventilati."

  # --- TRENTINO-ALTO ADIGE ---
  mela_trentina:
    region: "Trentino-Alto Adige"
    crop: "apple"
    crop_display: "Mela Melinda DOP"
    zona: "Val di Non"
    suolo_tessitura: "Morenico / Ghiaioso"
    max_yield: 60.0
    temp_range: [-20, 32]
    harvest_months: [9, 10]
    ideal_ph: [6.0, 7.0]
    note: "Richiede fabbisogno freddo invernale (>1200 ore)."

  # --- EMILIA-ROMAGNA (secondo preset) ---
  pomodoro_emiliano:
    region: "Emilia-Romagna"
    crop: "tomato"
    crop_display: "Pomodoro da Industria"
    zona: "Parma / Piacenza"
    suolo_tessitura: "Alluvionale / Limoso"
    max_yield: 90.0
    temp_range: [-5, 38]
    harvest_months: [7, 8, 9]
    ideal_ph: [6.0, 7.0]
    note: "Pianura Padana irrigua, filiera industriale."

  # --- TOSCANA ---
  vite_toscana:
    region: "Toscana"
    crop: "wine_grape_docg"
    crop_display: "Vite Chianti DOCG (Sangiovese)"
    zona: "Chianti Classico"
    suolo_tessitura: "Galestro / Alberese"
    max_yield: 9.0
    temp_range: [-10, 38]
    harvest_months: [9, 10]
    ideal_ph: [6.5, 7.5]
    note: "Colline toscane, varietà Sangiovese."

  ulivo_toscano:
    region: "Toscana"
    crop: "olive"
    crop_display: "Olivo DOP Toscano"
    zona: "Colline Pisane/Senesi"
    suolo_tessitura: "Argilloso calcareo"
    max_yield: 4.5
    temp_range: [-8, 38]
    ideal_ph: [6.5, 7.5]
    note: "Olio IGP, cultivar Frantoio/Leccino/Moraiolo."

  # --- CAMPANIA ---
  pomodoro_sanmarzano:
    region: "Campania"
    crop: "tomato"
    crop_display: "Pomodoro San Marzano DOP"
    zona: "Agro Nocerino-Sarnese"
    suolo_tessitura: "Vulcanico (Vesuviano)"
    max_yield: 50.0
    temp_range: [0, 40]
    harvest_months: [7, 8, 9]
    ideal_ph: [6.0, 7.0]
    note: "Pomodoro lungo DOP, coltivazione tradizionale."
```

- [ ] **Step 2: Verify or add a `crops:` section with biological rules**

Check if a top-level `crops:` section exists in `configs/agri_knowledge.yaml` (read the full file). If not, add at the top-level (same indentation as `regional_presets:`):

```yaml
crops:
  olive:
    t_base: 10.0
    max_yield: 12.0
    flowering_months: [5, 6]
    frost_danger_months: [3, 4]
    critical_temp_flowering: 35
  
  durum_wheat:
    t_base: 0.0
    max_yield: 6.0
    flowering_months: [4, 5]
    frost_danger_months: [3]
    critical_temp_flowering: 32
  
  soft_wheat:
    t_base: 0.0
    max_yield: 10.0
    flowering_months: [5]
    frost_danger_months: [3, 4]
    critical_temp_flowering: 30
  
  wine_grape_docg:
    t_base: 10.0
    max_yield: 14.0
    flowering_months: [5, 6]
    frost_danger_months: [4]
    critical_temp_flowering: 38
    rule_10_temp: 10
    rule_10_rain: 10
  
  rice:
    t_base: 12.0
    max_yield: 9.0
    flowering_months: [7, 8]
    frost_danger_months: [4, 5]
    critical_temp_flowering: 35
  
  apple:
    t_base: 4.0
    max_yield: 70.0
    flowering_months: [4, 5]
    frost_danger_months: [4]
    critical_temp_flowering: 35
    chill_hours_required: 1200
  
  tomato:
    t_base: 10.0
    max_yield: 100.0
    flowering_months: [6, 7]
    frost_danger_months: [4, 5]
    critical_temp_flowering: 38

general:
  min_organic_matter: 1.5
```

- [ ] **Step 3: Ensure the YAML remains valid**

Run: `python -c "import yaml; yaml.safe_load(open('configs/agri_knowledge.yaml', encoding='utf-8'))"`
Expected: no output (no error).

- [ ] **Step 4: Add a test verifying all 12 presets load correctly**

Create `tests/test_presets_load.py`:

```python
"""Smoke test: tutti i preset territoriali sono validi e caricabili."""

from pathlib import Path

import yaml


def test_all_presets_have_required_fields():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    presets = data["regional_presets"]
    assert len(presets) >= 12
    for key, p in presets.items():
        assert "region" in p, f"{key}: missing region"
        assert "crop" in p, f"{key}: missing crop"
        assert "crop_display" in p, f"{key}: missing crop_display"
        assert "max_yield" in p, f"{key}: missing max_yield"
        assert "ideal_ph" in p, f"{key}: missing ideal_ph"


def test_presets_cover_at_least_ten_regions():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    regions = {p["region"] for p in data["regional_presets"].values()}
    assert len(regions) >= 10, f"Solo {len(regions)} regioni: {regions}"


def test_crops_biological_rules_exist():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    crops = data.get("crops", {})
    for crop_key in ["olive", "durum_wheat", "soft_wheat", "wine_grape_docg",
                     "rice", "apple", "tomato"]:
        assert crop_key in crops, f"Missing biological rules for {crop_key}"
        assert "t_base" in crops[crop_key]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_presets_load.py -v`
Expected: 3 tests PASS.

Run: `pytest -x`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/agri_knowledge.yaml tests/test_presets_load.py
git commit -m "feat(config): add 7 curated presets (10 regions, 12 total) + crop rules"
```

---

## Task 9: Create `ui/theme.py` — palette and CSS injector

**Files:**
- Create: `src/agripipe/ui/__init__.py`
- Create: `src/agripipe/ui/theme.py`
- Test: `tests/test_ui_theme.py`

- [ ] **Step 1: Announce frontend-design skill invocation**

Before implementing UI files, invoke the `frontend-design` skill to anchor design principles (typography scale, spacing rhythm, micro-interaction patterns). Apply its guidance to the CSS written below. **Do this via the Skill tool** before Step 2.

- [ ] **Step 2: Write the failing test**

Create `tests/test_ui_theme.py`:

```python
"""Test del modulo theme (palette + CSS)."""

from agripipe.ui import theme


def test_palette_has_all_required_colors():
    required = {"sage", "forest", "earth", "water", "cream", "card",
                "leaf", "wheat", "pomegranate", "text", "text_muted"}
    assert required.issubset(theme.PALETTE.keys())


def test_palette_values_are_valid_hex():
    import re
    hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for name, value in theme.PALETTE.items():
        assert hex_re.match(value), f"{name}={value} is not a valid hex color"


def test_badge_color_map_exists():
    assert theme.BADGE_COLORS["green"] == theme.PALETTE["leaf"]
    assert theme.BADGE_COLORS["yellow"] == theme.PALETTE["wheat"]
    assert theme.BADGE_COLORS["red"] == theme.PALETTE["pomegranate"]


def test_build_stylesheet_contains_all_colors():
    css = theme.build_stylesheet()
    for hex_value in theme.PALETTE.values():
        assert hex_value in css, f"Missing {hex_value} in CSS"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ui_theme.py -v`
Expected: FAIL with `ModuleNotFoundError: agripipe.ui`.

- [ ] **Step 4: Create package and theme module**

Create `src/agripipe/ui/__init__.py`:

```python
"""UI package: Streamlit theme + reusable components for AgriPipe."""
```

Create `src/agripipe/ui/theme.py`:

```python
"""Palette Clean & Nature e iniezione CSS in Streamlit.

Fonte unica di verità per i colori e lo stile di AgriPipe. Modificare un colore
qui = modifica propagata a tutta la UI. Chi cambia la palette NON deve toccare
components.py.
"""

from __future__ import annotations

PALETTE = {
    "sage":        "#7FA77F",
    "forest":      "#3D5A3D",
    "earth":       "#8B6F47",
    "water":       "#4A90A4",
    "cream":       "#FAFAF7",
    "card":        "#FFFFFF",
    "leaf":        "#6BAF6B",
    "wheat":       "#D4A64A",
    "pomegranate": "#B84A3E",
    "text":        "#2B2B2B",
    "text_muted":  "#6B6B6B",
}

BADGE_COLORS = {
    "green":  PALETTE["leaf"],
    "yellow": PALETTE["wheat"],
    "red":    PALETTE["pomegranate"],
}


def build_stylesheet() -> str:
    """Costruisce il foglio di stile CSS per Streamlit.
    
    Returns:
        Blocco CSS come stringa, pronto da passare a ``st.markdown``
        con ``unsafe_allow_html=True``.
    """
    p = PALETTE
    return f"""
    /* === AgriPipe Clean & Nature === */
    .stApp {{ background: {p["cream"]}; }}
    .block-container {{ max-width: 1200px; padding-top: 2rem; }}
    
    h1, h2, h3 {{ color: {p["forest"]}; font-weight: 600; }}
    p, span, li {{ color: {p["text"]}; line-height: 1.6; }}
    
    /* Step header */
    .agri-step {{
        display: flex; align-items: center; gap: 0.75rem;
        font-size: 1.4rem; font-weight: 600; color: {p["forest"]};
        border-left: 4px solid {p["sage"]};
        padding: 0.5rem 0 0.5rem 1rem;
        margin: 2rem 0 1rem 0;
    }}
    .agri-step-number {{
        background: {p["sage"]}; color: white; font-size: 0.9rem;
        padding: 0.15rem 0.55rem; border-radius: 999px;
    }}
    
    /* Card generica */
    .agri-card {{
        background: {p["card"]};
        border: 1px solid {p["earth"]}33;
        border-radius: 8px;
        padding: 1.25rem;
    }}
    
    /* Info card (riga 3 card) */
    .agri-info-card {{
        background: {p["card"]}; border: 1px solid {p["earth"]}22;
        border-radius: 8px; padding: 1rem; text-align: center;
    }}
    .agri-info-card .label {{ font-size: 0.8rem; color: {p["text_muted"]};
        text-transform: uppercase; letter-spacing: 0.5px; }}
    .agri-info-card .value {{ font-size: 1.25rem; font-weight: 600;
        color: {p["forest"]}; margin-top: 0.3rem; }}
    
    /* Motivational banner */
    .agri-motivation {{
        background: {p["sage"]}22; border-left: 4px solid {p["sage"]};
        padding: 1rem 1.25rem; border-radius: 6px; color: {p["forest"]};
        font-style: italic; margin: 1rem 0;
    }}
    
    /* Badge grid (Score Card) */
    .agri-badge-grid {{ display: grid; grid-template-columns: 1fr 1fr;
        gap: 1rem; margin: 1rem 0; }}
    .agri-badge {{
        background: {p["card"]}; border-radius: 8px; padding: 1.25rem;
        border-top: 4px solid var(--badge-color);
        border-right: 1px solid {p["earth"]}22;
        border-bottom: 1px solid {p["earth"]}22;
        border-left: 1px solid {p["earth"]}22;
    }}
    .agri-badge-header {{ display: flex; align-items: center;
        gap: 0.5rem; font-weight: 600; color: {p["forest"]}; }}
    .agri-badge-dot {{ display: inline-block; width: 12px; height: 12px;
        border-radius: 50%; background: var(--badge-color); }}
    .agri-badge-headline {{ font-size: 1.1rem; margin: 0.5rem 0;
        color: {p["text"]}; }}
    .agri-badge-tip {{ font-size: 0.9rem; color: {p["text_muted"]}; }}
    
    /* Overall sustainability message */
    .agri-overall {{
        text-align: center; font-size: 1.05rem; font-weight: 500;
        padding: 1rem; color: {p["forest"]};
    }}
    
    /* Hero */
    .agri-hero {{
        text-align: center; padding: 1.5rem 0;
        border-bottom: 1px solid {p["earth"]}22; margin-bottom: 1.5rem;
    }}
    .agri-hero h1 {{ font-size: 2.2rem; margin: 0; color: {p["forest"]}; }}
    .agri-hero .subtitle {{ color: {p["text_muted"]}; font-size: 1rem; }}
    
    /* Streamlit button override */
    .stButton > button {{
        background: {p["forest"]}; color: white; border-radius: 8px;
        border: none; font-weight: 600; padding: 0.6rem 1.5rem;
    }}
    .stButton > button:hover {{ background: {p["sage"]}; }}
    """


def inject_css() -> None:
    """Inietta il foglio di stile nella pagina Streamlit corrente.
    
    Chiamare UNA volta all'inizio di ``app.py`` dopo ``st.set_page_config``.
    """
    import streamlit as st
    st.markdown(f"<style>{build_stylesheet()}</style>", unsafe_allow_html=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ui_theme.py -v`
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agripipe/ui/__init__.py src/agripipe/ui/theme.py tests/test_ui_theme.py
git commit -m "feat(ui): add Clean & Nature theme (palette + CSS injector)"
```

---

## Task 10: Create `ui/components.py` — Part 1 (hero, step, motivation, info_cards)

**Files:**
- Create: `src/agripipe/ui/components.py`

Note: Streamlit components are hard to unit-test directly (they push HTML to a runtime). We rely on rendering smoke tests — importing and calling them without errors — and manual visual verification.

- [ ] **Step 1: Create the components module with the first set of renderers**

Create `src/agripipe/ui/components.py`:

```python
"""Componenti Streamlit riusabili per AgriPipe.

Ogni funzione emette HTML nel runtime Streamlit usando le classi CSS definite
in ``ui/theme.py``. Principio: NESSUNA logica di business qui, solo rendering.
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from agripipe.ui.theme import BADGE_COLORS


def render_hero(title: str = "🌱 AgriPipe",
                subtitle: str = "Da Excel sporco a dati ML-ready in 30 secondi") -> None:
    """Intestazione principale della pagina.
    
    Args:
        title: Titolo con emoji/icona.
        subtitle: Frase di valore mostrata sotto il titolo.
    """
    st.markdown(
        f"""
        <div class="agri-hero">
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step(number: int, icon: str, title: str) -> None:
    """Intestazione di uno step numerato (1..5) con icona.
    
    Args:
        number: Numero dello step (1..5).
        icon: Emoji agronomica (es. "📍", "🌾").
        title: Titolo dello step.
    """
    st.markdown(
        f"""
        <div class="agri-step">
            <span class="agri-step-number">Step {number}</span>
            <span>{icon} {title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_motivation(message: str) -> None:
    """Banner motivazionale Verde Salvia.
    
    Args:
        message: Frase unica che spiega il valore (es. risparmio tempo).
    """
    st.markdown(
        f'<div class="agri-motivation">💬 {message}</div>',
        unsafe_allow_html=True,
    )


def render_info_cards(cards: Iterable[tuple[str, str, str]]) -> None:
    """Riga di card informative equidistanziate.
    
    Args:
        cards: Iterable di tuple ``(icon, label, value)``. Tipicamente 3 card.
    """
    cards = list(cards)
    cols = st.columns(len(cards))
    for col, (icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="agri-info-card">
                    <div class="label">{icon} {label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
```

- [ ] **Step 2: Smoke test — verify imports resolve**

Run: `python -c "from agripipe.ui.components import render_hero, render_step, render_motivation, render_info_cards; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/agripipe/ui/components.py
git commit -m "feat(ui): add hero, step, motivation, info_cards components"
```

---

## Task 11: Extend `ui/components.py` — Part 2 (metrics, download_row)

**Files:**
- Modify: `src/agripipe/ui/components.py`

- [ ] **Step 1: Append new component functions**

Append to `src/agripipe/ui/components.py`:

```python
def render_metrics(total_input: int, total_output: int, anomalies: int) -> None:
    """Tre metriche principali dopo la pulizia.
    
    Args:
        total_input: Righe del dataframe grezzo.
        total_output: Righe del dataframe pulito.
        anomalies: Numero di anomalie corrette (da diagnostics).
    """
    c1, c2, c3 = st.columns(3)
    reliability = (total_output / total_input * 100) if total_input else 100.0
    c1.metric("Righe pulite", f"{total_output:,}")
    c2.metric("Anomalie corrette", f"{anomalies:,}")
    c3.metric("Affidabilità dato", f"{reliability:.1f}%")


def render_download_row(
    excel_bytes: bytes,
    bundle_zip_bytes: bytes,
    name_prefix: str,
) -> None:
    """Due bottoni download affiancati: Excel + Bundle ML (.zip).
    
    Args:
        excel_bytes: Contenuto binario del file Excel pulito.
        bundle_zip_bytes: Contenuto binario del file .zip (pt + json).
        name_prefix: Prefisso per i nomi file (es. "ulivo_pugliese").
    """
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📥 Excel Pulito",
            data=excel_bytes,
            file_name=f"agripipe_{name_prefix}_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Per agronomi e gestionali Excel.")
    with c2:
        st.download_button(
            label="💾 Bundle ML (.zip)",
            data=bundle_zip_bytes,
            file_name=f"agripipe_{name_prefix}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("Tensor PyTorch + metadata.json, pronti per training.")
```

- [ ] **Step 2: Smoke test — import check**

Run: `python -c "from agripipe.ui.components import render_metrics, render_download_row; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/agripipe/ui/components.py
git commit -m "feat(ui): add metrics and download_row components"
```

---

## Task 12: Extend `ui/components.py` — Part 3 (scorecard, before_after_plots)

**Files:**
- Modify: `src/agripipe/ui/components.py`

- [ ] **Step 1: Append scorecard and plots components**

Append to `src/agripipe/ui/components.py`:

```python
from agripipe.sustainability import Badge


def render_scorecard(badges: dict[str, Badge], overall: str) -> None:
    """Griglia 2×2 dei badge di sostenibilità + messaggio di sintesi.
    
    Args:
        badges: Output di ``sustainability.compute_scorecard``.
        overall: Output di ``sustainability.overall_message(badges)``.
    """
    keys = list(badges.keys())
    rows = [keys[0:2], keys[2:4]]
    for row_keys in rows:
        cols = st.columns(2)
        for col, key in zip(cols, row_keys):
            b = badges[key]
            color_hex = BADGE_COLORS[b.color]
            with col:
                st.markdown(
                    f"""
                    <div class="agri-badge" style="--badge-color: {color_hex}">
                        <div class="agri-badge-header">
                            <span class="agri-badge-dot"></span>
                            <span>{b.icon} {b.name}</span>
                        </div>
                        <div class="agri-badge-headline">{b.headline}</div>
                        <div class="agri-badge-tip">{b.tip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.markdown(f'<div class="agri-overall">{overall}</div>', unsafe_allow_html=True)


def render_before_after_plots(df_raw, df_clean) -> None:
    """Boxplot + KDE prima/dopo per ogni colonna numerica chiave.
    
    Usa la palette AgriPipe (Marrone Terra per grezzo, Verde Salvia per pulito).
    
    Args:
        df_raw: DataFrame grezzo originale.
        df_clean: DataFrame pulito dopo AgriCleaner.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    from agripipe.ui.theme import PALETTE
    
    numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    priority = [c for c in ["yield", "temp", "ph", "humidity"] if c in numeric_cols]
    
    for col in priority:
        if col not in df_clean.columns: continue
        with st.expander(f"📊 {col.upper()}", expanded=(col == "yield")):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            data_cmp = pd.concat([
                df_raw[[col]].assign(Stato="Grezzo"),
                df_clean[[col]].assign(Stato="Ottimizzato"),
            ])
            sns.boxplot(data=data_cmp, x="Stato", y=col, ax=ax1,
                        palette=[PALETTE["earth"], PALETTE["sage"]])
            ax1.set_title(f"Anomalie in {col}")
            sns.kdeplot(df_raw[col].dropna(), fill=True, label="Grezzo",
                        ax=ax2, color=PALETTE["earth"])
            sns.kdeplot(df_clean[col].dropna(), fill=True, label="Ottimizzato",
                        ax=ax2, color=PALETTE["sage"])
            ax2.set_title(f"Distribuzione di {col}")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
```

- [ ] **Step 2: Smoke test**

Run: `python -c "from agripipe.ui.components import render_scorecard, render_before_after_plots; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/agripipe/ui/components.py
git commit -m "feat(ui): add scorecard grid and before/after plots components"
```

---

## Task 13: Rewrite `app.py` — cascaded selector + full pipeline composition

**Files:**
- Rewrite: `src/agripipe/app.py`

- [ ] **Step 1: Replace `app.py` entirely**

Overwrite `src/agripipe/app.py`:

```python
"""AgriPipe Streamlit app — composizione di ui.theme + ui.components."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.export import export_ml_bundle
from agripipe.sustainability import compute_scorecard, overall_message
from agripipe.ui import components, theme


st.set_page_config(
    page_title="AgriPipe — Pro Edition",
    page_icon="🌱",
    layout="wide",
)
theme.inject_css()


@st.cache_data
def _load_knowledge() -> dict:
    return yaml.safe_load(
        Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8")
    )


def _read_uploaded(uploaded) -> pd.DataFrame:
    if uploaded.name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def _build_config(preset: dict, df: pd.DataFrame) -> CleanerConfig:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return CleanerConfig(
        numeric_columns=numeric,
        date_columns=[c for c in ["date", "data"] if c in df.columns],
        physical_bounds={
            "ph":    tuple(preset.get("ideal_ph",   [0.0, 14.0])),
            "yield": (0.0, float(preset.get("max_yield", 100.0))),
            "temp":  tuple(preset.get("temp_range", [-20.0, 50.0])),
        },
        missing_strategy="time",
        knowledge_path="configs/agri_knowledge.yaml",
    )


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Dati_Ottimizzati")
    return buf.getvalue()


knowledge = _load_knowledge()
presets = knowledge.get("regional_presets", {})

components.render_hero()

# --- STEP 1: Territorio (cascaded) ---
components.render_step(1, "📍", "Inquadramento Territoriale")

regions = sorted({p["region"] for p in presets.values()})
region = st.selectbox("🗺️ Regione", regions, index=0)

crops_in_region = {k: p for k, p in presets.items() if p["region"] == region}
selected_key = st.selectbox(
    "🌾 Coltura",
    options=list(crops_in_region.keys()),
    format_func=lambda k: crops_in_region[k]["crop_display"],
)
st.caption(f"🌿 {len(crops_in_region)} coltura/e disponibile/i in {region}.")

preset = presets[selected_key]
components.render_info_cards([
    ("🌰", "Suolo", preset["suolo_tessitura"]),
    ("🌾", "Resa max", f"{preset['max_yield']} t/ha"),
    ("🧪", "pH ideale", f"{preset['ideal_ph'][0]}–{preset['ideal_ph'][1]}"),
])
st.caption(f"ℹ️ **{preset.get('zona', '')}** — {preset['note']}")

# --- STEP 2: Upload ---
components.render_step(2, "🌾", "Dati del Campo")
components.render_motivation(
    "Carica il tuo Excel: in 30 secondi avrai dati pronti per l'IA. "
    "Tempo risparmiato: ~4 ore di pulizia manuale."
)
uploaded = st.file_uploader(
    "Trascina qui il file Excel o CSV", type=["xlsx", "csv"]
)

if uploaded is not None:
    st.success(f"✅ File per '{preset.get('zona', region)}' pronto per l'analisi.")
    
    if st.button("🚀 Avvia Ottimizzazione Personalizzata", use_container_width=True):
        with st.spinner("Applicando regole territoriali specifiche..."):
            df_raw = _read_uploaded(uploaded)
            cleaner = AgriCleaner(_build_config(preset, df_raw))
            df_clean = cleaner.clean(df_raw)
        
        # --- STEP 3: Risultati ---
        components.render_step(3, "🧪", "Risultati dell'Ottimizzazione")
        anomalies = (
            cleaner.diagnostics.outliers_removed
            + cleaner.diagnostics.out_of_bounds_removed
            + cleaner.diagnostics.values_imputed
        )
        components.render_metrics(len(df_raw), len(df_clean), anomalies)
        st.write("---")
        
        with tempfile.TemporaryDirectory() as tmp:
            bundle_paths = export_ml_bundle(
                df_clean, cleaner, preset, tmp, name=selected_key,
            )
            zip_bytes = bundle_paths["zip"].read_bytes()
        
        components.render_download_row(
            excel_bytes=_to_excel_bytes(df_clean),
            bundle_zip_bytes=zip_bytes,
            name_prefix=selected_key,
        )
        
        # --- STEP 4: Sustainability Score Card ---
        components.render_step(4, "🌱", "Sustainability Score Card")
        badges = compute_scorecard(cleaner.diagnostics, total_rows=len(df_clean))
        components.render_scorecard(badges, overall_message(badges))
        
        # --- STEP 5: Analisi Visiva ---
        components.render_step(5, "📊", "Analisi Visiva Prima/Dopo")
        components.render_before_after_plots(df_raw, df_clean)
        
        with st.expander("📋 Anteprima dati ottimizzati"):
            st.dataframe(df_clean.head(20))
```

- [ ] **Step 2: Smoke test — launch Streamlit and check the app loads**

Run: `streamlit run src/agripipe/app.py --server.headless true --server.port 8765 &`
Wait 3 seconds, then: `curl -s http://localhost:8765 | head -c 200`
Expected: HTML response containing `AgriPipe`.
Kill the process afterwards.

- [ ] **Step 3: Run full test suite**

Run: `pytest -x`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agripipe/app.py
git commit -m "feat(ui): rewrite app.py with cascaded selector + scorecard + ML bundle"
```

---

## Task 14: Update `cli.py` to save `metadata.json` next to `.pt`

**Files:**
- Modify: `src/agripipe/cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_e2e.py` (before the `print` at the end):

```python
def test_cli_run_produces_metadata_json(tmp_path: Path):
    """Il CLI deve produrre metadata.json accanto al .pt."""
    from typer.testing import CliRunner
    from agripipe.cli import app
    from agripipe.synth import SynthConfig, generate_dirty_excel
    
    input_file = tmp_path / "raw.xlsx"
    output_pt = tmp_path / "out.pt"
    config_file = tmp_path / "config.yaml"
    
    generate_dirty_excel(input_file, SynthConfig(n_rows=100, seed=1))
    config_file.write_text(
        "numeric_columns: [temp, humidity, ph, yield]\n"
        "categorical_columns: [field_id, crop_type]\n"
        "date_columns: [date]\n"
        "missing_strategy: median\n"
        "outlier_method: iqr\n"
        "physical_bounds:\n  ph: [0.0, 14.0]\n  humidity: [0.0, 100.0]\n",
        encoding="utf-8",
    )
    
    runner = CliRunner()
    result = runner.invoke(app, [
        "run", "-i", str(input_file), "-o", str(output_pt),
        "-c", str(config_file), "-t", "yield",
    ])
    assert result.exit_code == 0, result.output
    metadata_path = output_pt.with_suffix(".json")
    assert metadata_path.exists(), f"Expected {metadata_path}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e.py::test_cli_run_produces_metadata_json -v`
Expected: FAIL (metadata.json not yet written).

- [ ] **Step 3: Update `cli.py` run command**

Edit `src/agripipe/cli.py`, inside the `run` command. Replace the `torch.save(...)` block with:

```python
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": ds.features, "target": ds.target, "feature_names": ds.feature_names},
            output,
        )
        
        from agripipe.metadata import build_metadata, save_metadata_json
        from dataclasses import asdict
        metadata_path = output.with_suffix(".json")
        metadata = build_metadata(
            dataset=ds,
            preset={},
            cleaner_diagnostics=asdict(cleaner.diagnostics),
            target=target_col,
            name=output.stem,
        )
        save_metadata_json(metadata, metadata_path)
        
        typer.secho(
            f"✓ Tensor salvati in {output} (shape={tuple(ds.features.shape)})",
            fg=typer.colors.GREEN,
        )
        typer.secho(f"✓ Metadata in {metadata_path}", fg=typer.colors.GREEN)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e.py::test_cli_run_produces_metadata_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agripipe/cli.py tests/test_e2e.py
git commit -m "feat(cli): write metadata.json sidecar next to .pt output"
```

---

## Task 15: Add training-readiness E2E test

**Files:**
- Modify: `tests/test_e2e.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_e2e.py`:

```python
def test_exported_bundle_is_training_ready(tmp_path: Path, cleaner_config):
    """Il bundle esportato deve essere immediatamente utilizzabile per training."""
    from torch.utils.data import TensorDataset, DataLoader
    
    from agripipe.cleaner import AgriCleaner
    from agripipe.export import export_ml_bundle
    from agripipe.synth import SynthConfig, generate_dirty_excel
    
    # Setup: genera dati + pulisci
    input_file = tmp_path / "raw.xlsx"
    generate_dirty_excel(input_file, SynthConfig(n_rows=150, seed=7))
    df_raw = pd.read_excel(input_file)
    if "rainfall" not in cleaner_config.numeric_columns:
        cleaner_config.numeric_columns.append("rainfall")
    cleaner = AgriCleaner(cleaner_config)
    df_clean = cleaner.clean(df_raw)
    
    # Esporta
    preset = {"region": "Test", "crop": "wheat", "zona": "Test"}
    paths = export_ml_bundle(df_clean, cleaner, preset, tmp_path, name="train_ready")
    
    # Carica + simula training
    bundle = torch.load(paths["pt"], weights_only=False)
    features, target = bundle["features"], bundle["target"]
    assert features.isfinite().all(), "Features contiene NaN/Inf"
    assert target is not None and target.isfinite().all()
    
    loader = DataLoader(TensorDataset(features, target), batch_size=16, shuffle=True)
    model = torch.nn.Linear(features.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    
    batch_x, batch_y = next(iter(loader))
    pred = model(batch_x).squeeze(-1)
    loss = torch.nn.functional.mse_loss(pred, batch_y)
    loss.backward()
    opt.step()
    
    assert loss.isfinite(), "Loss NaN/Inf: bundle non utilizzabile per training"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_e2e.py::test_exported_bundle_is_training_ready -v`
Expected: PASS.

- [ ] **Step 3: Run full test suite as final gate**

Run: `pytest`
Expected: all tests PASS. Note the final coverage.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(e2e): verify exported bundle is PyTorch training-ready"
```

---

## Task 16: Add Google-style docstrings to existing modules

**Files:**
- Modify: `src/agripipe/indices.py`
- Modify: `src/agripipe/loader.py`
- Modify: `src/agripipe/tensorizer.py`
- Modify: `src/agripipe/dataset.py`

- [ ] **Step 1: Add docstrings to `indices.py`**

Replace the `compute_agronomic_indices` docstring in `src/agripipe/indices.py`:

```python
def compute_agronomic_indices(df: pd.DataFrame, knowledge: dict) -> pd.DataFrame:
    """Arricchisce il DataFrame con indici agronomici per il Machine Learning.
    
    Calcola (dove le colonne sono presenti):
        - **GDD daily/accumulated**: Gradi Giorno rispetto a ``t_base`` per coltura.
        - **Huglin index**: Indice qualità vitivinicola (solo vite).
        - **drought_7d_score**: Bilancio idrico mobile a 7 giorni.
        - **n_efficiency**: Nitrogen Use Efficiency (yield / N).
    
    Args:
        df: DataFrame pulito. Colonne richieste: ``crop_type``/``crop``,
            ``temp``/``temperatura``. Opzionali: ``date``, ``field_id``,
            ``rainfall``, ``yield``, ``n``.
        knowledge: Dizionario caricato da ``agri_knowledge.yaml``. La chiave
            ``crops`` contiene ``t_base`` per coltura.
    
    Returns:
        DataFrame con le colonne originali + gli indici calcolati. Se mancano
        colonne essenziali (crop + temp) restituisce il df invariato.
    
    Example:
        >>> df_with_indices = compute_agronomic_indices(df, knowledge)
        >>> "gdd_accumulated" in df_with_indices.columns
        True
    """
```

- [ ] **Step 2: Add docstring to `loader.py`**

The existing `load_raw` in `src/agripipe/loader.py` already has a good docstring. Verify it follows Google style (Args/Returns/Raises blocks). If not, refactor to match.

- [ ] **Step 3: Add docstrings to `tensorizer.py`**

Update `src/agripipe/tensorizer.py`. Replace the class docstring:

```python
class Tensorizer:
    """Converte un DataFrame pulito in tensor PyTorch normalizzati.
    
    Normalizza le colonne numeriche con ``StandardScaler`` (media 0, dev std 1)
    e codifica le colonne categoriche con ``LabelEncoder``. Serializzabile per
    inferenza futura.
    
    Attributes:
        numeric_columns: Colonne continue da normalizzare.
        categorical_columns: Colonne categoriche da codificare.
        target: Colonna target (opzionale, esclusa dalle features).
        target_dtype: Tipo del tensor target (``"float32"`` | ``"long"``).
        scaler: ``StandardScaler`` fit sui numeric_columns.
        encoders: Dict ``{col_name: LabelEncoder}`` per le categoriche.
    """
```

And the `transform` method:

```python
def transform(self, df: pd.DataFrame) -> TensorBundle:
    """Applica la trasformazione già fit a un nuovo DataFrame.
    
    Args:
        df: DataFrame con le stesse colonne usate in ``fit_transform``.
    
    Returns:
        ``TensorBundle`` con ``features``, ``target`` (o None) e
        ``feature_names``.
    
    Raises:
        RuntimeError: Se ``_fit`` non è ancora stato chiamato.
        ValueError: Se il tensor risultante contiene NaN/Inf.
    """
```

- [ ] **Step 4: Add docstrings to `dataset.py`**

Replace the `AgriDataset` class in `src/agripipe/dataset.py`:

```python
class AgriDataset(Dataset):
    """PyTorch Dataset costruito sopra un DataFrame già pulito.
    
    Wrapper leggero attorno a ``Tensorizer`` che espone ``__len__`` e
    ``__getitem__`` per l'uso con ``torch.utils.data.DataLoader``.
    
    Attributes:
        tensorizer: Istanza di ``Tensorizer`` fit sui dati.
        features: Tensor 2D ``[N, D]`` float32, già normalizzato.
        target: Tensor 1D ``[N]`` o None per dataset unsupervised.
        feature_names: Ordine delle colonne nelle features.
    
    Example:
        >>> ds = AgriDataset(df_clean, numeric_columns=["temp", "ph"], target="yield")
        >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
    """
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agripipe/indices.py src/agripipe/loader.py src/agripipe/tensorizer.py src/agripipe/dataset.py
git commit -m "docs: adopt Google-style docstrings on core modules"
```

---

## Task 17: Final verification + coverage check

**Files:** no code changes, quality gate.

- [ ] **Step 1: Run full test suite with coverage report**

Run: `pytest --cov=agripipe --cov-report=term-missing`
Expected: all tests PASS, coverage ≥85% on new modules (`sustainability.py`, `metadata.py`, `export.py`, `ui/theme.py`).

- [ ] **Step 2: Run lint (ruff + black)**

Run: `ruff check src/agripipe tests`
Expected: no errors.

Run: `black --check src/agripipe tests`
Expected: no formatting issues. If any, run `black src/agripipe tests` and commit a formatting fix.

- [ ] **Step 3: Launch the Streamlit app and verify manually**

Run: `streamlit run src/agripipe/app.py`

Verify in the browser (http://localhost:8501):
- Hero header shows with Verde Bosco color
- Step 1 shows cascaded Regione → Coltura dropdowns
- Selecting a region filters crops correctly
- Uploading an Excel triggers the pipeline
- Step 4 shows 4 badges in a 2×2 grid with semaphore colors
- Download buttons produce valid `.xlsx` and `.zip` files
- Step 5 plots use Verde Salvia / Marrone Terra palette

Kill Streamlit after verification.

- [ ] **Step 4: Update DOCUMENTAZIONE_LOG.md with the upgrade summary**

Append to `DOCUMENTAZIONE_LOG.md`:

```markdown
### [2026-04-18] - Focus: Pro Upgrade per X Farm

**Obiettivo**: trasformare AgriPipe in un'applicazione professionale e facilissima da usare.

**Risultato**:
- UI Clean & Nature: palette Verde Salvia/Marrone Terra/Blu Acqua, step numerati, messaggi motivazionali.
- Selettore cascata Regione → Coltura (10 regioni italiane, 12 preset DOP/DOCG/IGP).
- Cleaner con imputazione time-series (rispetto del ciclo colturale) e fallback automatico a median.
- Sustainability Score Card: 4 badge semaforo (Azoto, Peronospora, Irrigazione, Suolo).
- Bundle ML-ready: `.pt` + `metadata.json` zippati, pronti per `DataLoader` senza attrito.
- Docstring Google-style su tutte le funzioni pubbliche.

**Architettura**:
- Nuovi moduli: `sustainability.py`, `metadata.py`, `export.py`, `ui/theme.py`, `ui/components.py`.
- Principio di isolamento: un file = una responsabilità.

**Coverage finale**: ≥85% sui moduli nuovi.

**Prossimo Step**: estendere il knowledge a tutte e 20 le regioni italiane + dashboard storica delle analisi.
```

- [ ] **Step 5: Commit final**

```bash
git add DOCUMENTAZIONE_LOG.md
git commit -m "docs: log Pro Upgrade completion (UI, scorecard, ML bundle)"
```

---

## Acceptance criteria (from spec)

| # | Criterio | Verificato da |
|---|---|---|
| 1 | Upload → bundle ML in <1 min senza istruzioni | Manual test Task 17 |
| 2 | Data Scientist apre .pt + .json → training in <3 righe | `test_exported_bundle_is_training_ready` |
| 3 | Score Card leggibile in <10s | Manual test Task 17 |
| 4 | Tutte le API pubbliche con docstring Google | Task 16 + review |
| 5 | Test 100% pass, coverage ≥85% nuovi moduli | Task 17 Step 1 |
| 6 | Valore aggiunto territoriale evidente | Cascata 10 regioni + 12 preset |
