
# How to Vet a Dataset Before Modeling — A Practical EDA Checklist

You learned an expensive lesson yesterday. Let's turn it into a permanent skill. The rule is:

> **Never train a model until EDA has proven that signal exists.**

Here's a systematic workflow to catch "garbage datasets" *before* wasting hours.

---

## Part 1 — The 15-Minute Dataset Triage

Run this **first**, every single time, before touching a model.

### Step 1: Structural Sanity

```python
df.shape                 # rows, cols
df.info()                # dtypes, non-null counts
df.head(10)
df.tail(10)
df.describe().T          # stats for numeric cols
df.nunique()             # cardinality — catch constant/ID columns
```

**Red flags:**

- A column with `nunique == 1` (constant) → drop it
- A column with `nunique == len(df)` → probably an ID → drop it
- Numeric columns stored as `object` dtype → encoding issue upstream
- `describe()` showing min/max that violate reality (negative prices, age > 200)

### Step 2: The Correlation Reality Check (most important)

```python
corr = df.corr(numeric_only=True)['Price'].sort_values()
print(corr)
```

**Interpretation for your house dataset:**

| Max abs correlation | Verdict                                     |
| ------------------- | ------------------------------------------- |
| > 0.5               | Strong signal — great                      |
| 0.2 – 0.5          | Weak but usable                             |
| 0.05 – 0.2         | Very weak — needs feature engineering      |
| **< 0.05**    | **🚨 No signal — STOP, don't model** |

Your dataset hit the last row. That alone would have saved you the whole day.

### Step 3: Visual Signal Check

Numbers lie; plots don't. For **regression** targets:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter every feature vs target
num_cols = df.select_dtypes('number').columns.drop('Price')
fig, axes = plt.subplots(1, len(num_cols), figsize=(4*len(num_cols), 4))
for ax, col in zip(axes, num_cols):
    ax.scatter(df[col], df['Price'], s=5, alpha=0.4)
    ax.set_xlabel(col); ax.set_ylabel('Price')
plt.tight_layout(); plt.show()
```

**If every scatter looks like a shapeless cloud → no linear signal.** For your data, that's exactly what you'd see.

For **categorical** features vs target:

```python
for col in ['Location', 'Condition', 'Garage']:
    sns.boxplot(data=df, x=col, y='Price')
    plt.title(f'Price by {col}')
    plt.show()
```

If the boxes all sit at the same height → the category doesn't matter.

### Step 4: The "Target Mean Baseline" Test

This is the **single fastest kill-switch** for a bad dataset:

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

# Baseline: always predict the mean
baseline = cross_val_score(
    DummyRegressor(strategy='mean'),
    X, y, cv=5, scoring='r2'
).mean()
print("Baseline R² (predict mean):", baseline)   # always ≈ 0

# Your model
model_r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean()
print("Model R²:", model_r2)
```

**Decision rule:**

- If `model_r2` is not **meaningfully higher** than baseline (e.g., +0.10 or more) → **the features carry no information.** Stop.

For your dataset, model R² ≈ 0.00 = baseline. Immediate red flag.

### Step 5: Shuffle Test (Detects "Leakage" and "Fake" Signals)

Scramble the target and retrain. If your model still scores well, something is wrong (e.g., an ID column leaking).

```python
from sklearn.utils import shuffle

y_shuffled = shuffle(y, random_state=0)
r2_shuffled = cross_val_score(
    LinearRegression(), X, y_shuffled, cv=5, scoring='r2'
).mean()
print("R² with shuffled target:", r2_shuffled)  # should be ≈ 0 or negative
```

If real R² ≈ shuffled R² ≈ 0 → **the dataset is noise.**

---

## Part 2 — Feature Engineering That Reveals Signal

Sometimes raw features are weak but combinations are strong. Try these before giving up:

### 1. Domain-driven ratios

```python
df['Age']          = 2024 - df['YearBuilt']
df['AreaPerRoom']  = df['Area'] / (df['Bedrooms'] + df['Bathrooms'] + 1)
df['TotalRooms']   = df['Bedrooms'] + df['Bathrooms']
df['IsNew']        = (df['YearBuilt'] > 2010).astype(int)
```

Then recheck correlation:

```python
df.corr(numeric_only=True)['Price'].sort_values()
```

If even engineered features stay at |r| < 0.05 → give up on linear models with this data.

### 2. Non-linear relationship check

A feature can correlate ~0 with the target but still be useful non-linearly. Test with a tree model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf_r2 = cross_val_score(
    RandomForestRegressor(n_estimators=200, random_state=0),
    X, y, cv=5, scoring='r2'
).mean()
print("Random Forest R²:", rf_r2)
```

**Key insight:** If linear R² ≈ 0 but RF R² is high → non-linear signal exists.
If **both** ≈ 0 → the data is truly random. You now *know* it, no guessing.

### 3. Feature importance check

```python
rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
import pandas as pd
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print(imp)
```

If all importances are ≈ 1/n_features → random data confirmed.

---

## Part 3 — Choosing the Right Model for the Right Data

Once you've confirmed signal exists, pick the model based on **data characteristics**, not preference.

| Situation                          | Best first model                            | Why                        |
| ---------------------------------- | ------------------------------------------- | -------------------------- |
| Linear relationships, few features | **Linear / Ridge / Lasso**            | Interpretable, fast        |
| Linear + many correlated features  | **Ridge / ElasticNet**                | Handles multicollinearity  |
| Non-linear, tabular, < 100k rows   | **Random Forest / Gradient Boosting** | Captures interactions      |
| Tabular, need max accuracy         | **XGBoost / LightGBM / CatBoost**     | Usually SOTA on tabular    |
| Small n (< 500)                    | **Linear + CV, then RF**              | Avoid overfitting          |
| High-dimensional (p > n)           | **Lasso / ElasticNet**                | Built-in feature selection |
| Time-series prices                 | **ARIMA / Prophet / LSTM**            | Respect temporal order     |
| Text/categorical-heavy             | **Target encoding + GBM**             | Handles categories well    |

### Decision flowchart

```
1. Is there signal? (correlation / RF test)
   NO  → Stop. Fix data.
   YES → continue

2. Is the relationship roughly linear?
   YES → Linear / Ridge / Lasso
   NO  → Tree-based (RF → GBM)

3. Do you need interpretability?
   YES → Linear or SHAP on GBM
   NO  → GBM / XGBoost

4. Is n small (< 1000)?
   YES → Use cross-validation everywhere, prefer simpler models
   NO  → Try boosting, tune hyperparams

5. Is it temporal?
   YES → TimeSeriesSplit, not random split
```

---

## Part 4 — The Pre-Modeling Checklist (Print This)

Before fitting anything, confirm **all** of these:

- [ ] `df.info()` — dtypes correct, no surprise nulls
- [ ] Dropped ID / constant columns
- [ ] `df.describe()` — no impossible values
- [ ] Target distribution plotted — no crazy skew without reason
- [ ] **Correlation of each feature with target computed** ← the money step
- [ ] At least one feature has |corr| > 0.1, OR RF test shows R² > 0.1
- [ ] Scatter plots of top-correlated features look non-random
- [ ] Baseline (predict mean) R² computed
- [ ] Shuffle test passes (shuffled R² ≈ 0)

**If any check fails → do not proceed to hyperparameter tuning.** You'd be polishing a rock.

---

## Part 5 — Applied to Your House Dataset

Here's exactly what *should have happened* yesterday:

```python
# 1. Correlations
corr = df.corr(numeric_only=True)['Price'].drop('Price')
print(corr.abs().max())        # → ~0.02  🚨

# 2. RF test
rf_r2 = cross_val_score(RandomForestRegressor(n_estimators=200),
                        X, y, cv=5, scoring='r2').mean()
print(rf_r2)                   # → ~0.00  🚨

# 3. Shuffle test
print(cross_val_score(LinearRegression(), X,
      shuffle(y, random_state=0), cv=5, scoring='r2').mean())  # → ~0.00
```

Three independent tests all say the same thing in **under 60 seconds**: *no signal*. That's your early-exit trigger.

---

## The One-Sentence Rule to Remember

> **"Before I tune a model, I must have already seen — with a number or a plot — that at least one feature moves with the target."**

Everything you suffered yesterday collapses into that single habit.

---

Want me to:

1. Build you a **reusable `audit_dataset()` function** that runs all these checks and prints a verdict?
2. Show the same workflow on a **known-good dataset** (e.g., California Housing) so you can compare what "signal" looks like?
3. Go deeper into **non-linear signal detection** (mutual information, partial dependence)?
