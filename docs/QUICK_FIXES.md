# Quick Fixes - Immediate Improvements

## 🚀 Quick Wins (Can implement in < 30 minutes)

### 1. Fix Duplicate Items in attacker.py

**File:** `attacker.py` lines 45-52

**Current:**
```python
to_normalize = [
    "offense_snaps", "touches", "targets", "receptions",
    "rushing_yards", "receiving_yards", "yards_after_catch", "total_yards",
    "rush_touchdown", "receiving_touchdown", "total_tds",
    "rush_attempts", "rush_attempts_redzone", "targets_redzone",
    "touches", "total_yards", "rush_touchdown",  # DUPLICATES
    "receiving_yards", "offense_snaps"  # DUPLICATES
]
```

**Fixed:**
```python
to_normalize = [
    "offense_snaps", "touches", "targets", "receptions",
    "rushing_yards", "receiving_yards", "yards_after_catch", "total_yards",
    "rush_touchdown", "receiving_touchdown", "total_tds",
    "rush_attempts", "rush_attempts_redzone", "targets_redzone"
]
# Remove duplicates: "touches", "total_yards", "rush_touchdown", 
# "receiving_yards", "offense_snaps" already listed above
```

### 2. Remove Duplicate Import in attacker.py

**File:** `attacker.py` line 15

**Current:**
```python
import pandas as pd  # line 2
# ...
import pandas as pd  # line 15 - DUPLICATE
```

**Fixed:** Remove line 15

### 3. Add File Existence Checks

**File:** `MLNFL.py` line 8

**Current:**
```python
df = pd.read_csv("yearly_player_stats_defense.csv")
```

**Fixed:**
```python
import os
from pathlib import Path

csv_path = Path("yearly_player_stats_defense.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Data file not found: {csv_path}")
df = pd.read_csv(csv_path)
```

### 4. Use Chronological Split in attacker.py

**File:** `attacker.py` line 112

**Current:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Fixed:**
```python
# Chronological split by season (more appropriate for time series)
if 'season' in X.columns:
    max_season = X['season'].max()
    train_mask = X['season'] < max_season
    test_mask = X['season'] == max_season
    
    X_train = X.loc[train_mask].drop(columns=['season'], errors='ignore')
    X_test = X.loc[test_mask].drop(columns=['season'], errors='ignore')
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
else:
    # Fallback to random split if no season column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 5. Add Model Persistence

**File:** `attacker.py` after line 131

**Add:**
```python
import joblib
from pathlib import Path

# Save model and scaler
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

model_path = output_dir / f"{position_name.lower().replace(' ', '_')}_model.pkl"
scaler_path = output_dir / f"{position_name.lower().replace(' ', '_')}_scaler.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
```

### 6. Add Comprehensive Metrics

**File:** `attacker.py` lines 125-130

**Current:**
```python
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 {position_name} model performance:")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}\n")
```

**Fixed:**
```python
from sklearn.metrics import mean_squared_error, median_absolute_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Baseline comparison
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
improvement = ((baseline_mae - mae) / baseline_mae) * 100

print(f"📊 {position_name} model performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
print(f"MedAE: {medae:.2f}")
print(f"Improvement over baseline: {improvement:.1f}%\n")
```

### 7. Use GridSearchCV in MLNFL.py

**File:** `MLNFL.py` lines 165-166

**Current:**
```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

**Fixed:**
```python
# Use GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]}
ridge_cv = GridSearchCV(
    Ridge(), 
    param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)
ridge_cv.fit(X_train_scaled, y_train)
ridge = ridge_cv.best_estimator_

print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
print(f"Best CV score: {ridge_cv.best_score_:.3f}")
```

### 8. Add Error Handling to heatMap2.py

**File:** `heatMap2.py` lines 67-68

**Current:**
```python
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
```

**Fixed:**
```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    train_path = Path(TRAIN_PATH)
    test_path = Path(TEST_PATH)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    if train_df.empty:
        raise ValueError(f"Training file is empty: {TRAIN_PATH}")
    if test_df.empty:
        raise ValueError(f"Test file is empty: {TEST_PATH}")
        
    logger.info(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows")
    
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise
```

### 9. Create requirements.txt

**New File:** `requirements.txt`

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
kagglehub>=0.1.0
scipy>=1.10.0
joblib>=1.3.0
```

### 10. Create .gitignore

**New File:** `.gitignore`

```gitignore
# Data files
*.csv
!requirements.txt
data/
output/
models/*.pkl
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# OS
Thumbs.db
```

## 📋 Implementation Checklist

- [ ] Fix duplicate items in `to_normalize` list
- [ ] Remove duplicate pandas import
- [ ] Add file existence checks to all CSV reads
- [ ] Change random split to chronological split in attacker.py
- [ ] Add model persistence (save/load)
- [ ] Add comprehensive evaluation metrics
- [ ] Use GridSearchCV in MLNFL.py
- [ ] Add error handling to heatMap2.py
- [ ] Create requirements.txt
- [ ] Create .gitignore

## 🎯 Estimated Time

- Quick fixes (1-4): 15 minutes
- Medium fixes (5-8): 30 minutes
- Setup files (9-10): 5 minutes

**Total: ~50 minutes for all quick fixes**

