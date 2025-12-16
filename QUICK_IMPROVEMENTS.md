# Quick Improvements Guide

## 🚀 Fastest Ways to Improve Predictions

### 1. Install Advanced Models (5 minutes)
```bash
pip install xgboost lightgbm
```

### 2. Use Improved Training Script (10 minutes)
```bash
python src/train_improved_offensive.py
```

This will:
- Try XGBoost, LightGBM, and Random Forest
- Automatically select the best model
- Add trend features
- Remove correlated features
- Perform hyperparameter tuning

### 3. Quick Code Changes

#### For Offensive Models (src/attacker.py)
Replace line 199:
```python
# OLD:
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# NEW:
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, 
                     random_state=42, n_jobs=-1)
```

#### For Defensive Models (src/MLNFL.py)
The GridSearchCV is already good, but expand the parameter grid:
```python
# OLD:
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

# NEW:
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
}
```

### 4. Add Simple Rolling Features

In `src/attacker.py`, after line 105, add:
```python
# Add rolling averages if season/week data available
if 'season' in avg_features.columns:
    from improved_models import add_trend_features
    avg_features = add_trend_features(avg_features, group_col='player_id')
```

## 📈 Expected Improvements

- **XGBoost**: +2-5% R² improvement
- **Feature Engineering**: +3-7% R² improvement  
- **Hyperparameter Tuning**: +1-3% R² improvement
- **Combined**: Potential to reach R² > 0.98

## 🎯 Priority Order

1. **Install XGBoost/LightGBM** (5 min) - Biggest impact
2. **Run improved training script** (10 min) - Easy win
3. **Add trend features** (15 min) - Good improvement
4. **Expand hyperparameter grids** (10 min) - Better tuning

---

See `docs/IMPROVING_PREDICTIONS.md` for comprehensive guide.

