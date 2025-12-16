# Improving Predictive Abilities - Comprehensive Guide

## 📊 Current Performance

Based on your comparison results:
- **CB Model**: R² = 0.955, MAE = 0.12 (Excellent!)
- **Limited data**: Only 6 teams matched (need more defensive player data)

## 🎯 Improvement Strategies

### 1. **Advanced Feature Engineering** ⭐⭐⭐ (High Impact)

#### Time Series Features
```python
# Rolling averages (last 3, 5, 8 games)
def add_rolling_features(df, group_col='player_id', stat_cols=['fantasy_points']):
    df = df.sort_values(['player_id', 'season', 'week'])
    for col in stat_cols:
        df[f'{col}_rolling_3'] = df.groupby(group_col)[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{col}_rolling_5'] = df.groupby(group_col)[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{col}_rolling_8'] = df.groupby(group_col)[col].rolling(8, min_periods=1).mean().reset_index(0, drop=True)
    return df
```

#### Trend Features
```python
# Year-over-year trends
def add_trend_features(df):
    df['fantasy_points_trend'] = df.groupby('player_id')['fantasy_points'].pct_change()
    df['games_played_trend'] = df.groupby('player_id')['games_played'].diff()
    return df
```

#### Opponent Strength Features
```python
# Opponent-adjusted stats
def add_opponent_features(df):
    # Calculate opponent defensive strength
    opponent_def_strength = df.groupby('opponent_team')['fantasy_points'].mean()
    df['opponent_def_rank'] = df['opponent_team'].map(opponent_def_strength).rank()
    return df
```

### 2. **Better Model Selection** ⭐⭐⭐ (High Impact)

#### Try Gradient Boosting Models
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# XGBoost often outperforms Random Forest
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# LightGBM is faster and often more accurate
lgbm_model = LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

#### Ensemble Methods
```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Voting ensemble
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=200)),
    ('xgb', XGBRegressor(n_estimators=200)),
    ('lgbm', LGBMRegressor(n_estimators=200))
])

# Stacking (more advanced)
stacking = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=200)),
        ('xgb', XGBRegressor(n_estimators=200))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
```

### 3. **Hyperparameter Tuning** ⭐⭐ (Medium Impact)

#### For Random Forest (Currently Missing)
```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)
```

### 4. **Cross-Validation Strategy** ⭐⭐ (Medium Impact)

#### Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

# Use time series split instead of single split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

#### Walk-Forward Validation
```python
# Validate on each season sequentially
for test_season in range(2020, 2025):
    train_mask = df['season'] < test_season
    test_mask = df['season'] == test_season
    # Train on all previous seasons, test on current
```

### 5. **Feature Selection** ⭐⭐ (Medium Impact)

#### Remove Low-Importance Features
```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# Select top K features
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X_train, y_train)

# Or use Recursive Feature Elimination
rfe = RFE(RandomForestRegressor(n_estimators=100), n_features_to_select=20)
X_selected = rfe.fit_transform(X_train, y_train)
```

#### Remove Highly Correlated Features
```python
# Remove features with correlation > 0.95
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
X_train = X_train.drop(columns=high_corr_features)
```

### 6. **Advanced Features** ⭐ (Lower Impact, But Valuable)

#### Injury/Availability Features
```python
# Games missed in previous season
df['games_missed_last_season'] = df.groupby('player_id')['games_missed'].shift(1)

# Injury risk score (based on age and games missed history)
df['injury_risk'] = (df['age'] / 30) * (df['games_missed_last_season'] / 17)
```

#### Schedule Strength
```python
# Strength of schedule (opponent win percentage)
opponent_wins = df.groupby('opponent_team')['win'].mean()
df['sos'] = df['opponent_team'].map(opponent_wins)
```

#### Weather/Stadium Features (if available)
```python
# Dome vs outdoor
# Weather conditions
# Home vs away
```

### 7. **Data Quality Improvements** ⭐⭐ (Medium Impact)

#### Handle Outliers
```python
from scipy import stats

# Remove statistical outliers
z_scores = np.abs(stats.zscore(df['fantasy_points']))
df = df[z_scores < 3]  # Keep within 3 standard deviations
```

#### Better Missing Value Handling
```python
# Use more sophisticated imputation
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
```

### 8. **Position-Specific Improvements** ⭐⭐⭐ (High Impact)

#### QB-Specific Features
```python
# Pass attempts per game trend
# Completion percentage trend
# Red zone efficiency
# Deep ball accuracy
# Rushing attempts (mobile QBs)
```

#### RB-Specific Features
```python
# Touchdown rate in red zone
# Yards per carry by down
# Receiving targets trend
# Snap share trend
```

#### WR-Specific Features
```python
# Target share trend
# Air yards share
# Red zone target rate
# Deep target rate
```

#### Defensive Position Features
```python
# Snap percentage trend
# Tackle efficiency
# Coverage snaps vs run snaps ratio
# Blitz rate
```

## 🚀 Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ Add hyperparameter tuning to Random Forest
2. ✅ Try XGBoost/LightGBM models
3. ✅ Add rolling average features
4. ✅ Implement time series cross-validation

### Phase 2: Medium Effort (3-5 days)
5. ✅ Add opponent strength features
6. ✅ Implement feature selection
7. ✅ Add trend features
8. ✅ Try ensemble methods

### Phase 3: Advanced (1-2 weeks)
9. ✅ Add injury/availability features
10. ✅ Implement stacking
11. ✅ Add schedule strength
12. ✅ Position-specific feature engineering

## 📈 Expected Improvements

- **Current**: R² = 0.955, MAE = 0.12 (CB model)
- **With XGBoost**: +2-5% R² improvement
- **With Feature Engineering**: +3-7% R² improvement
- **With Ensemble**: +1-3% R² improvement
- **Combined**: Potential R² > 0.98, MAE < 0.10

## 🎯 Specific Recommendations for Your Models

### For Offensive Models (attacker.py)
1. **Add hyperparameter tuning** (currently missing)
2. **Try XGBoost** instead of just Random Forest
3. **Add rolling averages** for key stats
4. **Add opponent strength** adjustments

### For Defensive Models (MLNFL.py)
1. **Expand GridSearchCV** to more parameters
2. **Try XGBoost** in addition to Ridge
3. **Add snap percentage trends**
4. **Add opponent offensive strength** features

### For Team Win Predictions (heatMap2.py)
1. **Try more sophisticated models** (XGBoost, Neural Networks)
2. **Add head-to-head history** features
3. **Add recent form** (last 5 games)
4. **Add injury reports** if available

## 💡 Next Steps

1. Start with Phase 1 improvements (quick wins)
2. Measure improvement after each change
3. Keep what works, discard what doesn't
4. Iterate and refine

---

**Remember**: Small improvements compound. Even 1-2% better predictions can make a significant difference in fantasy football!

