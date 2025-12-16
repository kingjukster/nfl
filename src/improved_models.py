"""
Improved model implementations with advanced techniques.

This module provides enhanced versions of the prediction models with:
- Better feature engineering
- Advanced models (XGBoost, LightGBM)
- Hyperparameter tuning
- Ensemble methods
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import joblib

# Try to import advanced models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

logger = logging.getLogger(__name__)


def add_rolling_features(df, group_col='player_id', stat_cols=None, windows=[3, 5, 8]):
    """
    Add rolling average features for time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    group_col : str
        Column to group by (typically 'player_id')
    stat_cols : list, optional
        Columns to create rolling features for. If None, uses common stat columns.
    windows : list
        Rolling window sizes
        
    Returns:
    --------
    pd.DataFrame : DataFrame with rolling features added
    """
    if stat_cols is None:
        stat_cols = ['fantasy_points_standard', 'fantasy_points_ppr', 
                     'rushing_yards', 'receiving_yards', 'passing_yards']
        # Only use columns that exist
        stat_cols = [col for col in stat_cols if col in df.columns]
    
    df = df.sort_values([group_col, 'season', 'week'] if 'week' in df.columns else [group_col, 'season'])
    
    for col in stat_cols:
        for window in windows:
            new_col = f'{col}_rolling_{window}'
            df[new_col] = df.groupby(group_col)[col].rolling(
                window, min_periods=1
            ).mean().reset_index(0, drop=True)
    
    return df


def add_trend_features(df, group_col='player_id'):
    """
    Add trend features (year-over-year, game-over-game changes).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    group_col : str
        Column to group by
        
    Returns:
    --------
    pd.DataFrame : DataFrame with trend features added
    """
    df = df.sort_values([group_col, 'season'])
    
    # Year-over-year changes
    if 'fantasy_points_standard' in df.columns:
        df['fantasy_points_yoy'] = df.groupby(group_col)['fantasy_points_standard'].pct_change()
        df['fantasy_points_yoy_abs'] = df.groupby(group_col)['fantasy_points_standard'].diff()
    
    # Games played trend
    if 'games_played' in df.columns:
        df['games_played_trend'] = df.groupby(group_col)['games_played'].diff()
    
    # Age trend (should be +1 per year)
    if 'age' in df.columns:
        df['age_trend'] = df.groupby(group_col)['age'].diff()
    
    return df


def add_opponent_features(df, opponent_col='opponent_team', target_col='fantasy_points_standard'):
    """
    Add opponent strength features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    opponent_col : str
        Column name for opponent team
    target_col : str
        Target column to calculate opponent strength from
        
    Returns:
    --------
    pd.DataFrame : DataFrame with opponent features added
    """
    if opponent_col not in df.columns:
        logger.warning(f"Opponent column '{opponent_col}' not found. Skipping opponent features.")
        return df
    
    # Calculate opponent defensive/offensive strength
    opponent_strength = df.groupby(opponent_col)[target_col].mean()
    df['opponent_strength'] = df[opponent_col].map(opponent_strength)
    df['opponent_strength_rank'] = df['opponent_strength'].rank(pct=True)
    
    return df


def train_improved_model(X_train, y_train, X_test, y_test, model_type='xgb', 
                        tune_hyperparameters=True, position_name="Model"):
    """
    Train an improved model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        Test target
    model_type : str
        Model type: 'rf', 'xgb', 'lgbm', 'ensemble'
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
    position_name : str
        Name for logging
        
    Returns:
    --------
    tuple : (model, scaler, metrics_dict)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select and tune model
    if model_type == 'xgb' and XGBOOST_AVAILABLE:
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            base_model = XGBRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=30, cv=5,
                scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            logger.info(f"{position_name}: Best XGBoost params: {search.best_params_}")
        else:
            model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, 
                               random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
    
    elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=30, cv=5,
                scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            logger.info(f"{position_name}: Best LightGBM params: {search.best_params_}")
        else:
            model = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train_scaled, y_train)
    
    elif model_type == 'ensemble':
        # Create ensemble of multiple models
        estimators = []
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lgbm', LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)))
        estimators.append(('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)))
        
        model = VotingRegressor(estimators=estimators)
        model.fit(X_train_scaled, y_train)
    
    else:
        # Fallback to Random Forest with tuning
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=30, cv=5,
                scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            logger.info(f"{position_name}: Best RF params: {search.best_params_}")
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred)
    }
    
    # Baseline comparison
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    metrics['Baseline_MAE'] = baseline_mae
    if baseline_mae > 0:
        metrics['Improvement_%'] = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
    else:
        metrics['Improvement_%'] = 0.0
    
    return model, scaler, metrics


def remove_correlated_features(X, threshold=0.95):
    """
    Remove highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame
    threshold : float
        Correlation threshold (remove if > threshold)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with correlated features removed
    """
    if not isinstance(X, pd.DataFrame):
        return X
    
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    
    if to_drop:
        logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop[:5]}...")
        X = X.drop(columns=to_drop)
    
    return X


def time_series_cross_validate(X, y, model, n_splits=5):
    """
    Perform time series cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model : sklearn model
        Model to evaluate
    n_splits : int
        Number of splits
        
    Returns:
    --------
    dict : Cross-validation metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {'mae': [], 'rmse': [], 'r2': []}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        scores['mae'].append(mean_absolute_error(y_test, y_pred))
        scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        scores['r2'].append(r2_score(y_test, y_pred))
    
    return {
        'MAE_mean': np.mean(scores['mae']),
        'MAE_std': np.std(scores['mae']),
        'RMSE_mean': np.mean(scores['rmse']),
        'RMSE_std': np.std(scores['rmse']),
        'R²_mean': np.mean(scores['r2']),
        'R²_std': np.std(scores['r2'])
    }

