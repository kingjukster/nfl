"""
Utility functions for NFL Prediction Project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_csv_safe(file_path: str, required: bool = True) -> pd.DataFrame:
    """
    Safely load a CSV file with error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    required : bool
        If True, raise error if file doesn't exist; if False, return empty DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    path = Path(file_path)
    
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {file_path}")
        else:
            logger.warning(f"Optional file not found: {file_path}")
            return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"File is empty: {file_path}")
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV file: {file_path}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def chronological_split(X: pd.DataFrame, y: pd.Series, test_season: int = None) -> tuple:
    """
    Split data chronologically by season.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features DataFrame (must have 'season' column)
    y : pd.Series
        Target Series
    test_season : int, optional
        Season to use for testing. If None, uses the latest season.
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
        Split data
    """
    if 'season' not in X.columns:
        raise ValueError("X must have a 'season' column for chronological split")
    
    if test_season is None:
        test_season = int(X['season'].max())
    
    train_mask = X['season'] < test_season
    test_mask = X['season'] == test_season
    
    if test_mask.sum() == 0:
        raise ValueError(f"No data found for test season {test_season}")
    
    X_train = X.loc[train_mask].drop(columns=['season'], errors='ignore')
    X_test = X.loc[test_mask].drop(columns=['season'], errors='ignore')
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    logger.info(f"Chronological split - Train: {len(X_train)} rows, Test: {len(X_test)} rows (season {test_season})")
    
    return X_train, X_test, y_train, y_test


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_train : np.ndarray, optional
        Training values for baseline comparison
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
    )
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'MedAE': median_absolute_error(y_true, y_pred)
    }
    
    # Baseline comparison if training data provided
    if y_train is not None:
        baseline_pred = np.full_like(y_true, y_train.mean())
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        metrics['Baseline_MAE'] = baseline_mae
        if baseline_mae > 0:
            metrics['Improvement_%'] = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
        else:
            metrics['Improvement_%'] = 0.0
    
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print(f"\n📊 {model_name} Performance:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R²']:.3f}")
    print(f"  MedAE: {metrics['MedAE']:.2f}")
    
    if 'Improvement_%' in metrics:
        print(f"  Improvement over baseline: {metrics['Improvement_%']:.1f}%")


def remove_duplicates_from_list(lst: list) -> list:
    """
    Remove duplicates from a list while preserving order.
    
    Parameters:
    -----------
    lst : list
        List with potential duplicates
        
    Returns:
    --------
    list
        List without duplicates
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

