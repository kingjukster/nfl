"""
Data Processing Utilities

Reusable functions for processing and transforming NFL data.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

from src.constants import normalize_team_name

logger = logging.getLogger(__name__)


def normalize_team_names_in_df(df: pd.DataFrame, team_column: str = 'team') -> pd.DataFrame:
    """
    Normalize team names in a dataframe column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing team names
    team_column : str
        Name of column containing team names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized team names
    """
    if team_column not in df.columns:
        logger.warning(f"Column '{team_column}' not found in dataframe")
        return df
    
    df = df.copy()
    df[team_column] = df[team_column].apply(normalize_team_name)
    return df


def chronological_split(X: pd.DataFrame, y: pd.Series, 
                        test_season: int = None,
                        season_col: str = 'season') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically by season (more appropriate for time series).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    test_season : int, optional
        Season to use as test set. If None, uses most recent season.
    season_col : str
        Name of season column
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if season_col not in X.columns:
        logger.warning(f"Season column '{season_col}' not found, using random split")
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    if test_season is None:
        test_season = X[season_col].max()
    
    train_mask = X[season_col] < test_season
    test_mask = X[season_col] >= test_season
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    logger.info(f"Chronological split: Train seasons < {test_season}, Test season >= {test_season}")
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def filter_by_season(df: pd.DataFrame, season: int, 
                     season_col: str = 'season') -> pd.DataFrame:
    """
    Filter dataframe to specific season.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to filter
    season : int
        Season year to filter to
    season_col : str
        Name of season column
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    if season_col not in df.columns:
        logger.warning(f"Season column '{season_col}' not found")
        return df
    
    filtered = df[df[season_col] == season].copy()
    logger.debug(f"Filtered to season {season}: {len(filtered)} rows")
    return filtered

