"""
Data Loading Utilities

Safe, reusable functions for loading data files with proper error handling.
"""
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def load_csv_safe(file_path: str, required: bool = True, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load CSV file with comprehensive error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    required : bool
        If True, raise exception on error; if False, return None
    **kwargs
        Additional arguments to pass to pd.read_csv
        
    Returns:
    --------
    pd.DataFrame or None
        Loaded dataframe or None if error and not required
    """
    path = Path(file_path)
    
    if not path.exists():
        error_msg = f"Data file not found: {file_path}"
        if required:
            raise FileNotFoundError(error_msg)
        logger.warning(error_msg)
        return None
    
    try:
        df = pd.read_csv(path, **kwargs)
        
        if df.empty:
            error_msg = f"Empty dataset: {file_path}"
            if required:
                raise ValueError(error_msg)
            logger.warning(error_msg)
            return None
        
        logger.debug(f"Loaded {len(df)} rows from {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        error_msg = f"Empty CSV file: {file_path}"
        if required:
            raise ValueError(error_msg)
        logger.warning(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error loading {file_path}: {e}"
        if required:
            raise
        logger.error(error_msg)
        return None


def load_json_safe(file_path: str, required: bool = True) -> Optional[dict]:
    """
    Load JSON file with comprehensive error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to JSON file
    required : bool
        If True, raise exception on error; if False, return None
        
    Returns:
    --------
    dict or None
        Loaded JSON data or None if error and not required
    """
    path = Path(file_path)
    
    if not path.exists():
        error_msg = f"JSON file not found: {file_path}"
        if required:
            raise FileNotFoundError(error_msg)
        logger.warning(error_msg)
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in {file_path}: {e}"
        if required:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error loading {file_path}: {e}"
        if required:
            raise
        logger.error(error_msg)
        return None


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, 
                      required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate dataframe meets minimum requirements.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    min_rows : int
        Minimum number of rows required
    required_columns : list of str, optional
        List of required column names
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        logger.warning(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        return False
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return False
    
    return True


def ensure_columns_exist(df: pd.DataFrame, columns: List[str], 
                         fill_value=None) -> pd.DataFrame:
    """
    Ensure specified columns exist in dataframe, creating them if needed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    columns : list of str
        Column names to ensure exist
    fill_value
        Value to use for new columns (default: None)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all required columns
    """
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
            logger.debug(f"Created missing column: {col}")
    
    return df

