"""
QB metrics validation functions.

Validates QB comparison datasets for visualization.
"""

import pandas as pd
from typing import List


def validate_qb_metrics(df: pd.DataFrame, 
                       required_metrics: List[str] = None,
                       normalized_range: tuple = (0.0, 1.0)) -> None:
    """
    Validate QB metrics DataFrame.
    
    Validation rules:
    - Required columns present (player_name, team)
    - Metrics in valid ranges (completion % 0-100, normalized metrics 0-1)
    - Handle missing values explicitly
    - No negative values for rate metrics
    
    Args:
        df: DataFrame with QB metrics
        required_metrics: List of metric column names that must be present
        normalized_range: Tuple (min, max) for normalized metrics
    """
    if df.empty:
        raise ValueError("QB metrics DataFrame cannot be empty")
    
    # Check required columns
    required_cols = ['player_name', 'team']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check required metrics if specified
    if required_metrics:
        for metric in required_metrics:
            if metric not in df.columns:
                raise ValueError(f"Missing required metric: {metric}")
    
    # Validate ranges for common metrics
    metric_ranges = {
        'completion_pct': (0.0, 100.0),
        'completion_pct_norm': normalized_range,
        'epa_per_play_norm': normalized_range,
        'td_rate_norm': normalized_range,
        'int_rate_norm': normalized_range,
        'sack_rate_norm': normalized_range,
        'win_rate_norm': normalized_range,
    }
    
    for metric, (min_val, max_val) in metric_ranges.items():
        if metric in df.columns:
            col_data = df[metric].dropna()
            if len(col_data) > 0:
                if col_data.min() < min_val or col_data.max() > max_val:
                    raise ValueError(
                        f"Metric {metric} out of range: "
                        f"min={col_data.min():.3f}, max={col_data.max():.3f}, "
                        f"expected [{min_val}, {max_val}]"
                    )
    
    # Check for negative values in rate metrics
    rate_metrics = [col for col in df.columns if 'rate' in col.lower() or 'pct' in col.lower()]
    for metric in rate_metrics:
        if metric in df.columns:
            negative_count = (df[metric] < 0).sum()
            if negative_count > 0:
                raise ValueError(f"Metric {metric} has {negative_count} negative values")
    
    # Check for missing values (warn but don't fail - missing values are handled explicitly)
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Missing values found in QB metrics:\n{missing_counts[missing_counts > 0]}")

