"""
Adapter to convert raw QB stats to QBComparisonDataset.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from src.models.qb_metrics import QBComparisonDataset
from src.validation import validate_qb_metrics


class StatsAdapter:
    """
    Converts raw QB statistics to QBComparisonDataset.
    
    Handles normalization and metric selection.
    """
    
    @staticmethod
    def to_qb_dataset(
        qb_stats_df: pd.DataFrame,
        season: int,
        metrics: Optional[List[str]] = None
    ) -> QBComparisonDataset:
        """
        Convert raw QB stats to normalized QBComparisonDataset.
        
        Args:
            qb_stats_df: DataFrame with QB statistics
            season: Season year
            metrics: Optional list of metric column names to include
        
        Returns:
            QBComparisonDataset with normalized metrics
        """
        if qb_stats_df.empty:
            raise ValueError("QB stats DataFrame cannot be empty")
        
        # Required columns
        required_cols = ['player_name', 'team']
        for col in required_cols:
            if col not in qb_stats_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Default metrics if not specified
        if metrics is None:
            metrics = [
                'epa_per_play_norm',
                'completion_pct_norm',
                'td_rate_norm',
                'int_rate_norm',
                'sack_rate_norm',
                'win_rate_norm'
            ]
        
        # Filter to only available metrics
        available_metrics = [m for m in metrics if m in qb_stats_df.columns]
        
        if not available_metrics:
            raise ValueError("No normalized metrics found in DataFrame")
        
        # Create subset with required columns + metrics
        columns_to_keep = required_cols + available_metrics
        dataset_df = qb_stats_df[columns_to_keep].copy()
        
        # Validate metrics
        validate_qb_metrics(dataset_df, required_metrics=available_metrics)
        
        # Create QBComparisonDataset
        return QBComparisonDataset(
            data=dataset_df,
            metrics=available_metrics,
            season=season
        )
    
    @staticmethod
    def normalize_metrics(
        qb_stats_df: pd.DataFrame,
        metric_config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Normalize QB metrics to 0-1 scale.
        
        Args:
            qb_stats_df: DataFrame with raw QB stats
            metric_config: Optional configuration for normalization
        
        Returns:
            DataFrame with normalized metrics added
        """
        df = qb_stats_df.copy()
        
        # Default normalization config
        if metric_config is None:
            metric_config = {
                'epa_per_play': {'method': 'min_max', 'invert': False},
                'completion_pct': {'method': 'min_max', 'invert': False, 'scale': 100.0},
                'td_rate': {'method': 'min_max', 'invert': False},
                'int_rate': {'method': 'min_max', 'invert': True},  # Lower is better
                'sack_rate': {'method': 'min_max', 'invert': True},  # Lower is better
                'win_rate': {'method': 'min_max', 'invert': False},
            }
        
        # Apply normalization
        for metric, config in metric_config.items():
            if metric not in df.columns:
                continue
            
            norm_col = f"{metric}_norm"
            method = config.get('method', 'min_max')
            invert = config.get('invert', False)
            scale = config.get('scale', 1.0)
            
            if method == 'min_max':
                col_data = df[metric].dropna()
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    if max_val > min_val:
                        df[norm_col] = (df[metric] / scale - min_val) / (max_val - min_val)
                        if invert:
                            df[norm_col] = 1.0 - df[norm_col]
                    else:
                        df[norm_col] = 0.5  # All same value
                else:
                    df[norm_col] = 0.0
        
        return df

