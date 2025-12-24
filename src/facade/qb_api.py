"""
QB comparison rendering facade.

Clean public API for QB visualization.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from src.models.qb_metrics import QBComparisonDataset
from src.adapters.stats_adapter import StatsAdapter
from src.charts.bar_strategy import GroupedBarStrategy
from src.charts.radar_strategy import RadarStrategy
from src.charts.base import ChartStrategy
from src.validation import validate_qb_metrics
from src.visualization import save_figure_safe


def render_qb_comparison(
    qb_df: pd.DataFrame,
    out_path: Path,
    season: int,
    strategy: str = "bars",
    options: Dict[str, Any] = None
) -> None:
    """
    Render QB comparison chart.
    
    Clean public API for QB visualization.
    
    Args:
        qb_df: DataFrame with QB statistics (must have normalized metrics)
        out_path: Output file path
        season: Season year
        strategy: Chart strategy ('bars' or 'radar')
        options: Optional chart options:
            - figsize: Tuple (width, height)
            - title: Chart title
            - max_qbs: Maximum QBs for radar (default: 3)
            - show_values: Show values on bars (default: True)
    """
    if options is None:
        options = {}
    
    # Validate QB metrics
    validate_qb_metrics(qb_df)
    
    # Adapt to QBComparisonDataset
    adapter = StatsAdapter()
    qb_dataset = adapter.to_qb_dataset(qb_df, season)
    
    # Select strategy
    if strategy == "bars":
        chart_strategy: ChartStrategy = GroupedBarStrategy()
    elif strategy == "radar":
        chart_strategy = RadarStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'bars' or 'radar'")
    
    # Create figure
    fig = chart_strategy.make_figure(qb_dataset, options=options)
    
    # Save
    if save_figure_safe(fig, str(out_path), dpi=300):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Saved QB comparison chart to {out_path}")


def render_qb_comparison_from_raw(
    qb_df: pd.DataFrame,
    out_path: Path,
    season: int,
    strategy: str = "bars",
    options: Dict[str, Any] = None
) -> None:
    """
    Render QB comparison from raw stats (with normalization).
    
    Convenience function that normalizes metrics before rendering.
    
    Args:
        qb_df: DataFrame with raw QB statistics
        out_path: Output file path
        season: Season year
        strategy: Chart strategy ('bars' or 'radar')
        options: Optional chart options
    """
    # Normalize metrics
    adapter = StatsAdapter()
    normalized_df = adapter.normalize_metrics(qb_df)
    
    # Render
    render_qb_comparison(normalized_df, out_path, season, strategy, options)

