"""
Visualize quarterback playoff performance using radar charts.

This module creates radar/spider charts showing QB playoff statistics
to complement team-level playoff predictions.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to load OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.debug("OpenAI library not available")

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    # Try to load .env manually
    try:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
    except Exception as e:
        logger.debug(f"Could not load .env file: {e}")

# Import modular utilities
from src.constants import get_team_color
from src.visualization import setup_plot_style, apply_professional_styling, save_figure_safe


def create_qb_playoff_bars(qb_stats_df: pd.DataFrame, output_path: str, 
                            season: int) -> None:
    """
    Create a grouped horizontal bar chart showing QB playoff performance metrics.
    
    Note: This function now uses the new facade API internally.
    For new code, prefer using src.facade.qb_api.render_qb_comparison().
    
    Parameters:
    -----------
    qb_stats_df : pd.DataFrame
        DataFrame with normalized QB stats (from normalize_qb_metrics)
    output_path : str
        Path to save the image
    season : int
        Season year (for title)
    """
    # Try using new facade API first
    try:
        from pathlib import Path
        from src.facade.qb_api import render_qb_comparison
        render_qb_comparison(
            qb_stats_df,
            Path(output_path),
            season,
            strategy="bars",
            options={
                'title': "Quarterback Playoff Performance Comparison (Playoffs Only)",
                'show_values': True
            }
        )
        return
    except Exception as e:
        logger.debug(f"Facade API not available, using legacy implementation: {e}")
    
    if qb_stats_df.empty:
        logger.warning("No QB stats provided for bar chart")
        return
    
    # Define metrics for bar chart (must match normalized columns)
    metrics = [
        'epa_per_play_norm',
        'completion_pct_norm',
        'td_rate_norm',
        'int_rate_norm',  # Already inverted in normalization
        'sack_rate_norm',  # Already inverted in normalization
        'win_rate_norm'
    ]
    
    # Metric labels for display
    metric_labels = [
        'EPA/Play',
        'Completion %',
        'TD Rate',
        'INT Avoidance',
        'Sack Avoidance',
        'Win Rate'
    ]
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in qb_stats_df.columns]
    available_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in qb_stats_df.columns]
    
    if not available_metrics:
        logger.warning("No normalized metrics found in QB stats")
        return
    
    # Get OpenAI suggestions for improvements
    openai_suggestions = get_visualization_improvements_from_openai("bar")
    if openai_suggestions:
        logger.info("Using OpenAI suggestions for bar chart improvements")
    
    # Create figure with better styling
    setup_plot_style('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    apply_professional_styling(fig, ax, background_color='#F8F9FA')
    
    # Get team colors for each QB
    n_qbs = len(qb_stats_df)
    colors = [get_team_color(qb_row.get('team', '?')) for _, qb_row in qb_stats_df.iterrows()]
    
    # Prepare data for grouped bars
    y_pos = np.arange(len(available_labels))
    bar_width = 0.13  # Slightly wider bars
    spacing = 0.02  # Spacing between bars in a group
    
    # Plot bars for each QB
    for qb_idx, (_, qb_row) in enumerate(qb_stats_df.iterrows()):
        qb_name = qb_row.get('player_name', 'Unknown')
        qb_team = qb_row.get('team', '?')
        # Shorten long names for better display
        if len(qb_name) > 15:
            qb_name = qb_name.split()[0] + ' ' + qb_name.split()[-1][0] + '.'
        label = f"{qb_name} ({qb_team})"
        
        values = [qb_row[m] for m in available_metrics]
        # Center the group and offset each bar
        offset = (qb_idx - (n_qbs - 1) / 2) * (bar_width + spacing)
        
        # Apply better styling with enhanced colors
        bars = ax.barh(y_pos + offset, values, bar_width, 
                      label=label, color=colors[qb_idx], alpha=0.9, 
                      edgecolor='white', linewidth=1.5, zorder=3)
        
        # Add value labels on bars with better contrast
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.05:  # Only label if bar is large enough
                # Use white text on dark bars, black on light bars
                text_color = 'white' if sum(plt.matplotlib.colors.to_rgb(colors[qb_idx])) < 1.5 else 'black'
                ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                       f'{val:.2f}', va='center', fontsize=9, 
                       fontweight='bold', color=text_color, zorder=4)
    
    # Set y-axis with better styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(available_labels, fontsize=12, fontweight='bold', color='#333333')
    ax.invert_yaxis()  # Top metric at top
    
    # Set x-axis with better styling
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Normalized Performance (0-1 scale)', fontsize=13, fontweight='bold', color='#333333', labelpad=10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8, color='#CCCCCC', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Add title with better styling
    title = f"Quarterback Playoff Performance Comparison (Playoffs Only)"
    ax.set_title(title, size=18, fontweight='bold', pad=25, color='#1a1a1a')
    season_text = f"Season {season}"
    ax.text(0.5, 1.02, season_text, transform=ax.transAxes, 
           ha='center', fontsize=14, style='italic', color='#666666')
    
    # Add subtitle
    subtitle = "Metrics aggregated from playoff games used in 10,000-run Monte Carlo simulation"
    fig.text(0.5, 0.97, subtitle, ha='center', fontsize=11, style='italic', color='#666666')
    
    # Add legend with better styling and shadow
    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.98, 
                      edgecolor='#CCCCCC', facecolor='white', frameon=True,
                      shadow=True, fancybox=True)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_boxstyle('round,pad=0.5')
    for text in legend.get_texts():
        text.set_color('#333333')
        text.set_fontweight('bold')
    
    # Add note with better styling
    note_text = f"Minimum 3 playoff games required | Showing {n_qbs} QBs"
    fig.text(0.5, 0.015, note_text, ha='center', fontsize=9, style='italic', color='#888888')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure
    if save_figure_safe(fig, output_path, dpi=300):
        logger.info(f"Saved QB playoff bar chart to {output_path}")
    
    plt.close()


def create_qb_playoff_radar(qb_stats_df: pd.DataFrame, output_path: str, 
                            season: int, max_qbs: int = 3) -> None:
    """
    Create a radar chart showing QB playoff performance metrics.
    
    Note: This function now uses the new facade API internally.
    For new code, prefer using src.facade.qb_api.render_qb_comparison() with strategy='radar'.
    """
    """
    Create a radar chart showing QB playoff performance metrics.
    
    Parameters:
    -----------
    qb_stats_df : pd.DataFrame
        DataFrame with normalized QB stats (from normalize_qb_metrics)
    output_path : str
        Path to save the image
    season : int
        Season year (for title)
    max_qbs : int
        Maximum number of QBs to display (default: 6)
    """
    # Use new facade API
    from pathlib import Path
    from src.facade.qb_api import render_qb_comparison
    
    try:
        render_qb_comparison(
            qb_stats_df,
            Path(output_path),
            season,
            strategy="radar",
            options={
                'title': "Top Quarterback Playoff Profiles (Head-to-Head Comparison)",
                'max_qbs': max_qbs
            }
        )
    except Exception as e:
        logger.error(f"Error using facade API, falling back to legacy implementation: {e}")
        # Fallback to legacy implementation if facade fails
        _create_qb_playoff_radar_legacy(qb_stats_df, output_path, season, max_qbs)


def _create_qb_playoff_radar_legacy(qb_stats_df: pd.DataFrame, output_path: str, 
                            season: int, max_qbs: int = 3) -> None:
    """Legacy implementation (kept for backward compatibility)."""
    if qb_stats_df.empty:
        logger.warning("No QB stats provided for radar chart")
        return
    
    # Select top QBs
    if len(qb_stats_df) > max_qbs:
        if 'win_rate' in qb_stats_df.columns:
            qb_stats_df = qb_stats_df.sort_values('win_rate', ascending=False)
        elif 'epa_per_play' in qb_stats_df.columns:
            qb_stats_df = qb_stats_df.sort_values('epa_per_play', ascending=False)
        qb_stats_df = qb_stats_df.head(max_qbs)
    
    # Define metrics for radar chart (must match normalized columns)
    metrics = [
        'epa_per_play_norm',
        'completion_pct_norm',
        'td_rate_norm',
        'int_rate_norm',  # Already inverted in normalization
        'sack_rate_norm',  # Already inverted in normalization
        'win_rate_norm'
    ]
    
    # Metric labels for display
    metric_labels = [
        'EPA/Play',
        'Completion %',
        'TD Rate',
        'INT Avoidance',  # Inverted INT rate
        'Sack Avoidance',  # Inverted sack rate
        'Win Rate'
    ]
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in qb_stats_df.columns]
    available_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in qb_stats_df.columns]
    
    if not available_metrics:
        logger.warning("No normalized metrics found in QB stats")
        return
    
    # Number of metrics
    N = len(available_metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Limit to max 2-3 QBs for readability
    if len(qb_stats_df) > max_qbs:
        # Sort by win_rate or epa_per_play (use original, not normalized)
        if 'win_rate' in qb_stats_df.columns:
            qb_stats_df = qb_stats_df.sort_values('win_rate', ascending=False)
        elif 'epa_per_play' in qb_stats_df.columns:
            qb_stats_df = qb_stats_df.sort_values('epa_per_play', ascending=False)
        qb_stats_df = qb_stats_df.head(max_qbs)
    
    # Get OpenAI suggestions for improvements
    openai_suggestions = get_visualization_improvements_from_openai("radar")
    if openai_suggestions:
        logger.info("Using OpenAI suggestions for radar chart improvements")
    
    # Create figure with better styling
    setup_plot_style('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))
    apply_professional_styling(fig, ax, background_color='#F8F9FA')
    
    # Get team colors for each QB
    colors = [get_team_color(qb_row.get('team', '?')) for _, qb_row in qb_stats_df.iterrows()]
    
    # Plot each QB (lines only, no fills) with better styling
    for idx, (_, qb_row) in enumerate(qb_stats_df.iterrows()):
        qb_name = qb_row.get('player_name', 'Unknown')
        qb_team = qb_row.get('team', '?')
        # Shorten long names
        if len(qb_name) > 15:
            qb_name = qb_name.split()[0] + ' ' + qb_name.split()[-1][0] + '.'
        
        values = [qb_row[m] for m in available_metrics]
        values += values[:1]  # Complete the circle
        
        # Plot line with better styling and subtle glow effect
        ax.plot(angles, values, 'o-', linewidth=3.5, 
                label=f"{qb_name} ({qb_team})",
                color=colors[idx], markersize=11, markerfacecolor=colors[idx],
                markeredgecolor='white', markeredgewidth=2.0, zorder=3,
                alpha=0.95)
        
        # Add subtle glow by plotting a slightly thicker line behind
        ax.plot(angles, values, '-', linewidth=5.0, 
                color=colors[idx], alpha=0.15, zorder=2)
    
    # Add metric labels with better styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=13, fontweight='bold', color='#333333')
    
    # Set y-axis limits (0-1 for normalized metrics) with better styling
    ax.set_ylim(0, 1.1)
    # Major gridlines only with better styling
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='#666666')
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=1.2, color='#CCCCCC', zorder=1)
    
    # Add title with better styling
    title = f"Top Quarterback Playoff Profiles (Head-to-Head Comparison)"
    ax.set_title(title, size=18, fontweight='bold', pad=30, color='#1a1a1a')
    season_text = f"Season {season}"
    ax.text(0.5, 1.05, season_text, transform=ax.transAxes, 
           ha='center', fontsize=14, style='italic', color='#666666')
    
    # Add subtitle
    subtitle = "Metrics aggregated from playoff games used in 10,000-run Monte Carlo simulation"
    fig.text(0.5, 0.96, subtitle, ha='center', fontsize=11, style='italic', color='#666666')
    
    # Add legend (outside plot area) with better styling and shadow
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), 
                      fontsize=11, framealpha=0.98, edgecolor='#CCCCCC', 
                      facecolor='white', frameon=True, shadow=True, fancybox=True)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_boxstyle('round,pad=0.5')
    for text in legend.get_texts():
        text.set_color('#333333')
        text.set_fontweight('bold')
    
    # Add note about minimum games with better styling
    note_text = f"Minimum 3 playoff games required | Showing top {len(qb_stats_df)} QBs"
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic', color='#888888')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    if save_figure_safe(fig, output_path, dpi=300):
        logger.info(f"Saved QB playoff radar chart to {output_path}")
    
    plt.close()


def create_qb_playoff_radar_multiple(qb_stats_df: pd.DataFrame, output_path: str,
                                     season: int, qbs_per_chart: int = 3) -> None:
    """
    Create multiple radar charts with fewer QBs per chart for better readability.
    
    Parameters:
    -----------
    qb_stats_df : pd.DataFrame
        DataFrame with normalized QB stats
    output_path : str
        Base path for output (will add _chart1.png, _chart2.png, etc.)
    season : int
        Season year
    qbs_per_chart : int
        Number of QBs per chart (default: 3)
    """
    if qb_stats_df.empty:
        logger.warning("No QB stats provided")
        return
    
    # Split into chunks
    n_charts = (len(qb_stats_df) + qbs_per_chart - 1) // qbs_per_chart
    
    for chart_idx in range(n_charts):
        start_idx = chart_idx * qbs_per_chart
        end_idx = min(start_idx + qbs_per_chart, len(qb_stats_df))
        chart_df = qb_stats_df.iloc[start_idx:end_idx]
        
        # Modify output path
        output_path_obj = Path(output_path)
        chart_output = output_path_obj.parent / f"{output_path_obj.stem}_chart{chart_idx + 1}{output_path_obj.suffix}"
        
        # Create chart for this subset
        create_qb_playoff_radar(chart_df, str(chart_output), season, max_qbs=qbs_per_chart)


def get_visualization_improvements_from_openai(chart_type: str = "bar") -> Optional[Dict]:
    """
    Use OpenAI API to get suggestions for improving QB playoff visualizations.
    
    Parameters:
    -----------
    chart_type : str
        Type of chart ('bar' or 'radar')
        
    Returns:
    --------
    Dict with improvement suggestions, or None if OpenAI unavailable
    """
    if not OPENAI_AVAILABLE:
        logger.debug("OpenAI not available for visualization improvements")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI_API_KEY not found in environment")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a data visualization expert. Analyze and provide specific, actionable improvements for an NFL quarterback playoff performance {chart_type} chart.

Current implementation:
- Uses matplotlib for visualization
- Shows QB playoff stats: EPA/Play, Completion %, TD Rate, INT Avoidance, Sack Avoidance, Win Rate
- Normalized metrics (0-1 scale)
- Team colors for each QB
- Horizontal grouped bars (for bar chart) or radar/spider chart (for radar chart)

Provide specific, implementable suggestions for:
1. Color scheme improvements (better contrast, accessibility)
2. Typography and text styling
3. Layout and spacing optimizations
4. Visual hierarchy enhancements
5. Data presentation improvements (annotations, labels)
6. Professional polish (shadows, gradients, modern design elements)

Format your response as actionable code suggestions that can be directly implemented in matplotlib.
Focus on practical improvements that enhance readability and visual appeal.
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert specializing in sports analytics and matplotlib visualizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        suggestions = response.choices[0].message.content
        logger.info("Received visualization improvement suggestions from OpenAI")
        
        return {
            'suggestions': suggestions,
            'chart_type': chart_type
        }
        
    except Exception as e:
        logger.warning(f"Error getting OpenAI suggestions: {e}")
        return None


def get_visualization_improvements_from_openai(chart_type: str = "bar") -> Optional[Dict]:
    """
    Use OpenAI API to get suggestions for improving QB playoff visualizations.
    
    Parameters:
    -----------
    chart_type : str
        Type of chart ('bar' or 'radar')
        
    Returns:
    --------
    Dict with improvement suggestions, or None if OpenAI unavailable
    """
    if not OPENAI_AVAILABLE:
        logger.debug("OpenAI not available for visualization improvements")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI_API_KEY not found in environment")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a data visualization expert. Analyze and provide specific, actionable improvements for an NFL quarterback playoff performance {chart_type} chart.

Current implementation:
- Uses matplotlib for visualization
- Shows QB playoff stats: EPA/Play, Completion %, TD Rate, INT Avoidance, Sack Avoidance, Win Rate
- Normalized metrics (0-1 scale)
- Team colors for each QB
- Horizontal grouped bars (for bar chart) or radar/spider chart (for radar chart)

Provide specific, implementable suggestions for:
1. Color scheme improvements (better contrast, accessibility)
2. Typography and text styling
3. Layout and spacing optimizations
4. Visual hierarchy enhancements
5. Data presentation improvements (annotations, labels)
6. Professional polish (shadows, gradients, modern design elements)

Format your response as actionable code suggestions that can be directly implemented in matplotlib.
Focus on practical improvements that enhance readability and visual appeal.
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert specializing in sports analytics and matplotlib visualizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        suggestions = response.choices[0].message.content
        logger.info("Received visualization improvement suggestions from OpenAI")
        
        return {
            'suggestions': suggestions,
            'chart_type': chart_type
        }
        
    except Exception as e:
        logger.warning(f"Error getting OpenAI suggestions: {e}")
        return None

