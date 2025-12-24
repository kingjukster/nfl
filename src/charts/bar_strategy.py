"""
Grouped bar chart strategy.

Best for: "Who is best per metric?" question.
Default strategy for QB comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.charts.base import ChartStrategy
from src.models.qb_metrics import QBComparisonDataset
from src.constants import get_team_color
from src.visualization import setup_plot_style, apply_professional_styling


class GroupedBarStrategy(ChartStrategy):
    """
    Grouped horizontal bar chart strategy.
    
    Shows all QBs side-by-side for each metric.
    Best for comparing "who is best per metric?"
    """
    
    def make_figure(
        self,
        data: QBComparisonDataset,
        theme: str = 'default',
        options: Dict[str, Any] = None
    ) -> plt.Figure:
        """
        Create grouped bar chart figure.
        
        Args:
            data: QBComparisonDataset with normalized metrics
            theme: Theme name (default: 'seaborn-v0_8-darkgrid')
            options: Optional dict with:
                - figsize: Tuple (width, height)
                - title: Chart title
                - show_values: bool (show values on bars)
        """
        if options is None:
            options = {}
        
        # Setup styling
        theme_name = theme if theme != 'default' else 'seaborn-v0_8-darkgrid'
        setup_plot_style(theme_name)
        
        figsize = options.get('figsize', (16, 10))
        fig, ax = plt.subplots(figsize=figsize)
        apply_professional_styling(fig, ax, background_color='#F8F9FA')
        
        # Get metrics
        metrics = data.metrics
        metric_labels = self._get_metric_labels(metrics)
        
        # Prepare data
        n_qbs = data.n_qbs
        y_pos = np.arange(len(metrics))
        bar_width = 0.13
        spacing = 0.02
        
        # Get team colors
        colors = []
        for _, row in data.data.iterrows():
            team = row.get('team', '?')
            colors.append(get_team_color(team))
        
        # Plot bars for each QB
        for qb_idx, (_, qb_row) in enumerate(data.data.iterrows()):
            qb_name = qb_row.get('player_name', 'Unknown')
            qb_team = qb_row.get('team', '?')
            
            # Shorten long names
            if len(qb_name) > 15:
                qb_name = qb_name.split()[0] + ' ' + qb_name.split()[-1][0] + '.'
            label = f"{qb_name} ({qb_team})"
            
            values = [qb_row[m] for m in metrics]
            offset = (qb_idx - (n_qbs - 1) / 2) * (bar_width + spacing)
            
            bars = ax.barh(
                y_pos + offset,
                values,
                bar_width,
                label=label,
                color=colors[qb_idx],
                alpha=0.9,
                edgecolor='white',
                linewidth=1.5,
                zorder=3
            )
            
            # Add value labels if requested
            if options.get('show_values', True):
                for i, (bar, val) in enumerate(zip(bars, values)):
                    if val > 0.05:
                        text_color = 'white' if sum(plt.matplotlib.colors.to_rgb(colors[qb_idx])) < 1.5 else 'black'
                        ax.text(
                            val + 0.015,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:.2f}',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color=text_color,
                            zorder=4
                        )
        
        # Set y-axis
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_labels, fontsize=12, fontweight='bold', color='#333333')
        ax.invert_yaxis()
        
        # Set x-axis
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Normalized Performance (0-1 scale)', fontsize=13, fontweight='bold', color='#333333', labelpad=10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8, color='#CCCCCC', zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Title
        title = options.get('title', f"Quarterback Playoff Performance Comparison (Playoffs Only)")
        ax.set_title(title, size=18, fontweight='bold', pad=25, color='#1a1a1a')
        season_text = f"Season {data.season}"
        ax.text(0.5, 1.02, season_text, transform=ax.transAxes, ha='center', fontsize=14, style='italic', color='#666666')
        
        # Legend
        legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.98, edgecolor='#CCCCCC', facecolor='white', frameon=True, shadow=True, fancybox=True)
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_boxstyle('round,pad=0.5')
        for text in legend.get_texts():
            text.set_color('#333333')
            text.set_fontweight('bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        return fig
    
    @staticmethod
    def _get_metric_labels(metrics: List[str]) -> List[str]:
        """Get display labels for metrics."""
        label_map = {
            'epa_per_play_norm': 'EPA/Play',
            'completion_pct_norm': 'Completion %',
            'td_rate_norm': 'TD Rate',
            'int_rate_norm': 'INT Avoidance',
            'sack_rate_norm': 'Sack Avoidance',
            'win_rate_norm': 'Win Rate'
        }
        return [label_map.get(m, m.replace('_norm', '').replace('_', ' ').title()) for m in metrics]

