"""
Radar chart strategy.

Best for: "What is each QB's profile?" or "Matchup profile comparison?"
Use for 2-3 QBs head-to-head comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.charts.base import ChartStrategy
from src.models.qb_metrics import QBComparisonDataset
from src.constants import get_team_color
from src.visualization import setup_plot_style, apply_professional_styling


class RadarStrategy(ChartStrategy):
    """
    Radar/spider chart strategy.
    
    Shows QB profiles as radar charts.
    Best for comparing 2-3 QBs head-to-head.
    """
    
    def make_figure(
        self,
        data: QBComparisonDataset,
        theme: str = 'default',
        options: Dict[str, Any] = None
    ) -> plt.Figure:
        """
        Create radar chart figure.
        
        Args:
            data: QBComparisonDataset (should be limited to 2-3 QBs)
            theme: Theme name
            options: Optional dict with:
                - figsize: Tuple (width, height)
                - title: Chart title
                - max_qbs: Maximum QBs to show (default: 3)
        """
        if options is None:
            options = {}
        
        max_qbs = options.get('max_qbs', 3)
        
        # Limit to top QBs if needed
        qb_data = data.data.copy()
        if len(qb_data) > max_qbs:
            # Sort by win_rate or epa_per_play if available
            if 'win_rate' in qb_data.columns:
                qb_data = qb_data.sort_values('win_rate', ascending=False)
            elif 'epa_per_play' in qb_data.columns:
                qb_data = qb_data.sort_values('epa_per_play', ascending=False)
            qb_data = qb_data.head(max_qbs)
        
        # Setup styling
        theme_name = theme if theme != 'default' else 'seaborn-v0_8-whitegrid'
        setup_plot_style(theme_name)
        
        figsize = options.get('figsize', (14, 12))
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        apply_professional_styling(fig, ax, background_color='#F8F9FA')
        
        # Get metrics
        metrics = data.metrics
        metric_labels = self._get_metric_labels(metrics)
        N = len(metrics)
        
        # Compute angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Get team colors
        colors = []
        for _, row in qb_data.iterrows():
            team = row.get('team', '?')
            colors.append(get_team_color(team))
        
        # Plot each QB
        for idx, (_, qb_row) in enumerate(qb_data.iterrows()):
            qb_name = qb_row.get('player_name', 'Unknown')
            qb_team = qb_row.get('team', '?')
            
            # Shorten long names
            if len(qb_name) > 15:
                qb_name = qb_name.split()[0] + ' ' + qb_name.split()[-1][0] + '.'
            
            values = [qb_row[m] for m in metrics]
            values += values[:1]  # Complete the circle
            
            # Plot line
            ax.plot(
                angles,
                values,
                'o-',
                linewidth=3.5,
                label=f"{qb_name} ({qb_team})",
                color=colors[idx],
                markersize=11,
                markerfacecolor=colors[idx],
                markeredgecolor='white',
                markeredgewidth=2.0,
                zorder=3,
                alpha=0.95
            )
            
            # Subtle glow effect
            ax.plot(
                angles,
                values,
                '-',
                linewidth=5.0,
                color=colors[idx],
                alpha=0.15,
                zorder=2
            )
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=13, fontweight='bold', color='#333333')
        
        # Set y-axis
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='#666666')
        ax.grid(True, alpha=0.5, linestyle='-', linewidth=1.2, color='#CCCCCC', zorder=1)
        
        # Title
        title = options.get('title', f"Top Quarterback Playoff Profiles (Head-to-Head Comparison)")
        ax.set_title(title, size=18, fontweight='bold', pad=30, color='#1a1a1a')
        season_text = f"Season {data.season}"
        ax.text(0.5, 1.05, season_text, transform=ax.transAxes, ha='center', fontsize=14, style='italic', color='#666666')
        
        # Legend
        legend = ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.4, 1.15),
            fontsize=11,
            framealpha=0.98,
            edgecolor='#CCCCCC',
            facecolor='white',
            frameon=True,
            shadow=True,
            fancybox=True
        )
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_boxstyle('round,pad=0.5')
        for text in legend.get_texts():
            text.set_color('#333333')
            text.set_fontweight('bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
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

