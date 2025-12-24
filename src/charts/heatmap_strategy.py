"""
Heatmap chart strategy.

Best for: Win probability matrices showing team vs team matchups.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.charts.base import ChartStrategy
from src.visualization import setup_plot_style, apply_professional_styling


class HeatmapStrategy(ChartStrategy):
    """
    Heatmap strategy for win probability matrices.
    
    Shows team vs team win probabilities as a heatmap.
    """
    
    def make_figure(
        self,
        data: np.ndarray,  # 2D array of probabilities
        theme: str = 'default',
        options: Dict[str, Any] = None
    ) -> plt.Figure:
        """
        Create heatmap figure.
        
        Args:
            data: 2D numpy array of win probabilities (n_teams x n_teams)
            theme: Theme name
            options: Optional dict with:
                - figsize: Tuple (width, height)
                - title: Chart title
                - team_labels: List of team names
                - cmap: Colormap name
        """
        if options is None:
            options = {}
        
        # Setup styling
        theme_name = theme if theme != 'default' else 'seaborn-v0_8-whitegrid'
        setup_plot_style(theme_name)
        
        figsize = options.get('figsize', (14, 12))
        fig, ax = plt.subplots(figsize=figsize)
        apply_professional_styling(fig, ax, background_color='#F8F9FA')
        
        # Get team labels
        team_labels = options.get('team_labels', [f'Team {i+1}' for i in range(data.shape[0])])
        cmap = options.get('cmap', 'RdYlGn')
        
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        
        # Set ticks and labels
        n = len(team_labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(team_labels, rotation=45, ha='right')
        ax.set_yticklabels(team_labels)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(
                    j, i, f'{data[i, j]:.2f}',
                    ha="center", va="center",
                    color="black", fontsize=8
                )
        
        # Title
        title = options.get('title', 'Win Probability Heatmap')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Opponent Team", fontsize=12)
        ax.set_ylabel("Team", fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("P(Row Team Wins)", fontsize=11)
        
        plt.tight_layout()
        
        return fig

