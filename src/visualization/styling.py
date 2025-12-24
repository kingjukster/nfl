"""
Visualization Styling Utilities

Common styling functions for matplotlib plots.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def setup_plot_style(style: str = 'seaborn-v0_8-darkgrid', 
                     fallback_styles: Optional[list] = None) -> None:
    """
    Setup matplotlib plot style with fallbacks.
    
    Parameters:
    -----------
    style : str
        Primary style to use
    fallback_styles : list of str, optional
        Fallback styles to try if primary fails
    """
    if fallback_styles is None:
        fallback_styles = ['seaborn-darkgrid', 'default']
    
    try:
        plt.style.use(style)
    except:
        for fallback in fallback_styles:
            try:
                plt.style.use(fallback)
                logger.debug(f"Using fallback style: {fallback}")
                break
            except:
                continue
        else:
            plt.style.use('default')
            logger.warning("Using default matplotlib style")


def apply_professional_styling(fig, ax, title: str = None, 
                              subtitle: str = None,
                              background_color: str = '#F8F9FA') -> None:
    """
    Apply professional styling to a matplotlib figure.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    title : str, optional
        Main title
    subtitle : str, optional
        Subtitle text
    background_color : str
        Background color for figure
    """
    # Set background colors
    fig.patch.set_facecolor('white')
    ax.set_facecolor(background_color)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Add title
    if title:
        ax.set_title(title, size=18, fontweight='bold', pad=25, color='#1a1a1a')
    
    # Add subtitle
    if subtitle:
        fig.text(0.5, 0.97, subtitle, ha='center', fontsize=11, 
                style='italic', color='#666666', transform=fig.transFigure)


def save_figure_safe(fig, output_path: str, dpi: int = 300, 
                    bbox_inches: str = 'tight') -> bool:
    """
    Save figure with error handling.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str
        Path to save figure
    dpi : int
        Resolution for saved figure
    bbox_inches : str
        Bounding box setting
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved figure to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving figure to {output_path}: {e}")
        return False

