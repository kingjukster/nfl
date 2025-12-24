"""
Team Visualization Utilities

Functions for working with team colors and logos in visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import logging

try:
    from PIL import Image, ImageSequence
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageSequence = None

from src.constants import get_team_color as _get_team_color, get_team_logo_path
from src.config import data_config

logger = logging.getLogger(__name__)


def get_team_color(team: str) -> str:
    """
    Get team color for visualization.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
        
    Returns:
    --------
    str
        Hex color code
    """
    return _get_team_color(team)


def load_team_logo(team: str, size: Optional[tuple] = None) -> Optional[np.ndarray]:
    """
    Load team logo as numpy array for matplotlib.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
    size : tuple, optional
        Desired size (width, height) in pixels
        
    Returns:
    --------
    np.ndarray or None
        Logo as numpy array, or None if not found
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot load logos")
        return None
    
    logo_path = get_team_logo_path(team)
    if not logo_path or not logo_path.exists():
        logger.debug(f"Logo not found for {team}")
        return None
    
    try:
        img = Image.open(logo_path)
        
        # Handle GIF animations (take first frame)
        if hasattr(img, 'is_animated') and img.is_animated:
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            img = frames[0] if frames else img
        
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize if requested
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        
        return np.array(img)
    except Exception as e:
        logger.warning(f"Error loading logo for {team}: {e}")
        return None


def apply_team_colors(teams: List[str]) -> List[str]:
    """
    Get list of team colors for a list of teams.
    
    Parameters:
    -----------
    teams : list of str
        List of team abbreviations
        
    Returns:
    --------
    list of str
        List of hex color codes
    """
    return [get_team_color(team) for team in teams]

