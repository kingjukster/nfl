"""
Bracket rendering facade.

Clean public API for bracket visualization.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from src.models.bracket import FrozenBracket
from src.adapters.simulation_adapter import SimulationOutputAdapter
from src.layout.bracket_layout import compute_bracket_layout
from src.render.bracket_renderer import BracketRenderer
from src.validation import validate_frozen_bracket


def render_bracket(
    bracket: FrozenBracket,
    out_path: Path,
    season: int,
    conference: str,
    logo_dir: Optional[Path] = None,
    options: Dict[str, Any] = None
) -> None:
    """
    Render bracket diagram.
    
    Clean public API for bracket visualization.
    
    Args:
        bracket: FrozenBracket to render
        out_path: Output file path
        season: Season year
        conference: 'AFC' or 'NFC'
        logo_dir: Optional directory containing team logos
        options: Optional rendering options:
            - figsize: Tuple (width, height)
            - dpi: Output DPI
    """
    if options is None:
        options = {}
    
    # Validate bracket
    validate_frozen_bracket(bracket)
    
    # Compute layout
    layout_model = compute_bracket_layout(bracket, conference)
    
    # Render
    renderer = BracketRenderer(logo_dir=logo_dir)
    figsize = options.get('figsize', (14, 10))
    dpi = options.get('dpi', 300)
    
    renderer.render(layout_model, out_path, season, figsize=figsize, dpi=dpi)


def render_bracket_from_simulation(
    simulation_output: Dict[str, Any],
    out_path: Path,
    season: int,
    conference: str,
    logo_dir: Optional[Path] = None,
    options: Dict[str, Any] = None
) -> None:
    """
    Render bracket from simulation output.
    
    Convenience function that adapts simulation output to bracket.
    
    Args:
        simulation_output: Output from simulate_full_playoffs()
        out_path: Output file path
        season: Season year
        conference: 'AFC' or 'NFC'
        logo_dir: Optional directory containing team logos
        options: Optional rendering options
    """
    # Adapt simulation output to FrozenBracket
    adapter = SimulationOutputAdapter()
    bracket = adapter.to_frozen_bracket(simulation_output, conference)
    
    # Render
    render_bracket(bracket, out_path, season, conference, logo_dir, options)

