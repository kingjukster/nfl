"""
Connector drawing utilities.

Orthogonal connectors with explicit endpoints (no inference).
"""

from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from src.layout.layout_model import Box, Connector


def _elbow_path(points: List[Tuple[float, float]]) -> MplPath:
    """Create path from points."""
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    return MplPath(points, codes)


def draw_elbow_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    primary: str = 'h',
    elbow: float = 0.030,
    head: bool = True,
    shrink: float = 0.010,
    color: str = '#666666',
    linewidth: float = 1.5,
    zorder: int = 1,
) -> None:
    """
    Draw structured 90° connector with arrowhead.
    
    Uses shrink so it stops before box edges (avoids drawing into boxes).
    
    Args:
        ax: Matplotlib axes
        start: (x, y) start point
        end: (x, y) end point
        primary: 'h' for horizontal-first, 'v' for vertical-first
        elbow: Elbow offset distance
        head: Whether to show arrowhead
        shrink: Shrink distance from endpoints
        color: Line color
        linewidth: Line width
        zorder: Z-order for layering
    """
    x0, y0 = start
    x3, y3 = end
    
    if primary == "h":
        # Horizontal-first: horizontal → vertical → horizontal
        if x3 >= x0:
            mx = x0 + elbow
        else:
            mx = x0 - elbow
        pts = [(x0, y0), (mx, y0), (mx, y3), (x3, y3)]
    else:
        # Vertical-first: vertical → horizontal → vertical
        if y3 >= y0:
            my = y0 + elbow
        else:
            my = y0 - elbow
        pts = [(x0, y0), (x0, my), (x3, my), (x3, y3)]
    
    path = _elbow_path(pts)
    arrowstyle = "-|>" if head else "-"
    
    patch = patches.FancyArrowPatch(
        path=path,
        arrowstyle=arrowstyle,
        mutation_scale=9,
        linewidth=linewidth,
        color=color,
        shrinkA=shrink,
        shrinkB=shrink,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _h_port(box: Box, side: str, pad: float = 0.010) -> Tuple[float, float]:
    """Get horizontal port just outside left/right edge."""
    if side == "right":
        return (box.x_right + pad, box.y_center)
    return (box.x_left - pad, box.y_center)


def _v_port(box: Box, side: str, pad: float = 0.010) -> Tuple[float, float]:
    """Get vertical port just outside top/bottom edge."""
    if side == "bottom":
        return (box.center[0], box.y_bottom - pad)
    return (box.center[0], box.y_top + pad)


def draw_connector(
    ax,
    connector: Connector,
    src_box: Box,
    dst_box: Box,
) -> None:
    """
    Draw a connector between two boxes.
    
    Uses explicit endpoints from LayoutModel (no inference).
    
    Args:
        ax: Matplotlib axes
        connector: Connector from LayoutModel
        src_box: Source box
        dst_box: Destination box
    """
    # Determine connection points based on style
    if connector.style == 'elbow_h':
        # Horizontal-first: connect right edge of src to left edge of dst (or vice versa)
        if src_box.x_left < dst_box.x_left:
            start = _h_port(src_box, "right")
            end = _h_port(dst_box, "left")
        else:
            start = _h_port(src_box, "left")
            end = _h_port(dst_box, "right")
        primary = 'h'
    elif connector.style == 'elbow_v':
        # Vertical-first: connect bottom of src to top of dst (or vice versa)
        if src_box.y_center < dst_box.y_center:
            start = _v_port(src_box, "bottom")
            end = _v_port(dst_box, "top")
        else:
            start = _v_port(src_box, "top")
            end = _v_port(dst_box, "bottom")
        primary = 'v'
    else:
        # Straight line (fallback)
        start = src_box.center
        end = dst_box.center
        primary = 'h'
    
    draw_elbow_arrow(
        ax,
        start,
        end,
        primary=primary,
        color=connector.color,
        linewidth=connector.linewidth,
        zorder=connector.zorder,
    )

