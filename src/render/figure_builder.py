"""
Figure Builder (Builder Pattern).

Step-by-step figure construction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from src.layout.layout_model import LayoutModel, Box
from src.render.connectors import draw_connector


class LogoCache:
    """Cached logo loader."""
    
    def __init__(self, logo_dir: Optional[Path] = None, max_px: int = 64):
        self.logo_dir = Path(logo_dir) if logo_dir else None
        self.max_px = int(max_px)
        self._cache: Dict[str, Optional[np.ndarray]] = {}
    
    @staticmethod
    def _norm_key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())
    
    def _resolve_logo_path(self, team_name: str) -> Optional[Path]:
        if not self.logo_dir:
            return None
        
        key = self._norm_key(team_name)
        candidates = [
            self.logo_dir / f"{key}.gif",
            self.logo_dir / f"{key}.png",
            self.logo_dir / f"{team_name}.gif",
            self.logo_dir / f"{team_name}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        
        for p in self.logo_dir.glob("*"):
            if p.is_file() and key in self._norm_key(p.stem):
                return p
        
        return None
    
    def _load_first_frame_rgba(self, path: Path) -> np.ndarray:
        try:
            from PIL import Image, ImageSequence
            img = Image.open(path)
            frame = next(ImageSequence.Iterator(img)).convert("RGBA")
            w, h = frame.size
            scale = min(self.max_px / max(w, h), 1.0)
            if scale < 1.0:
                frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return np.asarray(frame)
        except Exception:
            return None
    
    def get(self, team_name: str) -> Optional[np.ndarray]:
        """Get logo array for team (cached)."""
        key = self._norm_key(team_name)
        if key in self._cache:
            return self._cache[key]
        
        path = self._resolve_logo_path(team_name)
        if not path:
            self._cache[key] = None
            return None
        
        arr = self._load_first_frame_rgba(path)
        self._cache[key] = arr
        return arr


class FigureBuilder:
    """
    Builder for figure construction.
    
    Step-by-step construction: canvas → background → boxes → connectors → annotations → export
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.logo_cache = None
    
    def create_canvas(
        self,
        figsize: tuple = (14, 10),
        theme: str = 'default',
        logo_dir: Optional[Path] = None
    ) -> 'FigureBuilder':
        """Create matplotlib figure and axes."""
        self.fig = plt.figure(figsize=figsize, dpi=140)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        
        if logo_dir:
            self.logo_cache = LogoCache(logo_dir)
        
        return self
    
    def draw_background(self, color: str = 'white') -> 'FigureBuilder':
        """Draw background."""
        if self.ax:
            self.ax.set_facecolor(color)
        return self
    
    def draw_boxes(
        self,
        layout_model: LayoutModel,
        box_w: float = 0.195,
        box_h: float = 0.043,
        bar_w: float = 0.010
    ) -> 'FigureBuilder':
        """Draw all team boxes from layout model."""
        if not self.ax:
            raise ValueError("Must create canvas first")
        
        for box in layout_model.boxes.values():
            self._draw_single_box(box, box_w, box_h, bar_w)
        
        return self
    
    def _draw_single_box(
        self,
        box: Box,
        box_w: float,
        box_h: float,
        bar_w: float
    ) -> None:
        """Draw a single team box."""
        y_bottom = box.y_center - box_h / 2
        
        # Box styling based on winner status
        if box.is_winner:
            edge = "#222222"
            lw = 1.8
            face = "#f4f6fb"
        else:
            edge = "#666666"
            lw = 1.0
            face = "white"
        
        # Main box
        rect = patches.FancyBboxPatch(
            (box.x_left, y_bottom),
            box_w,
            box_h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        self.ax.add_patch(rect)
        
        # Accent bar (conference color)
        if box.conference == "AFC":
            bar_x = box.x_left
            bar_color = "#d61f2c"
        else:
            bar_x = box.x_left + box_w - bar_w
            bar_color = "#1f4fd6"
        
        self.ax.add_patch(patches.Rectangle(
            (bar_x, y_bottom),
            bar_w,
            box_h,
            linewidth=0,
            facecolor=bar_color,
            zorder=3
        ))
        
        # Logo
        if self.logo_cache:
            logo = self.logo_cache.get(box.team_name)
            if logo is not None:
                if box.conference == "AFC":
                    lx = box.x_left + bar_w + 0.018
                else:
                    lx = box.x_left + box_w - bar_w - 0.018
                oi = OffsetImage(logo, zoom=0.30)
                ab = AnnotationBbox(oi, (lx, box.y_center), frameon=False, box_alignment=(0.5, 0.5))
                self.ax.add_artist(ab)
        
        # Label
        label = f"{box.team_name.upper()} ({box.seed})"
        if box.conference == "AFC":
            tx = box.x_left + bar_w + 0.040
            ha = "left"
        else:
            tx = box.x_left + box_w - bar_w - 0.040
            ha = "right"
        
        self.ax.text(
            tx, box.y_center, label,
            ha=ha, va="center",
            fontsize=8.3, fontweight="bold",
            color="#111111", zorder=4
        )
        
        # Winner marker
        if box.is_winner:
            if box.conference == "AFC":
                wx = box.x_left + box_w - 0.020
                ha2 = "right"
            else:
                wx = box.x_left + 0.020
                ha2 = "left"
            self.ax.text(
                wx, box.y_center, "W",
                ha=ha2, va="center",
                fontsize=8.5, fontweight="bold",
                color="#111111", zorder=5
            )
    
    def draw_connectors(self, layout_model: LayoutModel) -> 'FigureBuilder':
        """Draw all connectors from layout model."""
        if not self.ax:
            raise ValueError("Must create canvas first")
        
        for connector in layout_model.connectors:
            src_box = layout_model.get_box(connector.src_id)
            dst_box = layout_model.get_box(connector.dst_id)
            if src_box and dst_box:
                draw_connector(self.ax, connector, src_box, dst_box)
        
        return self
    
    def annotate_winners(self, layout_model: LayoutModel) -> 'FigureBuilder':
        """Add annotations (labels) from layout model."""
        if not self.ax:
            raise ValueError("Must create canvas first")
        
        for label in layout_model.annotations:
            self.ax.text(
                label.x, label.y, label.text,
                fontsize=label.fontsize,
                fontweight=label.fontweight,
                color=label.color,
                ha=label.ha,
                va=label.va,
                zorder=label.zorder
            )
        
        return self
    
    def export(self, out_path: Path, dpi: int = 300) -> None:
        """Export figure to file."""
        if not self.fig:
            raise ValueError("Must create canvas first")
        
        self.fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(self.fig)

