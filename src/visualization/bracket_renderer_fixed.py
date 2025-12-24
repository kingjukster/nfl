from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageSequence
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.path import Path as MplPath

# Import new modules
from src.models.bracket import Team, Matchup, FrozenBracket
from src.validation import validate_bracket_dict
from src.layout.bracket_layout import compute_bracket_layout
from src.render.bracket_renderer import BracketRenderer

# Re-export for backward compatibility
__all__ = ['Team', 'Matchup', 'FrozenBracket', 'LogoCache', 'Box', 'freeze_conference', 'render_bracket', 'render_from_dict']


# ============================================================
# Logo utilities (GIF first frame)
# ============================================================

class LogoCache:
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
        img = Image.open(path)
        frame = next(ImageSequence.Iterator(img)).convert("RGBA")
        w, h = frame.size
        scale = min(self.max_px / max(w, h), 1.0)
        if scale < 1.0:
            frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.asarray(frame)

    def get(self, team_name: str) -> Optional[np.ndarray]:
        if not self.logo_dir:
            return None
        key = self._norm_key(team_name)
        if key in self._cache:
            return self._cache[key]

        path = self._resolve_logo_path(team_name)
        if not path:
            self._cache[key] = None
            return None

        try:
            arr = self._load_first_frame_rgba(path)
        except Exception:
            arr = None

        self._cache[key] = arr
        return arr


# ============================================================
# Freezing logic (NO drawing here)
# ============================================================

def _team(name: str, seed: int, conf: str) -> Team:
    return Team(name=str(name), seed=int(seed), conf=str(conf))

def _winner_name_seed(m: Dict[str, Any]) -> Tuple[str, int]:
    return str(m["winner_name"]), int(m["winner_seed"])

def freeze_conference(
    conf: str,
    wc_matchups: List[Dict[str, Any]],
    div_matchups: List[Dict[str, Any]],
    conf_matchup: Dict[str, Any],
) -> FrozenBracket:
    def wc_slot_key(m: Dict[str, Any]) -> int:
        seeds = tuple(sorted([int(m["a_seed"]), int(m["b_seed"])]))
        mapping = {(2, 7): 0, (3, 6): 1, (4, 5): 2}
        return mapping.get(seeds, 99)

    wc_sorted = sorted(wc_matchups, key=wc_slot_key)
    if len(wc_sorted) != 3:
        raise ValueError(f"{conf}: expected 3 WC games, got {len(wc_sorted)}")

    wc_frozen: List[Matchup] = []
    for m in wc_sorted:
        a = _team(m["a_name"], m["a_seed"], conf)
        b = _team(m["b_name"], m["b_seed"], conf)
        wn, ws = _winner_name_seed(m)
        w = _team(wn, ws, conf)
        wc_frozen.append(Matchup(a=a, b=b, winner=w))

    if len(div_matchups) != 2:
        raise ValueError(f"{conf}: expected 2 DIV games, got {len(div_matchups)}")

    def div_slot_key(m: Dict[str, Any]) -> int:
        seeds = [int(m["a_seed"]), int(m["b_seed"])]
        return 0 if 1 in seeds else 1

    div_sorted = sorted(div_matchups, key=div_slot_key)
    div_frozen: List[Matchup] = []
    for m in div_sorted:
        a = _team(m["a_name"], m["a_seed"], conf)
        b = _team(m["b_name"], m["b_seed"], conf)
        wn, ws = _winner_name_seed(m)
        w = _team(wn, ws, conf)
        div_frozen.append(Matchup(a=a, b=b, winner=w))

    cm = conf_matchup
    a = _team(cm["a_name"], cm["a_seed"], conf)
    b = _team(cm["b_name"], cm["b_seed"], conf)
    wn, ws = _winner_name_seed(cm)
    w = _team(wn, ws, conf)
    conf_frozen = Matchup(a=a, b=b, winner=w)

    return FrozenBracket(
        wc=wc_frozen,
        div=div_frozen,
        conf=conf_frozen,
        champ=conf_frozen.winner,
    )


# ============================================================
# Rendering
# ============================================================

@dataclass
class Box:
    x: float
    y: float  # center
    w: float
    h: float
    conf: str

    @property
    def left(self) -> float: return self.x
    @property
    def right(self) -> float: return self.x + self.w
    @property
    def top(self) -> float: return self.y + self.h / 2
    @property
    def bottom(self) -> float: return self.y - self.h / 2


def _draw_logo(ax, x: float, y: float, rgba_arr: np.ndarray, zoom: float = 0.30) -> None:
    oi = OffsetImage(rgba_arr, zoom=zoom)
    ab = AnnotationBbox(oi, (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def draw_team_box(
    ax,
    x_left: float,
    y_center: float,
    team: Team,
    box_w: float,
    box_h: float,
    bar_w: float,
    logo_cache: Optional[LogoCache] = None,
    *,
    is_winner: bool = False,
    show_w_marker: bool = True,
) -> Box:
    y_bottom = y_center - box_h / 2

    if is_winner:
        edge = "#222222"
        lw = 1.8
        face = "#f4f6fb"
    else:
        edge = "#666666"
        lw = 1.0
        face = "white"

    rect = patches.FancyBboxPatch(
        (x_left, y_bottom),
        box_w,
        box_h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(rect)

    # accent bar
    if team.conf == "AFC":
        bar_x = x_left
        bar_color = "#d61f2c"
    else:
        bar_x = x_left + box_w - bar_w
        bar_color = "#1f4fd6"
    ax.add_patch(patches.Rectangle((bar_x, y_bottom), bar_w, box_h, linewidth=0, facecolor=bar_color, zorder=3))

    # logo
    logo = logo_cache.get(team.name) if logo_cache else None
    if logo is not None:
        if team.conf == "AFC":
            lx = x_left + bar_w + 0.018
        else:
            lx = x_left + box_w - bar_w - 0.018
        _draw_logo(ax, lx, y_center, logo, zoom=0.30)

    # label
    label = f"{team.name.upper()} ({team.seed})"
    if team.conf == "AFC":
        tx = x_left + bar_w + 0.040
        ha = "left"
    else:
        tx = x_left + box_w - bar_w - 0.040
        ha = "right"

    ax.text(tx, y_center, label, ha=ha, va="center",
            fontsize=8.3, fontweight="bold", color="#111111", zorder=4)

    # winner marker (kept away from arrow endpoints)
    if is_winner and show_w_marker:
        if team.conf == "AFC":
            wx = x_left + box_w - 0.020
            ha2 = "right"
        else:
            wx = x_left + 0.020
            ha2 = "left"
        ax.text(wx, y_center, "W", ha=ha2, va="center",
                fontsize=8.5, fontweight="bold", color="#111111", zorder=5)

    return Box(x=x_left, y=y_center, w=box_w, h=box_h, conf=team.conf)


def draw_game_pair(
    ax,
    x_left: float,
    y_center: float,
    matchup: Matchup,
    box_w: float,
    box_h: float,
    bar_w: float,
    logo_cache: Optional[LogoCache],
    gap_scale: float = 0.62,
) -> Tuple[Box, Box]:
    dy = box_h * gap_scale
    a_win = (matchup.a.name == matchup.winner.name)
    b_win = (matchup.b.name == matchup.winner.name)
    top = draw_team_box(ax, x_left, y_center + dy, matchup.a, box_w, box_h, bar_w, logo_cache, is_winner=a_win)
    bot = draw_team_box(ax, x_left, y_center - dy, matchup.b, box_w, box_h, bar_w, logo_cache, is_winner=b_win)
    return top, bot


# --------------------------
# Connector helpers (no lines inside boxes)
# --------------------------

def _elbow_path(points: List[Tuple[float, float]]) -> MplPath:
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    return MplPath(points, codes)

def draw_elbow_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    primary: str,
    elbow: float = 0.030,
    head: bool = True,
    shrink: float = 0.010,
    z: int = 1,
) -> None:
    """
    Structured 90° connector with arrowhead, and *shrink* so it stops
    before the box edges (avoids drawing into boxes).
    """
    line_color = "#666666"
    lw = 1.15
    x0, y0 = start
    x3, y3 = end

    if primary == "h":
        # For horizontal-first: go horizontally, then vertically, then horizontally
        # Calculate midpoint x: move from x0 toward x3 by elbow amount
        if x3 >= x0:
            # Going right
            mx = x0 + elbow
        else:
            # Going left
            mx = x0 - elbow
        pts = [(x0, y0), (mx, y0), (mx, y3), (x3, y3)]
    else:
        # For vertical-first: go vertically, then horizontally, then vertically
        # Calculate midpoint y: move from y0 toward y3 by elbow amount
        if y3 >= y0:
            # Going down
            my = y0 + elbow
        else:
            # Going up
            my = y0 - elbow
        pts = [(x0, y0), (x0, my), (x3, my), (x3, y3)]

    path = _elbow_path(pts)
    arrowstyle = "-|>" if head else "-"

    patch = patches.FancyArrowPatch(
        path=path,
        arrowstyle=arrowstyle,
        mutation_scale=9,
        linewidth=lw,
        color=line_color,
        shrinkA=shrink,
        shrinkB=shrink,
        zorder=z,
    )
    ax.add_patch(patch)

def _h_port(box: Box, side: str, pad: float = 0.010) -> Tuple[float, float]:
    """
    Horizontal port just outside left/right edge (so arrows don't enter boxes).
    """
    if side == "right":
        return (box.right + pad, box.y)
    return (box.left - pad, box.y)

def _v_port(box: Box, side: str, pad: float = 0.010) -> Tuple[float, float]:
    """
    Vertical port just outside top/bottom edge.
    """
    if side == "bottom":
        return (box.x + box.w / 2, box.bottom - pad)
    return (box.x + box.w / 2, box.top + pad)

def connect_winner_to_participant(ax, winner_box: Box, participant_box: Box, *, flow: str) -> None:
    # start just outside winner edge, end just outside participant edge
    if flow == "right":
        start = _h_port(winner_box, "right")
        end = _h_port(participant_box, "left")
    else:
        start = _h_port(winner_box, "left")
        end = _h_port(participant_box, "right")
    draw_elbow_arrow(ax, start, end, primary="h", elbow=0.030, head=True, shrink=0.005, z=1)

def connect_down_to_sb(ax, src_box: Box, dst_box: Box) -> None:
    start = _v_port(src_box, "bottom")
    end = _v_port(dst_box, "top")
    draw_elbow_arrow(ax, start, end, primary="v", elbow=0.030, head=True, shrink=0.005, z=1)

def _match_participant_slot(m: Matchup, team: Team) -> str:
    if team.name == m.a.name:
        return "a"
    if team.name == m.b.name:
        return "b"
    if team.seed == m.a.seed:
        return "a"
    if team.seed == m.b.seed:
        return "b"
    return "a"


def render_bracket(
    afc: FrozenBracket,
    nfc: FrozenBracket,
    season: int,
    out_path: Path,
    logo_dir: Optional[Path] = None,
    sb_winner: Optional[str] = None,
) -> None:
    BOX_W = 0.195
    BOX_H = 0.043
    BAR_W = 0.010

    # Upper columns
    A_WC_X, A_DIV_X = 0.06, 0.28
    N_WC_X, N_DIV_X = 0.74, 0.52

    # Upper rows
    WC_Y = [0.80, 0.60, 0.40]
    DIV_Y = [0.70, 0.50]

    # Bottom finals
    CONF_GAME_Y = 0.22
    CONF_GAME_A_X = 0.22
    CONF_GAME_N_X = 0.58

    SB_TITLE_Y = 0.12
    SB_Y = 0.065
    SB_X_L = 0.29
    SB_X_R = 0.51

    fig = plt.figure(figsize=(14, 9), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Side labels
    ax.text(0.02, 0.50, "AFC", rotation=90, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#d61f2c")
    ax.text(0.98, 0.50, "NFC", rotation=270, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1f4fd6")

    logo_cache = LogoCache(logo_dir=logo_dir, max_px=64) if logo_dir else None

    # ========================================================
    # AFC WC games
    # ========================================================
    afc_wc_rows: List[Tuple[Box, Box, Matchup]] = []
    for i, m in enumerate(afc.wc):
        a_box, b_box = draw_game_pair(ax, A_WC_X, WC_Y[i], m, BOX_W, BOX_H, BAR_W, logo_cache)
        afc_wc_rows.append((a_box, b_box, m))

    # AFC DIV games
    afc_div_rows: List[Tuple[Box, Box, Matchup]] = []
    for i, m in enumerate(afc.div):
        a_box, b_box = draw_game_pair(ax, A_DIV_X, DIV_Y[i], m, BOX_W, BOX_H, BAR_W, logo_cache)
        afc_div_rows.append((a_box, b_box, m))

    # Connect AFC WC winners -> AFC DIV participant
    # Arrows removed
    # for a_box, b_box, m in afc_wc_rows:
    #     w = m.winner
    #     winner_box = a_box if w.name == m.a.name else b_box
    #
    #     target = None
    #     for da, db, dm in afc_div_rows:
    #         if w.name in (dm.a.name, dm.b.name) or w.seed in (dm.a.seed, dm.b.seed):
    #             slot = _match_participant_slot(dm, w)
    #             target_box = da if slot == "a" else db
    #             target = target_box
    #             break
    #     if target:
    #         connect_winner_to_participant(ax, winner_box, target, flow="right")

    # ========================================================
    # NFC WC games
    # ========================================================
    nfc_wc_rows: List[Tuple[Box, Box, Matchup]] = []
    for i, m in enumerate(nfc.wc):
        a_box, b_box = draw_game_pair(ax, N_WC_X, WC_Y[i], m, BOX_W, BOX_H, BAR_W, logo_cache)
        nfc_wc_rows.append((a_box, b_box, m))

    # NFC DIV games
    nfc_div_rows: List[Tuple[Box, Box, Matchup]] = []
    for i, m in enumerate(nfc.div):
        a_box, b_box = draw_game_pair(ax, N_DIV_X, DIV_Y[i], m, BOX_W, BOX_H, BAR_W, logo_cache)
        nfc_div_rows.append((a_box, b_box, m))

    # Connect NFC WC winners -> NFC DIV participant
    # Arrows removed
    # for a_box, b_box, m in nfc_wc_rows:
    #     w = m.winner
    #     winner_box = a_box if w.name == m.a.name else b_box
    #
    #     target = None
    #     for da, db, dm in nfc_div_rows:
    #         if w.name in (dm.a.name, dm.b.name) or w.seed in (dm.a.seed, dm.b.seed):
    #             slot = _match_participant_slot(dm, w)
    #             target_box = da if slot == "a" else db
    #             target = target_box
    #             break
    #     if target:
    #         connect_winner_to_participant(ax, winner_box, target, flow="left")

    # ========================================================
    # Conference Championships (bottom) — two participants each
    # ========================================================
    afc_conf_top = draw_team_box(
        ax, CONF_GAME_A_X, CONF_GAME_Y + BOX_H * 0.62, afc.conf.a,
        BOX_W, BOX_H, BAR_W, logo_cache,
        is_winner=(afc.conf.a.name == afc.conf.winner.name),
    )
    afc_conf_bot = draw_team_box(
        ax, CONF_GAME_A_X, CONF_GAME_Y - BOX_H * 0.62, afc.conf.b,
        BOX_W, BOX_H, BAR_W, logo_cache,
        is_winner=(afc.conf.b.name == afc.conf.winner.name),
    )

    nfc_conf_top = draw_team_box(
        ax, CONF_GAME_N_X, CONF_GAME_Y + BOX_H * 0.62, nfc.conf.a,
        BOX_W, BOX_H, BAR_W, logo_cache,
        is_winner=(nfc.conf.a.name == nfc.conf.winner.name),
    )
    nfc_conf_bot = draw_team_box(
        ax, CONF_GAME_N_X, CONF_GAME_Y - BOX_H * 0.62, nfc.conf.b,
        BOX_W, BOX_H, BAR_W, logo_cache,
        is_winner=(nfc.conf.b.name == nfc.conf.winner.name),
    )

    # Connect DIV winners -> conf participant slots
    # Arrows removed
    # for da, db, dm in afc_div_rows:
    #     div_winner_box = da if dm.winner.name == dm.a.name else db
    #     slot = _match_participant_slot(afc.conf, dm.winner)
    #     target = afc_conf_top if slot == "a" else afc_conf_bot
    #     connect_winner_to_participant(ax, div_winner_box, target, flow="right")
    #
    # for da, db, dm in nfc_div_rows:
    #     div_winner_box = da if dm.winner.name == dm.a.name else db
    #     slot = _match_participant_slot(nfc.conf, dm.winner)
    #     target = nfc_conf_top if slot == "a" else nfc_conf_bot
    #     connect_winner_to_participant(ax, div_winner_box, target, flow="left")

    # ========================================================
    # Super Bowl (bottom)
    # ========================================================
    ax.text(0.5, SB_TITLE_Y, "SUPER BOWL", ha="center", va="center", fontsize=16, fontweight="bold")

    # Highlight Super Bowl winner if provided
    sb_afc_is_winner = (sb_winner is not None and sb_winner == afc.champ.name)
    sb_nfc_is_winner = (sb_winner is not None and sb_winner == nfc.champ.name)
    sb_afc_box = draw_team_box(ax, SB_X_L, SB_Y, afc.champ, BOX_W, BOX_H, BAR_W, logo_cache, is_winner=sb_afc_is_winner)
    sb_nfc_box = draw_team_box(ax, SB_X_R, SB_Y, nfc.champ, BOX_W, BOX_H, BAR_W, logo_cache, is_winner=sb_nfc_is_winner)

    # Arrows removed
    # afc_winner_box = afc_conf_top if afc.conf.winner.name == afc.conf.a.name else afc_conf_bot
    # nfc_winner_box = nfc_conf_top if nfc.conf.winner.name == nfc.conf.a.name else nfc_conf_bot
    # connect_down_to_sb(ax, afc_winner_box, sb_afc_box)
    # connect_down_to_sb(ax, nfc_winner_box, sb_nfc_box)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


# ============================================================
# Adapter: render from dict
# ============================================================

def render_from_dict(
    bracket_dict: Dict[str, Any],
    season: int,
    out_path: Path,
    logo_dir: Optional[Path] = None,
) -> None:
    """
    Expected format:
    {
      "AFC": {"WC": [..3..], "DIV": [..2..], "CONF": {...}},
      "NFC": {"WC": [..3..], "DIV": [..2..], "CONF": {...}},

      # OPTIONAL (only if you want ONE SB winner highlighted):
      # "SUPERBOWL": {
      #   "a_name": "...", "a_seed": 1, "b_name": "...", "b_seed": 2,
      #   "winner_name": "...", "winner_seed": 2
      # }
    }

    If SUPERBOWL is provided, the winner will be highlighted.
    
    Note: This function uses the new validation module for early validation.
    For new code, prefer using src.facade.bracket_api.render_bracket().
    """
    # Validate bracket dict using new validation module
    try:
        validate_bracket_dict(bracket_dict)
    except ValueError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Bracket validation warning: {e}. Continuing with render...")
    
    afc = freeze_conference("AFC", bracket_dict["AFC"]["WC"], bracket_dict["AFC"]["DIV"], bracket_dict["AFC"]["CONF"])
    nfc = freeze_conference("NFC", bracket_dict["NFC"]["WC"], bracket_dict["NFC"]["DIV"], bracket_dict["NFC"]["CONF"])
    
    # Extract Super Bowl winner if provided
    sb_winner = None
    if "SUPERBOWL" in bracket_dict:
        sb_winner = bracket_dict["SUPERBOWL"].get("winner_name")
    
    render_bracket(afc, nfc, season=season, out_path=out_path, logo_dir=logo_dir, sb_winner=sb_winner)
