"""
Compute bracket layout from frozen bracket.

Deterministic layout computation - no drawing, just geometry.
"""

from typing import Dict, List
from src.models.bracket import FrozenBracket, Matchup, Team
from src.layout.layout_model import LayoutModel, Box, Connector, Label, BoxId
from src.layout.grid_calculator import GridCalculator


def compute_bracket_layout(
    frozen: FrozenBracket,
    conference: str,
    grid_calculator: GridCalculator = None
) -> LayoutModel:
    """
    Compute layout model from frozen bracket.
    
    Args:
        frozen: FrozenBracket for a conference
        conference: 'AFC' or 'NFC'
        grid_calculator: Optional GridCalculator (uses default if None)
    
    Returns:
        LayoutModel with all boxes, connectors, and annotations
    """
    if grid_calculator is None:
        grid_calculator = GridCalculator()
    
    boxes: Dict[BoxId, Box] = {}
    connectors: List[Connector] = []
    annotations: List[Label] = []
    
    # Generate box IDs deterministically
    box_id_counter = 0
    
    def make_box_id(prefix: str) -> BoxId:
        nonlocal box_id_counter
        box_id = BoxId(f"{conference}_{prefix}_{box_id_counter}")
        box_id_counter += 1
        return box_id
    
    # Track box IDs for connectors
    wc_box_ids: List[Dict[str, BoxId]] = []  # List of {team_name: box_id} for each matchup
    div_box_ids: List[Dict[str, BoxId]] = []
    conf_box_ids: Dict[str, BoxId] = {}
    
    # Wild Card round
    for slot_idx, matchup in enumerate(frozen.wc):
        # Top team
        x, y = grid_calculator.get_wc_position(conference, slot_idx, 'top')
        top_box_id = make_box_id(f"WC_{slot_idx}_top")
        top_box = Box(
            id=top_box_id,
            x_left=x,
            y_center=y,
            width=grid_calculator.BOX_WIDTH,
            height=grid_calculator.BOX_HEIGHT,
            team_name=matchup.a.name,
            seed=matchup.a.seed,
            conference=conference,
            is_winner=(matchup.winner.name == matchup.a.name)
        )
        boxes[top_box_id] = top_box
        
        # Bottom team
        x, y = grid_calculator.get_wc_position(conference, slot_idx, 'bottom')
        bottom_box_id = make_box_id(f"WC_{slot_idx}_bottom")
        bottom_box = Box(
            id=bottom_box_id,
            x_left=x,
            y_center=y,
            width=grid_calculator.BOX_WIDTH,
            height=grid_calculator.BOX_HEIGHT,
            team_name=matchup.b.name,
            seed=matchup.b.seed,
            conference=conference,
            is_winner=(matchup.winner.name == matchup.b.name)
        )
        boxes[bottom_box_id] = bottom_box
        
        wc_box_ids.append({
            matchup.a.name: top_box_id,
            matchup.b.name: bottom_box_id,
            'winner': top_box_id if matchup.winner.name == matchup.a.name else bottom_box_id
        })
    
    # Divisional round
    for slot_idx, matchup in enumerate(frozen.div):
        # Top team
        x, y = grid_calculator.get_div_position(conference, slot_idx, 'top')
        top_box_id = make_box_id(f"DIV_{slot_idx}_top")
        top_box = Box(
            id=top_box_id,
            x_left=x,
            y_center=y,
            width=grid_calculator.BOX_WIDTH,
            height=grid_calculator.BOX_HEIGHT,
            team_name=matchup.a.name,
            seed=matchup.a.seed,
            conference=conference,
            is_winner=(matchup.winner.name == matchup.a.name)
        )
        boxes[top_box_id] = top_box
        
        # Bottom team
        x, y = grid_calculator.get_div_position(conference, slot_idx, 'bottom')
        bottom_box_id = make_box_id(f"DIV_{slot_idx}_bottom")
        bottom_box = Box(
            id=bottom_box_id,
            x_left=x,
            y_center=y,
            width=grid_calculator.BOX_WIDTH,
            height=grid_calculator.BOX_HEIGHT,
            team_name=matchup.b.name,
            seed=matchup.b.seed,
            conference=conference,
            is_winner=(matchup.winner.name == matchup.b.name)
        )
        boxes[bottom_box_id] = bottom_box
        
        div_box_ids.append({
            matchup.a.name: top_box_id,
            matchup.b.name: bottom_box_id,
            'winner': top_box_id if matchup.winner.name == matchup.a.name else bottom_box_id
        })
        
        # Connect WC winners to DIV
        # Determine which WC winner goes to this DIV slot
        # This is based on bracket structure: seed 1 plays highest WC seed, others play remaining
        if slot_idx == 0:
            # First DIV slot: seed 1 vs highest WC seed
            # Find WC winner with highest seed
            highest_wc_winner = None
            highest_wc_seed = 0
            for wc_matchup in frozen.wc:
                if wc_matchup.winner.seed > highest_wc_seed:
                    highest_wc_seed = wc_matchup.winner.seed
                    highest_wc_winner = wc_matchup.winner.name
            
            if highest_wc_winner and highest_wc_winner in wc_box_ids[0]:
                # Connect from WC winner to DIV bottom (seed 1 is top)
                wc_winner_box_id = wc_box_ids[0][highest_wc_winner]
                connectors.append(Connector(
                    src_id=wc_winner_box_id,
                    dst_id=bottom_box_id,
                    style='elbow_h',
                    color='#666666'
                ))
        else:
            # Second DIV slot: remaining WC winners
            # Connect from appropriate WC winners
            # This is simplified - actual logic depends on which WC winners advance
            pass
    
    # Conference Championship
    x, y = grid_calculator.get_conf_position(conference, 'top')
    conf_top_box_id = make_box_id("CONF_top")
    conf_top_box = Box(
        id=conf_top_box_id,
        x_left=x,
        y_center=y,
        width=grid_calculator.BOX_WIDTH,
        height=grid_calculator.BOX_HEIGHT,
        team_name=frozen.conf.a.name,
        seed=frozen.conf.a.seed,
        conference=conference,
        is_winner=(frozen.conf.winner.name == frozen.conf.a.name)
    )
    boxes[conf_top_box_id] = conf_top_box
    
    x, y = grid_calculator.get_conf_position(conference, 'bottom')
    conf_bottom_box_id = make_box_id("CONF_bottom")
    conf_bottom_box = Box(
        id=conf_bottom_box_id,
        x_left=x,
        y_center=y,
        width=grid_calculator.BOX_WIDTH,
        height=grid_calculator.BOX_HEIGHT,
        team_name=frozen.conf.b.name,
        seed=frozen.conf.b.seed,
        conference=conference,
        is_winner=(frozen.conf.winner.name == frozen.conf.b.name)
    )
    boxes[conf_bottom_box_id] = conf_bottom_box
    
    conf_box_ids = {
        frozen.conf.a.name: conf_top_box_id,
        frozen.conf.b.name: conf_bottom_box_id,
        'winner': conf_top_box_id if frozen.conf.winner.name == frozen.conf.a.name else conf_bottom_box_id
    }
    
    # Connect DIV winners to CONF
    for div_idx, div_matchup in enumerate(frozen.div):
        div_winner_name = div_matchup.winner.name
        div_winner_box_id = div_box_ids[div_idx]['winner']
        
        # Connect to appropriate CONF position
        if div_winner_name == frozen.conf.a.name:
            connectors.append(Connector(
                src_id=div_winner_box_id,
                dst_id=conf_top_box_id,
                style='elbow_h',
                color='#666666'
            ))
        elif div_winner_name == frozen.conf.b.name:
            connectors.append(Connector(
                src_id=div_winner_box_id,
                dst_id=conf_bottom_box_id,
                style='elbow_h',
                color='#666666'
            ))
    
    return LayoutModel(
        boxes=boxes,
        connectors=connectors,
        annotations=annotations
    )

