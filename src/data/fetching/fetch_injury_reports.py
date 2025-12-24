"""
Fetch NFL injury reports using nflverse/nflfastR data.

Uses nflverse injury report data (2009-present, strongest post-2015).
Provides weekly injury status: Out, Doubtful, Questionable
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_nflverse_available() -> bool:
    """Check if nflverse packages are available."""
    # Try nfl-data-py first (already installed and has import_injuries)
    try:
        import nfl_data_py as nfl
        if hasattr(nfl, 'import_injuries') or hasattr(nfl, 'import_injury_data'):
            return True
    except ImportError:
        pass
    
    # Try nflreadpy as alternative
    try:
        import nflreadpy
        return True
    except ImportError:
        logger.warning("nflverse packages not available. Options:")
        logger.warning("  1. nfl-data-py (already installed) - has import_injuries()")
        logger.warning("  2. pip install nflreadpy - alternative nflverse package")
        return False


def fetch_injury_reports(season: int, week: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Fetch injury reports for a season/week.
    
    Parameters:
    -----------
    season : int
        Season year
    week : int, optional
        Specific week (if None, fetches all weeks)
        
    Returns:
    --------
    pd.DataFrame : Injury report data with columns:
        - player_name, team, position
        - injury_status (Out, Doubtful, Questionable, Probable)
        - practice_status
        - game_available (boolean)
    """
    if not _check_nflverse_available():
        return None
    
    # Try nfl-data-py first (already installed)
    try:
        import nfl_data_py as nfl
        
        logger.info(f"Fetching injury reports for season {season}" + (f", week {week}" if week else ""))
        
        # nfl-data-py has import_injuries method
        if hasattr(nfl, 'import_injuries'):
            injuries = nfl.import_injuries([season])
            if injuries is not None and not injuries.empty:
                # Filter by week if specified
                if week and 'week' in injuries.columns:
                    injuries = injuries[injuries['week'] == week]
                logger.info(f"Successfully fetched {len(injuries)} injury records using nfl-data-py")
                return injuries
        elif hasattr(nfl, 'import_injury_data'):
            injuries = nfl.import_injury_data([season])
            if injuries is not None and not injuries.empty:
                if week and 'week' in injuries.columns:
                    injuries = injuries[injuries['week'] == week]
                logger.info(f"Successfully fetched {len(injuries)} injury records using nfl-data-py")
                return injuries
        
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error using nfl-data-py: {e}")
    
    # Try nflreadpy as alternative
    try:
        import nflreadpy
        
        logger.info(f"Fetching injury reports for season {season}" + (f", week {week}" if week else ""))
        
        # nflreadpy has load_injuries() function
        try:
            if week:
                # Try to fetch specific week (may need to filter)
                injuries = nflreadpy.load_injuries([season])
                if not injuries.empty and 'week' in injuries.columns:
                    injuries = injuries[injuries['week'] == week]
            else:
                injuries = nflreadpy.load_injuries([season])
            
            if injuries is not None and not injuries.empty:
                logger.info(f"Successfully fetched {len(injuries)} injury records using nflreadpy")
                return injuries
            else:
                logger.warning("No injury data returned")
                return None
                
        except AttributeError:
            # Try alternative function names
            try:
                injuries = nflreadpy.load_injury_data([season])
                if injuries is not None and not injuries.empty:
                    logger.info(f"Successfully fetched {len(injuries)} injury records using nflreadpy")
                    return injuries
            except:
                pass
            
            logger.warning("nflreadpy.load_injuries() not available")
            return None
            
    except ImportError:
        logger.error("Neither nfl-data-py nor nflreadpy available for injury data")
        logger.error("nfl-data-py should already be installed. Check: pip list | findstr nfl")
        return None
    except Exception as e:
        logger.error(f"Error fetching injury reports: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def calculate_team_injury_impact(injury_df: pd.DataFrame, team: str, 
                                 week: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate injury impact for a team.
    
    Parameters:
    -----------
    injury_df : pd.DataFrame
        Injury report data
    team : str
        Team abbreviation
    week : int, optional
        Specific week to analyze
        
    Returns:
    --------
    dict : Injury impact metrics:
        - key_players_out: Number of key players out
        - total_players_out: Total players out
        - injury_score: Weighted injury impact score
    """
    if injury_df is None or injury_df.empty:
        return {
            'key_players_out': 0,
            'total_players_out': 0,
            'injury_score': 0.0
        }
    
    # Filter by team
    team_injuries = injury_df[injury_df['team'] == team].copy()
    
    if week:
        if 'week' in team_injuries.columns:
            team_injuries = team_injuries[team_injuries['week'] == week]
    
    if team_injuries.empty:
        return {
            'key_players_out': 0,
            'total_players_out': 0,
            'injury_score': 0.0
        }
    
    # Player importance weights (based on position)
    position_weights = {
        'QB': 1.0,      # Most important
        'RB': 0.4,
        'WR': 0.35,
        'TE': 0.25,
        'OL': 0.3,     # Offensive line (aggregate)
        'DE': 0.3,
        'DT': 0.25,
        'LB': 0.3,
        'CB': 0.3,
        'S': 0.25,
        'K': 0.1,
        'P': 0.05
    }
    
    # Injury status weights (nfl_data_py uses 'report_status' column)
    status_weights = {
        'Out': 1.0,
        'Doubtful': 0.8,
        'Questionable': 0.5,
        'Probable': 0.2,
        'IR': 1.0,  # Injured Reserve
        'PUP': 0.9,  # Physically Unable to Perform
    }
    
    # Calculate injury score
    total_score = 0.0
    key_players_out = 0
    total_out = 0
    
    for _, row in team_injuries.iterrows():
        position = str(row.get('position', '')).upper()
        # nfl_data_py uses 'report_status' column
        status = str(row.get('report_status', '')).title()
        
        # Check if player is out or doubtful
        if status in ['Out', 'Doubtful', 'IR', 'PUP']:
            total_out += 1
            
            # Check if key player (QB or high-impact position)
            if position == 'QB' or position in ['RB', 'WR', 'DE', 'CB']:
                key_players_out += 1
        
        # Calculate weighted score
        pos_weight = position_weights.get(position, 0.2)
        status_weight = status_weights.get(status, 0.5)
        total_score += pos_weight * status_weight
    
    return {
        'key_players_out': key_players_out,
        'total_players_out': total_out,
        'injury_score': total_score
    }


def get_team_injury_adjustment(team: str, season: int, week: Optional[int] = None) -> float:
    """
    Get injury adjustment factor for a team (to apply to win probability).
    
    Parameters:
    -----------
    team : str
        Team abbreviation
    season : int
        Season year
    week : int, optional
        Specific week
        
    Returns:
    --------
    float : Adjustment factor (-0.20 to 0.0)
        Negative value reduces win probability
    """
    injury_df = fetch_injury_reports(season, week)
    
    if injury_df is None or injury_df.empty:
        return 0.0
    
    impact = calculate_team_injury_impact(injury_df, team, week)
    
    # Convert injury score to adjustment factor
    # Max adjustment: -20% for severe injuries
    # Scale: injury_score of 5.0+ = -20%, 2.5 = -10%, etc.
    max_score = 5.0  # Threshold for maximum impact
    adjustment = -0.20 * min(impact['injury_score'] / max_score, 1.0)
    
    # QB injury gets extra penalty
    if impact['key_players_out'] > 0:
        # Check if QB is out
        team_injuries = injury_df[injury_df['team'] == team]
        if week and 'week' in team_injuries.columns:
            team_injuries = team_injuries[team_injuries['week'] == week]
        
        # nfl_data_py uses 'report_status' column
        qb_out = team_injuries[
            (team_injuries['position'].str.upper() == 'QB') &
            (team_injuries['report_status'].isin(['Out', 'Doubtful', 'IR']))
        ]
        
        if not qb_out.empty:
            adjustment -= 0.12  # Additional -12% for QB injury
    
    return np.clip(adjustment, -0.20, 0.0)


def save_injury_reports(injury_df: pd.DataFrame, season: int, 
                       output_dir: Optional[Path] = None) -> Path:
    """
    Save injury reports to CSV file.
    
    Parameters:
    -----------
    injury_df : pd.DataFrame
        Injury report data
    season : int
        Season year
    output_dir : Path, optional
        Output directory (default: data/historical/nflfastr/injuries)
        
    Returns:
    --------
    Path : Path to saved file
    """
    if output_dir is None:
        output_dir = data_config.historical_data_dir / "nflfastr" / "injuries"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"injuries_{season}.csv"
    
    injury_df.to_csv(output_path, index=False)
    logger.info(f"Saved injury reports to {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL injury reports')
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year (default: 2024)')
    parser.add_argument('--week', type=int, default=None,
                       help='Specific week (optional)')
    parser.add_argument('--team', type=str, default=None,
                       help='Filter by team (optional)')
    parser.add_argument('--save', action='store_true',
                       help='Save to CSV file')
    
    args = parser.parse_args()
    
    injuries = fetch_injury_reports(args.season, args.week)
    
    if injuries is not None and not injuries.empty:
        print(f"\nFetched {len(injuries)} injury records")
        print(f"\nColumns: {list(injuries.columns)}")
        
        if args.team:
            team_injuries = injuries[injuries['team'] == args.team.upper()]
            print(f"\nInjuries for {args.team}:")
            print(team_injuries.to_string())
            
            impact = calculate_team_injury_impact(injuries, args.team.upper(), args.week)
            print(f"\nInjury Impact:")
            print(f"  Key players out: {impact['key_players_out']}")
            print(f"  Total players out: {impact['total_players_out']}")
            print(f"  Injury score: {impact['injury_score']:.2f}")
            print(f"  Win probability adjustment: {get_team_injury_adjustment(args.team.upper(), args.season, args.week):.1%}")
        else:
            print(f"\nSample data:")
            print(injuries.head(10).to_string())
        
        if args.save:
            save_injury_reports(injuries, args.season)
    else:
        print("No injury data available")

