"""
Fetch actual NFL team win/loss records using nfl-data-py
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_team_records(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch actual team win/loss records for a season.
    
    Returns DataFrame with columns: team, win, loss, tie, win_pct
    """
    try:
        import nfl_data_py as nfl
        
        # Fetch schedule data
        schedule = nfl.import_schedules([season])
        
        if schedule.empty:
            logger.warning(f"No schedule data for season {season}")
            return None
        
        # Calculate team records
        team_records = []
        
        for team in schedule['home_team'].unique():
            if pd.isna(team):
                continue
            
            # Home games
            home_games = schedule[schedule['home_team'] == team]
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            home_losses = (home_games['home_score'] < home_games['away_score']).sum()
            home_ties = (home_games['home_score'] == home_games['away_score']).sum()
            
            # Away games
            away_games = schedule[schedule['away_team'] == team]
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            away_losses = (away_games['away_score'] < away_games['home_score']).sum()
            away_ties = (away_games['away_score'] == away_games['home_score']).sum()
            
            # Total
            wins = home_wins + away_wins
            losses = home_losses + away_losses
            ties = home_ties + away_ties
            total_games = wins + losses + ties
            
            if total_games > 0:
                win_pct = (wins + 0.5 * ties) / total_games
            else:
                win_pct = 0.0
            
            team_records.append({
                'team': team,
                'win': wins,
                'loss': losses,
                'tie': ties,
                'win_pct': win_pct,
                'season': season
            })
        
        df = pd.DataFrame(team_records)
        
        if df.empty:
            logger.warning(f"No team records calculated for season {season}")
            return None
        
        logger.info(f"Fetched records for {len(df)} teams in season {season}")
        return df
        
    except ImportError:
        logger.warning("nfl-data-py not available, cannot fetch team records")
        return None
    except Exception as e:
        logger.warning(f"Error fetching team records: {e}")
        return None


def main():
    """Test fetching team records"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL team records')
    parser.add_argument('--season', type=int, default=2023)
    parser.add_argument('--output', type=str, default='data/processed/team_records.csv')
    
    args = parser.parse_args()
    
    records = fetch_team_records(args.season)
    
    if records is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(output_path, index=False)
        print(f"\nTeam records saved to {output_path}")
        print(f"\nSample records:")
        print(records.head(10))
    else:
        print("Failed to fetch team records")


if __name__ == '__main__':
    main()

