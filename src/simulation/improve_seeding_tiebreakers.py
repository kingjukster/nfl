"""
Enhanced seeding with full NFL tiebreaker rules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.simulation.predictor import PlayoffPredictor
from src.constants import NFL_CONFERENCES
import logging

logger = logging.getLogger(__name__)


class EnhancedSeedingPredictor(PlayoffPredictor):
    """
    Enhanced seeding predictor with full NFL tiebreaker rules.
    """
    
    def __init__(self, team_stats_path: str, game_results_path: str = None):
        super().__init__(team_stats_path)
        self.game_results_path = game_results_path
        self.game_results = None
    
    def load_game_results(self, season: int) -> pd.DataFrame:
        """Load game-by-game results for tiebreaker calculations"""
        if self.game_results_path:
            game_path = Path(self.game_results_path)
            if game_path.exists():
                games = pd.read_csv(game_path)
                if 'season' in games.columns:
                    games = games[games['season'] == season]
                return games
        
        # Try season-specific file
        season_path = Path(f"data/processed/game_results_{season}.csv")
        if season_path.exists():
            return pd.read_csv(season_path)
        
        # Try generic file
        generic_path = Path("data/processed/game_results.csv")
        if generic_path.exists():
            games = pd.read_csv(generic_path)
            if 'season' in games.columns:
                games = games[games['season'] == season]
            return games
        
        logger.warning(f"No game results found for season {season}")
        return pd.DataFrame()
    
    def get_team_division(self, team: str) -> str:
        """Get division for a team"""
        for conf in ['AFC', 'NFC']:
            for division, teams in NFL_CONFERENCES[conf]['teams'].items():
                if team in teams:
                    return division
        return None
    
    def get_team_conference(self, team: str) -> str:
        """Get conference for a team"""
        for conf in ['AFC', 'NFC']:
            for division, teams in NFL_CONFERENCES[conf]['teams'].items():
                if team in teams:
                    return conf
        return None
    
    def calculate_head_to_head(self, teams: List[str], game_data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate head-to-head records between teams"""
        h2h_records = {}
        
        for team in teams:
            h2h_records[team] = {'wins': 0, 'losses': 0, 'ties': 0}
            
            # Games vs other tied teams
            vs_tied = game_data[
                (game_data['team'] == team) & 
                (game_data['opponent'].isin(teams))
            ]
            
            for _, game in vs_tied.iterrows():
                if game['result'] == 'W':
                    h2h_records[team]['wins'] += 1
                elif game['result'] == 'L':
                    h2h_records[team]['losses'] += 1
                else:
                    h2h_records[team]['ties'] += 1
        
        return h2h_records
    
    def calculate_division_record(self, team: str, game_data: pd.DataFrame) -> Tuple[int, int, int]:
        """Calculate team's record within division"""
        division = self.get_team_division(team)
        if not division:
            return 0, 0, 0
        
        conf = self.get_team_conference(team)
        division_teams = NFL_CONFERENCES[conf]['teams'][division]
        
        # Games vs division opponents
        div_games = game_data[
            (game_data['team'] == team) &
            (game_data['opponent'].isin(division_teams))
        ]
        
        wins = len(div_games[div_games['result'] == 'W'])
        losses = len(div_games[div_games['result'] == 'L'])
        ties = len(div_games[div_games['result'] == 'T'])
        
        return wins, losses, ties
    
    def calculate_conference_record(self, team: str, game_data: pd.DataFrame) -> Tuple[int, int, int]:
        """Calculate team's record within conference"""
        conf = self.get_team_conference(team)
        if not conf:
            return 0, 0, 0
        
        conf_teams = []
        for division, teams in NFL_CONFERENCES[conf]['teams'].items():
            conf_teams.extend(teams)
        
        # Games vs conference opponents
        conf_games = game_data[
            (game_data['team'] == team) &
            (game_data['opponent'].isin(conf_teams))
        ]
        
        wins = len(conf_games[conf_games['result'] == 'W'])
        losses = len(conf_games[conf_games['result'] == 'L'])
        ties = len(conf_games[conf_games['result'] == 'T'])
        
        return wins, losses, ties
    
    def apply_tiebreakers(self, teams_df: pd.DataFrame, tied_teams: List[str], 
                         game_data: pd.DataFrame) -> List[str]:
        """
        Apply NFL tiebreaker rules to break ties.
        
        Returns sorted list of teams (best to worst).
        """
        if len(tied_teams) == 1:
            return tied_teams
        
        if game_data.empty:
            logger.warning("No game data available, using simple tiebreaker")
            # Fallback to points_for
            tied_df = teams_df[teams_df['team'].isin(tied_teams)]
            return tied_df.sort_values('points_for', ascending=False)['team'].tolist()
        
        # 1. Head-to-head record (if all tied teams played each other)
        h2h_records = self.calculate_head_to_head(tied_teams, game_data)
        
        # Check if all teams played each other
        all_played = True
        for team1 in tied_teams:
            for team2 in tied_teams:
                if team1 != team2:
                    games = game_data[
                        ((game_data['team'] == team1) & (game_data['opponent'] == team2)) |
                        ((game_data['team'] == team2) & (game_data['opponent'] == team1))
                    ]
                    if games.empty:
                        all_played = False
                        break
            if not all_played:
                break
        
        if all_played and len(tied_teams) <= 3:  # H2H only works for small groups
            # Sort by head-to-head win_pct
            h2h_win_pct = {}
            for team in tied_teams:
                rec = h2h_records[team]
                total = rec['wins'] + rec['losses'] + rec['ties']
                if total > 0:
                    h2h_win_pct[team] = (rec['wins'] + 0.5 * rec['ties']) / total
                else:
                    h2h_win_pct[team] = 0.0
            
            sorted_teams = sorted(tied_teams, key=lambda t: h2h_win_pct[t], reverse=True)
            if len(set(h2h_win_pct.values())) == len(h2h_win_pct):  # All different
                return sorted_teams
        
        # 2. Division record (if same division)
        same_division = True
        division = self.get_team_division(tied_teams[0])
        for team in tied_teams[1:]:
            if self.get_team_division(team) != division:
                same_division = False
                break
        
        if same_division and division:
            div_records = {}
            for team in tied_teams:
                wins, losses, ties = self.calculate_division_record(team, game_data)
                total = wins + losses + ties
                if total > 0:
                    div_records[team] = (wins + 0.5 * ties) / total
                else:
                    div_records[team] = 0.0
            
            sorted_teams = sorted(tied_teams, key=lambda t: div_records[t], reverse=True)
            if len(set(div_records.values())) == len(div_records):  # All different
                return sorted_teams
        
        # 3. Conference record
        conf_records = {}
        for team in tied_teams:
            wins, losses, ties = self.calculate_conference_record(team, game_data)
            total = wins + losses + ties
            if total > 0:
                conf_records[team] = (wins + 0.5 * ties) / total
            else:
                conf_records[team] = 0.0
        
        sorted_teams = sorted(tied_teams, key=lambda t: conf_records[t], reverse=True)
        if len(set(conf_records.values())) == len(conf_records):  # All different
            return sorted_teams
        
        # 4. Common games (would need opponent tracking)
        # Skip for now - complex to implement
        
        # 5. Strength of victory (would need opponent records)
        # Skip for now
        
        # 6. Points scored
        tied_df = teams_df[teams_df['team'].isin(tied_teams)]
        points_col = None
        for col in ['total_off_points', 'points_for', 'PF']:
            if col in tied_df.columns:
                points_col = col
                break
        
        if points_col:
            sorted_teams = tied_df.sort_values(points_col, ascending=False)['team'].tolist()
            return sorted_teams
        
        # 7. Points allowed (reverse - fewer is better)
        pa_col = None
        for col in ['total_def_points', 'points_against', 'PA']:
            if col in tied_df.columns:
                pa_col = col
                break
        
        if pa_col:
            sorted_teams = tied_df.sort_values(pa_col, ascending=True)['team'].tolist()
            return sorted_teams
        
        # Fallback: return as-is
        return tied_teams
    
    def predict_seeding(self, season: int, conference: str = None) -> Dict[str, List]:
        """
        Predict seeding with enhanced tiebreakers.
        """
        # Load game results
        self.game_results = self.load_game_results(season)
        
        # Get base seeding
        seeding = super().predict_seeding(season, conference)
        
        # Apply tiebreakers to improve accuracy
        df = self.load_team_stats(season)
        df = self.calculate_win_pct(df)
        
        for conf, seeds in seeding.items():
            # Group teams by win_pct to find ties
            win_pct_groups = {}
            for seed in seeds:
                wp = seed.win_pct
                if wp not in win_pct_groups:
                    win_pct_groups[wp] = []
                win_pct_groups[wp].append(seed.team)
            
            # Apply tiebreakers to tied groups
            for wp, tied_teams in win_pct_groups.items():
                if len(tied_teams) > 1:
                    # These teams are tied - apply tiebreakers
                    sorted_teams = self.apply_tiebreakers(df, tied_teams, self.game_results)
                    
                    # Update seeds based on tiebreaker results
                    # (This is simplified - full implementation would re-rank all seeds)
                    logger.info(f"Applied tiebreakers to {conf} teams with win_pct {wp:.3f}: {sorted_teams}")
        
        return seeding


def main():
    """Test enhanced seeding"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced seeding with tiebreakers')
    parser.add_argument('--season', type=int, default=2023)
    
    args = parser.parse_args()
    
    # Fetch game results first
    from src.data.fetching.fetch_game_results import fetch_game_results
    games = fetch_game_results(args.season)
    if games is not None:
        games.to_csv(f'data/processed/game_results_{args.season}.csv', index=False)
        print(f"Fetched {len(games)} game records")
    
    # Test enhanced seeding
    predictor = EnhancedSeedingPredictor("data/processed/team_stats_with_fantasy_clean.csv")
    seeding = predictor.predict_seeding(args.season)
    
    print(f"\nEnhanced Seeding for {args.season}:")
    for conf in ['AFC', 'NFC']:
        if conf in seeding:
            print(f"\n{conf}:")
            for seed in seeding[conf]:
                print(f"  {seed.seed}. {seed.team} ({seed.win_pct:.3f})")


if __name__ == '__main__':
    main()

