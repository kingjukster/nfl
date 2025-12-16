"""
NFL Playoff Seeding and Bracket Prediction System

This module predicts:
1. Playoff seeding based on regular season performance
2. Playoff matchup win probabilities
3. Complete bracket simulation from Wild Card to Super Bowl
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NFL Conference and Division Structure
NFL_CONFERENCES = {
    'AFC': {
        'divisions': ['AFC_North', 'AFC_South', 'AFC_East', 'AFC_West'],
        'teams': {
            'AFC_North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_West': ['DEN', 'KC', 'LV', 'LAC']
        }
    },
    'NFC': {
        'divisions': ['NFC_North', 'NFC_South', 'NFC_East', 'NFC_West'],
        'teams': {
            'NFC_North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_West': ['ARI', 'LAR', 'SF', 'SEA']
        }
    }
}

# Stadium types for weather adjustments
DOME_TEAMS = ['ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'ARI', 'LAR', 'LV']
COLD_WEATHER_TEAMS = ['GB', 'CHI', 'MIN', 'BUF', 'NE', 'PIT', 'CLE', 'DEN']

# Team name mappings (handle variations)
TEAM_NAME_MAPPING = {
    'ARI': 'ARI', 'ARZ': 'ARI', 'Arizona': 'ARI',
    'ATL': 'ATL', 'Atlanta': 'ATL',
    'BAL': 'BAL', 'Baltimore': 'BAL',
    'BUF': 'BUF', 'Buffalo': 'BUF',
    'CAR': 'CAR', 'Carolina': 'CAR',
    'CHI': 'CHI', 'Chicago': 'CHI',
    'CIN': 'CIN', 'Cincinnati': 'CIN',
    'CLE': 'CLE', 'Cleveland': 'CLE',
    'DAL': 'DAL', 'Dallas': 'DAL',
    'DEN': 'DEN', 'Denver': 'DEN',
    'DET': 'DET', 'Detroit': 'DET',
    'GB': 'GB', 'GNB': 'GB', 'Green Bay': 'GB',
    'HOU': 'HOU', 'Houston': 'HOU',
    'IND': 'IND', 'Indianapolis': 'IND',
    'JAX': 'JAX', 'JAC': 'JAX', 'Jacksonville': 'JAX',
    'KC': 'KC', 'KAN': 'KC', 'Kansas City': 'KC',
    'LAC': 'LAC', 'SD': 'LAC', 'Los Angeles Chargers': 'LAC',
    'LAR': 'LAR', 'LA': 'LAR', 'STL': 'LAR', 'Los Angeles Rams': 'LAR',
    'LV': 'LV', 'OAK': 'LV', 'Las Vegas': 'LV', 'Oakland': 'LV',
    'MIA': 'MIA', 'Miami': 'MIA',
    'MIN': 'MIN', 'Minnesota': 'MIN',
    'NE': 'NE', 'NWE': 'NE', 'New England': 'NE',
    'NO': 'NO', 'NOR': 'NO', 'New Orleans': 'NO',
    'NYG': 'NYG', 'New York Giants': 'NYG',
    'NYJ': 'NYJ', 'NYJ': 'NYJ', 'New York Jets': 'NYJ',
    'PHI': 'PHI', 'Philadelphia': 'PHI',
    'PIT': 'PIT', 'Pittsburgh': 'PIT',
    'SF': 'SF', 'SFO': 'SF', 'San Francisco': 'SF',
    'SEA': 'SEA', 'Seattle': 'SEA',
    'TB': 'TB', 'TAM': 'TB', 'Tampa Bay': 'TB',
    'TEN': 'TEN', 'Tennessee': 'TEN',
    'WAS': 'WAS', 'WSH': 'WAS', 'Washington': 'WAS'
}


@dataclass
class PlayoffSeed:
    """Represents a playoff seed"""
    team: str
    conference: str
    seed: int
    division: str
    wins: float
    losses: float
    win_pct: float
    points_for: float = 0.0
    points_against: float = 0.0


@dataclass
class Matchup:
    """Represents a playoff matchup"""
    team1: str
    team2: str
    team1_seed: int
    team2_seed: int
    team1_win_prob: float
    team2_win_prob: float
    round: str
    conference: Optional[str] = None


class PlayoffPredictor:
    """Main class for predicting playoff seeding and outcomes"""
    
    def __init__(self, team_stats_path: str, win_prob_model_path: Optional[str] = None, 
                 game_results_path: Optional[str] = None):
        """
        Initialize the playoff predictor.
        
        Args:
            team_stats_path: Path to team statistics CSV
            win_prob_model_path: Optional path to saved win probability model
            game_results_path: Optional path to game-by-game results CSV
        """
        self.team_stats_path = Path(team_stats_path)
        self.win_prob_model_path = Path(win_prob_model_path) if win_prob_model_path else None
        self.game_results_path = game_results_path
        self.team_stats = None
        self.win_prob_model = None
        self.seeds = {}
        self.game_results = {}
        
    def load_team_stats(self, season: int) -> pd.DataFrame:
        """Load team statistics for a given season"""
        if not self.team_stats_path.exists():
            raise FileNotFoundError(f"Team stats file not found: {self.team_stats_path}")
        
        df = pd.read_csv(self.team_stats_path)
        
        # Filter to regular season only
        if 'season_type' in df.columns:
            df = df[df['season_type'] == 'REG'].copy()
        
        # Filter to requested season
        if 'season' in df.columns:
            df = df[df['season'] == season].copy()
        
        # If no data in main file, try to create minimal dataset from records
        if df.empty:
            logger.warning(f"No data in main file for season {season}, attempting to create from records")
            # Try to load records and create minimal dataset
            try:
                from src.fetch_team_records import fetch_team_records
                records = fetch_team_records(season)
                if records is not None and not records.empty:
                    # Create minimal dataframe with just team and win/loss data
                    df = records.copy()
                    # Add minimal required columns
                    if 'team' not in df.columns:
                        raise ValueError(f"Cannot create dataset for season {season}")
                    logger.info(f"Created minimal dataset from records for season {season}")
                else:
                    raise ValueError(f"No data found for season {season}")
            except Exception as e:
                logger.error(f"Could not create dataset from records: {e}")
                raise ValueError(f"No data found for season {season}")
        
        # Try to load win/loss data from multiple sources
        # 1. Try merged_file.csv
        merged_path = Path("data/processed/merged_file.csv")
        if merged_path.exists():
            try:
                merged_df = pd.read_csv(merged_path)
                if 'season' in merged_df.columns and 'team' in merged_df.columns:
                    merged_season = merged_df[merged_df['season'] == season].copy()
                    if 'season_type' in merged_season.columns:
                        merged_season = merged_season[merged_season['season_type'] == 'REG'].copy()
                    
                    if not merged_season.empty and 'team' in merged_season.columns:
                        # Normalize team names in merged data
                        merged_season['team'] = merged_season['team'].apply(self._normalize_team_name)
                        
                        # Merge win/loss columns if they exist
                        merge_cols = ['team']
                        for col in ['win', 'loss', 'tie', 'win_pct']:
                            if col in merged_season.columns:
                                merge_cols.append(col)
                        
                        if len(merge_cols) > 1:  # More than just 'team'
                            df = df.merge(
                                merged_season[merge_cols],
                                on='team',
                                how='left',
                                suffixes=('', '_merged')
                            )
                            # Use merged values if original doesn't have them
                            for col in ['win', 'loss', 'tie', 'win_pct']:
                                if f'{col}_merged' in df.columns:
                                    if col not in df.columns or df[col].isna().all():
                                        df[col] = df[f'{col}_merged']
                                    df = df.drop(columns=[f'{col}_merged'], errors='ignore')
            except Exception as e:
                logger.warning(f"Could not merge win/loss data from merged_file.csv: {e}")
        
        # 2. Try team_records.csv (if fetched separately) - check both generic and season-specific
        records_paths = [
            Path(f"data/processed/team_records_{season}.csv"),  # Season-specific
            Path("data/processed/team_records.csv")  # Generic
        ]
        
        for records_path in records_paths:
            if records_path.exists() and ('win' not in df.columns or df['win'].isna().all()):
                try:
                    records_df = pd.read_csv(records_path)
                    # If season column exists, filter; otherwise use all data
                    if 'season' in records_df.columns:
                        records_season = records_df[records_df['season'] == season].copy()
                    else:
                        records_season = records_df.copy()
                    
                    if not records_season.empty and 'team' in records_season.columns:
                        records_season['team'] = records_season['team'].apply(self._normalize_team_name)
                        merge_cols = ['team'] + [c for c in ['win', 'loss', 'tie', 'win_pct'] if c in records_season.columns]
                        if len(merge_cols) > 1:
                            df = df.merge(records_season[merge_cols], on='team', how='left', suffixes=('', '_records'))
                            for col in ['win', 'loss', 'tie', 'win_pct']:
                                if f'{col}_records' in df.columns:
                                    if col not in df.columns or df[col].isna().all():
                                        df[col] = df[f'{col}_records']
                                    df = df.drop(columns=[f'{col}_records'], errors='ignore')
                            logger.info(f"Loaded team records from {records_path}")
                            break  # Found records, no need to check other paths
                except Exception as e:
                    logger.warning(f"Could not load {records_path}: {e}")
                    continue
        
        # 3. Try fetching from nfl-data-py if still no data
        if 'win' not in df.columns or df['win'].isna().all():
            try:
                from src.fetch_team_records import fetch_team_records
                records = fetch_team_records(season)
                if records is not None and not records.empty:
                    records['team'] = records['team'].apply(self._normalize_team_name)
                    merge_cols = ['team'] + [c for c in ['win', 'loss', 'tie', 'win_pct'] if c in records.columns]
                    if len(merge_cols) > 1:
                        df = df.merge(records[merge_cols], on='team', how='left', suffixes=('', '_fetched'))
                        for col in ['win', 'loss', 'tie', 'win_pct']:
                            if f'{col}_fetched' in df.columns:
                                if col not in df.columns or df[col].isna().all():
                                    df[col] = df[f'{col}_fetched']
                                df = df.drop(columns=[f'{col}_fetched'], errors='ignore')
            except Exception as e:
                logger.warning(f"Could not fetch team records: {e}")
        
        # Normalize team names
        if 'team' in df.columns:
            df['team'] = df['team'].apply(self._normalize_team_name)
        
        logger.info(f"Loaded {len(df)} teams for season {season}")
        return df
    
    def load_game_results(self, season: int) -> pd.DataFrame:
        """Load game-by-game results for tiebreaker calculations"""
        if season in self.game_results:
            return self.game_results[season]
        
        # Try multiple paths
        paths_to_try = []
        
        if self.game_results_path:
            paths_to_try.append(Path(self.game_results_path))
        
        paths_to_try.extend([
            Path(f"data/processed/game_results_{season}.csv"),
            Path("data/processed/game_results.csv")
        ])
        
        for path in paths_to_try:
            if path.exists():
                try:
                    games = pd.read_csv(path)
                    if 'season' in games.columns:
                        games = games[games['season'] == season].copy()
                    self.game_results[season] = games
                    logger.info(f"Loaded game results from {path}")
                    return games
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    continue
        
        logger.warning(f"No game results found for season {season}")
        return pd.DataFrame()
    
    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to standard abbreviation"""
        team_upper = str(team).upper().strip()
        return TEAM_NAME_MAPPING.get(team_upper, team_upper)
    
    def calculate_win_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate win percentage from wins/losses/ties.
        Uses same logic as heatMap2.py to handle win_pct_off/win_pct_def variants.
        """
        df = df.copy()
        
        # 1) Try direct win_pct columns (same as heatMap2.py)
        for cand in ['win_pct', 'win_pct_off', 'win_pct_def']:
            if cand in df.columns:
                if cand != 'win_pct':
                    df['win_pct'] = df[cand]
                # Validate win_pct is reasonable (should be 0-1)
                if df['win_pct'].max() <= 1.0:
                    return df
                # If values are > 1, they might be percentages (0-100), convert
                if df['win_pct'].max() > 1.0:
                    df['win_pct'] = df['win_pct'] / 100.0
                return df
        
        # 2) Calculate from wins/losses/ties (check for suffixed variants too)
        def pick(col):
            for c in [col, f"{col}_off", f"{col}_def"]:
                if c in df.columns:
                    return c
            return None
        
        win_col = pick('win')
        loss_col = pick('loss')
        tie_col = pick('tie')
        
        if win_col and loss_col:
            wins = pd.to_numeric(df[win_col], errors='coerce').fillna(0)
            losses = pd.to_numeric(df[loss_col], errors='coerce').fillna(0)
            ties = pd.to_numeric(df[tie_col], errors='coerce').fillna(0) if tie_col else 0
            
            total_games = wins + losses + ties
            df['win_pct'] = np.where(total_games > 0, (wins + 0.5 * ties) / total_games, 0.0)
            
            # Validate - if win_pct seems wrong, log warning
            if df['win_pct'].max() > 1.0:
                logger.warning(f"Win_pct values > 1.0 detected (max: {df['win_pct'].max()}), may need normalization")
                # Try dividing by 100 if it looks like percentages
                if df['win_pct'].max() > 10.0:
                    df['win_pct'] = df['win_pct'] / 100.0
        else:
            # Fallback: use points if available (but this is less reliable)
            if 'total_off_points' in df.columns:
                # Normalize points to 0-1 range as proxy for win_pct
                max_points = df['total_off_points'].max()
                if max_points > 0:
                    df['win_pct'] = (df['total_off_points'] / max_points).clip(0, 1)
                    logger.warning("Using points as proxy for win_pct - results may be inaccurate")
                else:
                    df['win_pct'] = 0.5
            else:
                df['win_pct'] = 0.5
                logger.warning("No win/loss data found, using default win_pct=0.5")
        
        return df
    
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
    
    def calculate_strength_of_schedule(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        Calculate strength of schedule adjustment.
        Teams with harder schedules get a boost to their win_pct.
        """
        df = df.copy()
        game_data = self.load_game_results(season)
        
        if game_data.empty:
            logger.warning("No game data for SOS calculation, skipping")
            df['sos'] = 0.5
            df['sos_adjustment'] = 0.0
            return df
        
        # Calculate opponent win_pct for each team
        opponent_wp = {}
        for team in df['team'].unique():
            # Get all opponents this team played
            team_games = game_data[game_data['team'] == team]
            opponents = team_games['opponent'].unique()
            
            # Calculate average opponent win_pct
            opp_wp_list = []
            for opp in opponents:
                opp_data = df[df['team'] == opp]
                if not opp_data.empty and 'win_pct' in opp_data.columns:
                    opp_wp_list.append(opp_data.iloc[0]['win_pct'])
            
            if opp_wp_list:
                opponent_wp[team] = np.mean(opp_wp_list)
            else:
                opponent_wp[team] = df['win_pct'].mean()  # Fallback to league average
        
        df['sos'] = df['team'].map(opponent_wp).fillna(df['win_pct'].mean())
        
        # Calculate adjustment: teams with harder schedules (higher SOS) get boost
        # Use a more conservative adjustment to avoid over-correction
        league_avg_sos = df['sos'].mean()
        df['sos_adjustment'] = (df['sos'] - league_avg_sos) * 0.08  # Reduced from 0.1 to 0.08
        df['win_pct_sos_adjusted'] = (df['win_pct'] + df['sos_adjustment']).clip(0, 1)
        
        # Also calculate strength of victory (SOV) for tiebreakers
        if not game_data.empty:
            sov_dict = {}
            for team in df['team'].unique():
                # Get teams this team beat
                wins = game_data[(game_data['team'] == team) & (game_data['result'] == 'W')]
                beaten_teams = wins['opponent'].unique()
                
                if len(beaten_teams) > 0:
                    beaten_wp_list = []
                    for beaten_team in beaten_teams:
                        beaten_data = df[df['team'] == beaten_team]
                        if not beaten_data.empty and 'win_pct' in beaten_data.columns:
                            beaten_wp_list.append(beaten_data.iloc[0]['win_pct'])
                    
                    if beaten_wp_list:
                        sov_dict[team] = np.mean(beaten_wp_list)
                    else:
                        sov_dict[team] = df['win_pct'].mean()
                else:
                    sov_dict[team] = df['win_pct'].mean()
            
            df['sov'] = df['team'].map(sov_dict).fillna(df['win_pct'].mean())
        else:
            df['sov'] = df['win_pct']  # Fallback
        
        return df
    
    def apply_tiebreakers(self, teams_df: pd.DataFrame, tied_teams: List[str], 
                         game_data: pd.DataFrame) -> List[str]:
        """
        Apply NFL tiebreaker rules to break ties.
        Returns sorted list of teams (best to worst).
        """
        if len(tied_teams) == 1:
            return tied_teams
        
        if game_data.empty:
            # Fallback to points_for
            tied_df = teams_df[teams_df['team'].isin(tied_teams)]
            points_col = None
            for col in ['total_off_points', 'points_for', 'PF']:
                if col in tied_df.columns:
                    points_col = col
                    break
            if points_col:
                return tied_df.sort_values(points_col, ascending=False)['team'].tolist()
            return tied_teams
        
        # 1. Head-to-head record (if all tied teams played each other)
        h2h_records = {}
        for team in tied_teams:
            h2h_records[team] = {'wins': 0, 'losses': 0, 'ties': 0}
            vs_tied = game_data[
                (game_data['team'] == team) & 
                (game_data['opponent'].isin(tied_teams))
            ]
            for _, game in vs_tied.iterrows():
                if game['result'] == 'W':
                    h2h_records[team]['wins'] += 1
                elif game['result'] == 'L':
                    h2h_records[team]['losses'] += 1
                else:
                    h2h_records[team]['ties'] += 1
        
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
        
        if all_played and len(tied_teams) <= 3:
            h2h_win_pct = {}
            for team in tied_teams:
                rec = h2h_records[team]
                total = rec['wins'] + rec['losses'] + rec['ties']
                if total > 0:
                    h2h_win_pct[team] = (rec['wins'] + 0.5 * rec['ties']) / total
                else:
                    h2h_win_pct[team] = 0.0
            
            sorted_teams = sorted(tied_teams, key=lambda t: h2h_win_pct[t], reverse=True)
            if len(set(h2h_win_pct.values())) == len(h2h_win_pct):
                return sorted_teams
        
        # 2. Division record (if same division)
        same_division = True
        division = self.get_team_division(tied_teams[0])
        for team in tied_teams[1:]:
            if self.get_team_division(team) != division:
                same_division = False
                break
        
        if same_division and division:
            conf = self.get_team_conference(tied_teams[0])
            division_teams = NFL_CONFERENCES[conf]['teams'][division]
            div_records = {}
            for team in tied_teams:
                div_games = game_data[
                    (game_data['team'] == team) &
                    (game_data['opponent'].isin(division_teams))
                ]
                wins = len(div_games[div_games['result'] == 'W'])
                losses = len(div_games[div_games['result'] == 'L'])
                ties = len(div_games[div_games['result'] == 'T'])
                total = wins + losses + ties
                if total > 0:
                    div_records[team] = (wins + 0.5 * ties) / total
                else:
                    div_records[team] = 0.0
            
            sorted_teams = sorted(tied_teams, key=lambda t: div_records[t], reverse=True)
            if len(set(div_records.values())) == len(div_records):
                return sorted_teams
        
        # 3. Conference record
        conf_records = {}
        for team in tied_teams:
            conf = self.get_team_conference(team)
            if not conf:
                conf_records[team] = 0.0
                continue
            
            conf_teams = []
            for div, teams in NFL_CONFERENCES[conf]['teams'].items():
                conf_teams.extend(teams)
            
            conf_games = game_data[
                (game_data['team'] == team) &
                (game_data['opponent'].isin(conf_teams))
            ]
            wins = len(conf_games[conf_games['result'] == 'W'])
            losses = len(conf_games[conf_games['result'] == 'L'])
            ties = len(conf_games[conf_games['result'] == 'T'])
            total = wins + losses + ties
            if total > 0:
                conf_records[team] = (wins + 0.5 * ties) / total
            else:
                conf_records[team] = 0.0
        
        sorted_teams = sorted(tied_teams, key=lambda t: conf_records[t], reverse=True)
        if len(set(conf_records.values())) == len(conf_records):
            return sorted_teams
        
        # 4. Common games (min 4 games)
        common_games_records = {}
        for team in tied_teams:
            # Get opponents for this team
            team_opponents = set(game_data[game_data['team'] == team]['opponent'].unique())
            
            # Find common opponents with other tied teams
            common_opponents = team_opponents.copy()
            for other_team in tied_teams:
                if other_team != team:
                    other_opponents = set(game_data[game_data['team'] == other_team]['opponent'].unique())
                    common_opponents = common_opponents.intersection(other_opponents)
            
            # Calculate record vs common opponents (need at least 4 games)
            if len(common_opponents) >= 4:
                common_games = game_data[
                    (game_data['team'] == team) &
                    (game_data['opponent'].isin(common_opponents))
                ]
                wins = len(common_games[common_games['result'] == 'W'])
                losses = len(common_games[common_games['result'] == 'L'])
                ties = len(common_games[common_games['result'] == 'T'])
                total = wins + losses + ties
                if total >= 4:  # NFL requires min 4 common games
                    common_games_records[team] = (wins + 0.5 * ties) / total if total > 0 else 0.0
                else:
                    common_games_records[team] = None
            else:
                common_games_records[team] = None
        
        # If all teams have common games records, use them
        if all(v is not None for v in common_games_records.values()):
            sorted_teams = sorted(tied_teams, key=lambda t: common_games_records[t] or 0, reverse=True)
            if len(set(v for v in common_games_records.values() if v is not None)) == len([v for v in common_games_records.values() if v is not None]):
                return sorted_teams
        
        # 5. Strength of victory (win_pct of teams they beat)
        sov_records = {}
        for team in tied_teams:
            # Get teams this team beat
            wins = game_data[(game_data['team'] == team) & (game_data['result'] == 'W')]
            beaten_teams = wins['opponent'].unique()
            
            if len(beaten_teams) > 0:
                # Calculate average win_pct of beaten teams
                beaten_wp_list = []
                for beaten_team in beaten_teams:
                    beaten_data = teams_df[teams_df['team'] == beaten_team]
                    if not beaten_data.empty and 'win_pct' in beaten_data.columns:
                        beaten_wp_list.append(beaten_data.iloc[0]['win_pct'])
                
                if beaten_wp_list:
                    sov_records[team] = np.mean(beaten_wp_list)
                else:
                    sov_records[team] = 0.0
            else:
                sov_records[team] = 0.0
        
        if sov_records:
            sorted_teams = sorted(tied_teams, key=lambda t: sov_records.get(t, 0), reverse=True)
            if len(set(sov_records.values())) == len(sov_records):
                return sorted_teams
        
        # 6. Strength of schedule (already calculated in df)
        tied_df = teams_df[teams_df['team'].isin(tied_teams)]
        if 'sos' in tied_df.columns:
            sorted_teams = tied_df.sort_values('sos', ascending=False)['team'].tolist()
            if len(set(tied_df['sos'].values)) == len(tied_df):
                return sorted_teams
        
        # 7. Points scored
        points_col = None
        for col in ['total_off_points', 'points_for', 'PF']:
            if col in tied_df.columns:
                points_col = col
                break
        
        if points_col:
            return tied_df.sort_values(points_col, ascending=False)['team'].tolist()
        
        # 8. Points allowed (reverse - fewer is better)
        pa_col = None
        for col in ['total_def_points', 'points_against', 'PA']:
            if col in tied_df.columns:
                pa_col = col
                break
        
        if pa_col:
            return tied_df.sort_values(pa_col, ascending=True)['team'].tolist()
        
        return tied_teams
    
    def predict_seeding(self, season: int, conference: str = None) -> Dict[str, List[PlayoffSeed]]:
        """
        Predict playoff seeding for a given season.
        
        Args:
            season: Season year
            conference: 'AFC', 'NFC', or None for both
        
        Returns:
            Dictionary mapping conference to list of seeds (1-7)
        """
        df = self.load_team_stats(season)
        df = self.calculate_win_pct(df)
        
        # Add strength of schedule adjustment
        df = self.calculate_strength_of_schedule(df, season)
        
        # Add momentum features
        df = self.add_momentum_features(df, season)
        
        # Use SOS-adjusted win_pct for seeding if available
        if 'win_pct_sos_adjusted' in df.columns:
            df['win_pct'] = df['win_pct_sos_adjusted']
        
        # Adjust win_pct by momentum (small boost for hot teams)
        if 'momentum_score' in df.columns:
            df['win_pct'] = (df['win_pct'] + df['momentum_score'] * 0.05).clip(0, 1)
        
        # Ensure we have team column
        if 'team' not in df.columns:
            raise ValueError("Team statistics must include 'team' column")
        
        # Get points for tiebreakers
        points_for_col = None
        points_against_col = None
        for col in ['total_off_points', 'points_for', 'PF', 'points']:
            if col in df.columns:
                points_for_col = col
                break
        for col in ['total_def_points', 'points_against', 'PA', 'points_allowed']:
            if col in df.columns:
                points_against_col = col
                break
        
        # Load game results for tiebreakers
        game_data = self.load_game_results(season)
        
        seeding = {}
        conferences_to_process = [conference] if conference else ['AFC', 'NFC']
        
        for conf in conferences_to_process:
            if conf not in NFL_CONFERENCES:
                logger.warning(f"Unknown conference: {conf}")
                continue
            
            conf_teams = []
            for division, teams in NFL_CONFERENCES[conf]['teams'].items():
                for team in teams:
                    conf_teams.append(team)
            
            # Filter to teams in this conference
            conf_df = df[df['team'].isin(conf_teams)].copy()
            
            if conf_df.empty:
                logger.warning(f"No teams found for {conf} in season {season}")
                continue
            
            # Get division winners (top team by win_pct in each division)
            division_winners = []
            for division, teams in NFL_CONFERENCES[conf]['teams'].items():
                div_df = conf_df[conf_df['team'].isin(teams)].copy()
                if not div_df.empty:
                    # Sort by win_pct, then points_for if available
                    sort_cols = ['win_pct']
                    ascending_vals = [False]
                    if points_for_col and points_for_col in div_df.columns:
                        sort_cols.append(points_for_col)
                        ascending_vals.append(False)
                    
                    div_df = div_df.sort_values(sort_cols, ascending=ascending_vals)
                    winner = div_df.iloc[0]
                    division_winners.append({
                        'team': winner['team'],
                        'division': division,
                        'win_pct': winner['win_pct'],
                        'wins': winner.get('win', winner.get('wins', 0)),
                        'losses': winner.get('loss', winner.get('losses', 0)),
                        'points_for': winner.get(points_for_col, 0) if points_for_col else 0,
                        'points_against': winner.get(points_against_col, 0) if points_against_col else 0
                    })
            
            # Get wild card teams (non-division winners, top 3 by win_pct with tiebreakers)
            division_winner_teams = [dw['team'] for dw in division_winners]
            wildcard_df = conf_df[~conf_df['team'].isin(division_winner_teams)].copy()
            
            if not wildcard_df.empty:
                # Sort by win_pct first
                wildcard_df = wildcard_df.sort_values('win_pct', ascending=False)
                
                # Group by win_pct and apply tiebreakers to tied groups
                wc_by_wp = {}
                for _, row in wildcard_df.iterrows():
                    wp = round(row['win_pct'], 3)
                    if wp not in wc_by_wp:
                        wc_by_wp[wp] = []
                    wc_by_wp[wp].append(row['team'])
                
                # Apply tiebreakers and rebuild sorted list
                final_wc_teams = []
                for wp in sorted(wc_by_wp.keys(), reverse=True):
                    tied_wc = wc_by_wp[wp]
                    if len(tied_wc) > 1:
                        sorted_wc = self.apply_tiebreakers(wildcard_df, tied_wc, game_data)
                        final_wc_teams.extend(sorted_wc)
                    else:
                        final_wc_teams.extend(tied_wc)
                
                # Reorder wildcard_df based on tiebreaker results
                team_order = {team: i for i, team in enumerate(final_wc_teams)}
                wildcard_df['_sort_order'] = wildcard_df['team'].map(team_order)
                wildcard_df = wildcard_df.sort_values('_sort_order').drop(columns=['_sort_order'])
            
            # Combine division winners and wild cards
            all_playoff_teams = []
            
            # Seeds 1-4: Division winners (sorted by win_pct, apply tiebreakers if needed)
            division_winners_sorted = sorted(division_winners, 
                                            key=lambda x: (x['win_pct'], x['points_for']), 
                                            reverse=True)
            
            # Group by win_pct and apply tiebreakers
            dw_by_wp = {}
            for dw in division_winners_sorted:
                wp = round(dw['win_pct'], 3)  # Round to avoid floating point issues
                if wp not in dw_by_wp:
                    dw_by_wp[wp] = []
                dw_by_wp[wp].append(dw)
            
            # Apply tiebreakers to tied groups
            final_dw_order = []
            for wp in sorted(dw_by_wp.keys(), reverse=True):
                tied_dw = dw_by_wp[wp]
                if len(tied_dw) > 1:
                    # Apply tiebreakers
                    tied_teams = [dw['team'] for dw in tied_dw]
                    sorted_teams = self.apply_tiebreakers(df, tied_teams, game_data)
                    # Reorder tied_dw based on tiebreaker results
                    team_to_dw = {dw['team']: dw for dw in tied_dw}
                    tied_dw = [team_to_dw[team] for team in sorted_teams if team in team_to_dw]
                final_dw_order.extend(tied_dw)
            
            for i, dw in enumerate(final_dw_order[:4], 1):
                # Get team data from df for accurate wins/losses
                team_data = df[df['team'] == dw['team']]
                if not team_data.empty:
                    team_row = team_data.iloc[0]
                    wins = team_row.get('win', team_row.get('wins', dw['wins']))
                    losses = team_row.get('loss', team_row.get('losses', dw['losses']))
                else:
                    wins = dw['wins']
                    losses = dw['losses']
                
                all_playoff_teams.append(PlayoffSeed(
                    team=dw['team'],
                    conference=conf,
                    seed=i,
                    division=dw['division'],
                    wins=wins,
                    losses=losses,
                    win_pct=dw['win_pct'],
                    points_for=dw['points_for'],
                    points_against=dw['points_against']
                ))
            
            # Seeds 5-7: Wild card teams (apply tiebreakers if needed)
            wildcard_list = wildcard_df.head(3).to_dict('records')
            
            # Group by win_pct and apply tiebreakers
            wc_by_wp = {}
            for wc in wildcard_list:
                wp = round(wc['win_pct'], 3)
                if wp not in wc_by_wp:
                    wc_by_wp[wp] = []
                wc_by_wp[wp].append(wc)
            
            # Apply tiebreakers to tied groups
            final_wc_order = []
            for wp in sorted(wc_by_wp.keys(), reverse=True):
                tied_wc = wc_by_wp[wp]
                if len(tied_wc) > 1:
                    tied_teams = [wc['team'] for wc in tied_wc]
                    sorted_teams = self.apply_tiebreakers(df, tied_teams, game_data)
                    team_to_wc = {wc['team']: wc for wc in tied_wc}
                    tied_wc = [team_to_wc[team] for team in sorted_teams if team in team_to_wc]
                final_wc_order.extend(tied_wc)
            
            for i, wc in enumerate(final_wc_order[:3]):
                # Get team data from df for accurate wins/losses
                team_data = df[df['team'] == wc['team']]
                if not team_data.empty:
                    team_row = team_data.iloc[0]
                    wins = team_row.get('win', team_row.get('wins', wc.get('win', 0)))
                    losses = team_row.get('loss', team_row.get('losses', wc.get('loss', 0)))
                else:
                    wins = wc.get('win', wc.get('wins', 0))
                    losses = wc.get('loss', wc.get('losses', 0))
                
                all_playoff_teams.append(PlayoffSeed(
                    team=wc['team'],
                    conference=conf,
                    seed=len(all_playoff_teams) + 1,
                    division='Wild Card',
                    wins=wins,
                    losses=losses,
                    win_pct=wc['win_pct'],
                    points_for=wc.get(points_for_col, 0) if points_for_col else 0,
                    points_against=wc.get(points_against_col, 0) if points_against_col else 0
                ))
            
            seeding[conf] = all_playoff_teams[:7]  # Top 7 teams
            
            logger.info(f"{conf} Seeding:")
            for seed in seeding[conf]:
                logger.info(f"  {seed.seed}. {seed.team} ({seed.win_pct:.3f}, {seed.wins}-{seed.losses})")
        
        self.seeds = seeding
        return seeding
    
    def load_historical_playoff_data(self) -> pd.DataFrame:
        """
        Load historical playoff data for better experience tracking.
        Returns DataFrame with team, season, playoff_appearances, playoff_wins, etc.
        """
        # Try to load from file
        playoff_data_path = Path("data/processed/historical_playoff_data.csv")
        if playoff_data_path.exists():
            try:
                return pd.read_csv(playoff_data_path)
            except Exception as e:
                logger.warning(f"Could not load historical playoff data: {e}")
        
        # If not available, return empty DataFrame
        # In production, you'd fetch this from nfl-data-py or build a database
        return pd.DataFrame()
    
    def _add_playoff_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add playoff experience features based on historical playoff data.
        Tries to load from historical data, falls back to hardcoded values.
        """
        df = df.copy()
        
        # Try to load historical playoff data
        historical_data = self.load_historical_playoff_data()
        
        # Initialize with zeros
        df['playoff_appearances_3yr'] = 0
        df['playoff_wins_3yr'] = 0
        df['super_bowl_appearances_5yr'] = 0
        
        if not historical_data.empty and 'team' in historical_data.columns:
            # Use historical data if available
            for _, row in df.iterrows():
                team = row['team']
                season = row.get('season', None)
                
                if season:
                    # Get playoff data for this team in recent seasons
                    team_playoff = historical_data[
                        (historical_data['team'] == team) &
                        (historical_data['season'] >= season - 3) &
                        (historical_data['season'] < season)
                    ]
                    
                    if not team_playoff.empty:
                        df.loc[df['team'] == team, 'playoff_appearances_3yr'] = len(team_playoff)
                        df.loc[df['team'] == team, 'playoff_wins_3yr'] = team_playoff.get('playoff_wins', 0).sum()
                        
                        # Super Bowl appearances in last 5 years
                        sb_data = historical_data[
                            (historical_data['team'] == team) &
                            (historical_data['season'] >= season - 5) &
                            (historical_data['season'] < season) &
                            (historical_data.get('super_bowl_appearance', False) == True)
                        ]
                        df.loc[df['team'] == team, 'super_bowl_appearances_5yr'] = len(sb_data)
        else:
            # Fallback to hardcoded values (simplified)
            playoff_experience = {
                'KC': {'appearances_3yr': 3, 'wins_3yr': 8, 'sb_appearances_5yr': 3},
                'BUF': {'appearances_3yr': 3, 'wins_3yr': 3, 'sb_appearances_5yr': 0},
                'SF': {'appearances_3yr': 3, 'wins_3yr': 5, 'sb_appearances_5yr': 1},
                'CIN': {'appearances_3yr': 2, 'wins_3yr': 2, 'sb_appearances_5yr': 1},
                'PHI': {'appearances_3yr': 2, 'wins_3yr': 2, 'sb_appearances_5yr': 1},
                'DAL': {'appearances_3yr': 3, 'wins_3yr': 1, 'sb_appearances_5yr': 0},
                'TB': {'appearances_3yr': 3, 'wins_3yr': 5, 'sb_appearances_5yr': 1},
                'BAL': {'appearances_3yr': 2, 'wins_3yr': 0, 'sb_appearances_5yr': 0},
                'GB': {'appearances_3yr': 2, 'wins_3yr': 1, 'sb_appearances_5yr': 0},
                'LAR': {'appearances_3yr': 2, 'wins_3yr': 4, 'sb_appearances_5yr': 2},
            }
            
            for team, exp in playoff_experience.items():
                mask = df['team'] == team
                if mask.any():
                    df.loc[mask, 'playoff_appearances_3yr'] = exp['appearances_3yr']
                    df.loc[mask, 'playoff_wins_3yr'] = exp['wins_3yr']
                    df.loc[mask, 'super_bowl_appearances_5yr'] = exp['sb_appearances_5yr']
        
        # Calculate playoff experience boost
        df['playoff_experience_boost'] = (
            df['playoff_appearances_3yr'] * 0.02 +
            df['playoff_wins_3yr'] * 0.03 +
            df['super_bowl_appearances_5yr'] * 0.05
        ).clip(0, 0.15)  # Max 15% boost
        
        return df
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add playoff-specific engineered features."""
        df = df.copy()
        
        # Offensive efficiency
        if 'total_off_points' in df.columns and 'offense_snaps' in df.columns:
            df['offensive_efficiency'] = (
                df['total_off_points'] / df['offense_snaps'].replace(0, 1)
            )
        
        # Defensive efficiency
        if 'total_def_points' in df.columns and 'defense_snaps' in df.columns:
            df['defensive_efficiency'] = (
                df['total_def_points'] / df['defense_snaps'].replace(0, 1)
            )
        
        # Turnover differential
        int_def = df.get('interception_def', pd.Series(0, index=df.index))
        fum_def = df.get('fumble_def', pd.Series(0, index=df.index))
        int_off = df.get('interception_off', pd.Series(0, index=df.index))
        fum_off = df.get('fumble_off', pd.Series(0, index=df.index))
        df['turnover_diff'] = (int_def + fum_def) - (int_off + fum_off)
        
        # Points per drive (approximate)
        if 'total_off_points' in df.columns:
            estimated_drives = df.get('offense_snaps', 60) / 10  # ~10 snaps per drive
            df['points_per_drive'] = df['total_off_points'] / estimated_drives.replace(0, 1)
        
        # Add playoff experience
        df = self._add_playoff_experience(df)
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        Add momentum/trend features based on recent performance.
        Uses game-by-game data if available, otherwise uses season-level approximations.
        """
        df = df.copy()
        game_data = self.load_game_results(season)
        
        # Initialize momentum features
        df['win_streak'] = 0
        df['last_4_games_wp'] = df['win_pct']  # Default to season average
        df['momentum_score'] = 0.0
        
        if not game_data.empty:
            for team in df['team'].unique():
                team_games = game_data[game_data['team'] == team].copy()
                if team_games.empty:
                    continue
                
                # Sort by week if available
                if 'week' in team_games.columns:
                    team_games = team_games.sort_values('week')
                
                # Calculate win streak (from most recent games)
                streak = 0
                for _, game in team_games.tail(10).iterrows():  # Check last 10 games
                    if game['result'] == 'W':
                        streak += 1
                    elif game['result'] == 'L':
                        break
                    # Ties don't break streak but don't add to it
                
                df.loc[df['team'] == team, 'win_streak'] = streak
                
                # Last 4 games win_pct
                last_4 = team_games.tail(4)
                if len(last_4) > 0:
                    wins = len(last_4[last_4['result'] == 'W'])
                    losses = len(last_4[last_4['result'] == 'L'])
                    ties = len(last_4[last_4['result'] == 'T'])
                    total = wins + losses + ties
                    if total > 0:
                        last_4_wp = (wins + 0.5 * ties) / total
                        df.loc[df['team'] == team, 'last_4_games_wp'] = last_4_wp
                
                # Momentum score: combination of streak and recent performance
                team_idx = df[df['team'] == team].index[0]
                momentum = (streak * 0.1) + ((df.loc[team_idx, 'last_4_games_wp'] - df.loc[team_idx, 'win_pct']) * 0.5)
                df.loc[df['team'] == team, 'momentum_score'] = momentum
        
        return df
    
    def calculate_weighted_win_pct(self, df: pd.DataFrame, 
                                   recent_weight: float = 0.7) -> pd.DataFrame:
        """
        Calculate win percentage with more weight on recent performance.
        Increased default weight from 0.6 to 0.7 to better capture recent form.
        """
        df = df.copy()
        
        # If we have multiple seasons, weight recent ones more
        if 'season' in df.columns:
            df = df.sort_values(['team', 'season'])
            df['win_pct_recent'] = df.groupby('team')['win_pct'].rolling(
                window=2, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Weighted average (increased recent weight)
            df['win_pct_weighted'] = (
                recent_weight * df['win_pct_recent'] + 
                (1 - recent_weight) * df['win_pct']
            )
        else:
            df['win_pct_weighted'] = df['win_pct']
        
        return df
    
    def _load_win_prob_model(self, team_stats_df: pd.DataFrame):
        """
        Load or create improved win probability model using XGBoost.
        
        Returns the trained model and feature list
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            from itertools import combinations
            
            # Try XGBoost first, fallback to Random Forest, then GaussianNB
            try:
                from xgboost import XGBClassifier
                USE_XGBOOST = True
                logger.info("Using XGBoost for win probability model")
            except ImportError:
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    USE_XGBOOST = False
                    USE_RF = True
                    logger.info("XGBoost not available, using Random Forest")
                except ImportError:
                    from sklearn.naive_bayes import GaussianNB
                    USE_XGBOOST = False
                    USE_RF = False
                    logger.info("Using Gaussian Naive Bayes (fallback)")
            
            # Note: Engineered features should already be added before calling this method
            # Use same feature selection as heatMap2.py
            EXCLUDE = set(['team', 'season', 'season_type', 'win_pct', 'win', 'loss', 'tie', 
                          'record', 'win_off', 'loss_off', 'tie_off', 'record_off',
                          'win_def', 'loss_def', 'tie_def', 'record_def',
                          'win_pct_off', 'win_pct_def', 'win_pct_recent', 'win_pct_weighted'])
            
            numeric_cols = team_stats_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in EXCLUDE]
            
            if not features:
                logger.warning("No features available for win probability model")
                return None, None
            
            # Build pairwise training data
            pairs_X, pairs_y = [], []
            for season, G in team_stats_df.groupby('season'):
                G = G.reset_index(drop=True)
                Xs = G[features].fillna(0).to_numpy()
                ys = G['win_pct'].astype(float).to_numpy()
                for i, j in combinations(range(len(G)), 2):
                    d = Xs[i] - Xs[j]
                    y = 1 if ys[i] > ys[j] else 0
                    pairs_X.append(d)
                    pairs_y.append(y)
                    pairs_X.append(-d)
                    pairs_y.append(1 - y)
            
            if not pairs_X:
                logger.warning("No pairwise data for training win probability model")
                return None, None
            
            pairs_X = np.asarray(pairs_X)
            pairs_y = np.asarray(pairs_y)
            
            # Train improved model
            if USE_XGBOOST:
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='logloss'
                    )
                )
            elif USE_RF:
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            else:
                from sklearn.naive_bayes import GaussianNB
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True), 
                    GaussianNB()
                )
            
            clf.fit(pairs_X, pairs_y)
            
            logger.info(f"Trained improved win probability model with {len(features)} features")
            return clf, features
            
        except Exception as e:
            logger.warning(f"Error creating win probability model: {e}")
            return None, None
    
    def _get_stadium_factor(self, team1: str, team2: str, 
                           home_team: str, month: int = 1) -> float:
        """Get weather/stadium adjustment factor."""
        factor = 0.0
        
        # Dome teams have advantage in domes
        if home_team in DOME_TEAMS:
            if team1 in DOME_TEAMS and team2 not in DOME_TEAMS:
                factor += 0.02
            elif team2 in DOME_TEAMS and team1 not in DOME_TEAMS:
                factor -= 0.02
        
        # Cold weather teams have advantage in cold weather
        if month in [12, 1, 2]:  # Winter months
            if home_team in COLD_WEATHER_TEAMS:
                if team1 in COLD_WEATHER_TEAMS and team2 not in COLD_WEATHER_TEAMS:
                    factor += 0.015
                elif team2 in COLD_WEATHER_TEAMS and team1 not in COLD_WEATHER_TEAMS:
                    factor -= 0.015
        
        return factor
    
    def predict_matchup_win_prob(self, team1: str, team2: str, 
                                 team_stats_df: pd.DataFrame,
                                 home_team: str = None,
                                 win_prob_model=None, features=None,
                                 month: int = 1) -> float:
        """
        Predict win probability for team1 vs team2 with improvements.
        
        Improvements:
        - Uses XGBoost/Random Forest model if available
        - Applies home field advantage
        - Applies weather/stadium factors
        """
        # Try to get team stats
        team1_stats = team_stats_df[team_stats_df['team'] == team1]
        team2_stats = team_stats_df[team_stats_df['team'] == team2]
        
        if team1_stats.empty or team2_stats.empty:
            logger.warning(f"Missing stats for {team1} or {team2}, using default 0.5")
            return 0.5
        
        team1_stats = team1_stats.iloc[0]
        team2_stats = team2_stats.iloc[0]
        
        # Get base probability
        base_prob = None
        
        # If we have a win probability model, use it
        if win_prob_model is not None and features is not None:
            try:
                # Get feature values for both teams
                team1_feats = team1_stats[features].fillna(0).values
                team2_feats = team2_stats[features].fillna(0).values
                
                # Calculate difference (team1 - team2)
                diff = (team1_feats - team2_feats).reshape(1, -1)
                
                # Predict probability
                base_prob = win_prob_model.predict_proba(diff)[0, 1]
            except Exception as e:
                logger.warning(f"Error using win prob model: {e}, falling back to simple method")
        
        # Fallback to simple method if model failed
        if base_prob is None:
            team1_wp = team1_stats.get('win_pct', 0.5)
            team2_wp = team2_stats.get('win_pct', 0.5)
            wp_diff = team1_wp - team2_wp
            base_prob = 1 / (1 + np.exp(-5 * wp_diff))
        
        # Apply playoff experience boost
        team1_exp = team1_stats.get('playoff_experience_boost', 0)
        team2_exp = team2_stats.get('playoff_experience_boost', 0)
        exp_diff = team1_exp - team2_exp
        
        # Adjust probability based on experience difference
        # Teams with more experience get a boost
        base_prob += exp_diff * 0.5  # Scale experience boost
        
        # Apply home field advantage
        if home_team:
            home_advantage = 0.07  # ~7% boost for home team
            if home_team == team1:
                adjusted_prob = base_prob + home_advantage * (1 - base_prob)
            elif home_team == team2:
                adjusted_prob = base_prob - home_advantage * base_prob
            else:
                adjusted_prob = base_prob
        else:
            adjusted_prob = base_prob
        
        # Apply weather/stadium factors (smaller impact)
        if home_team:
            weather_factor = self._get_stadium_factor(team1, team2, home_team, month)
            adjusted_prob += weather_factor
        
        return np.clip(adjusted_prob, 0.05, 0.95)
    
    def simulate_playoff_round(self, seeds: List[PlayoffSeed], round_name: str,
                              team_stats_df: pd.DataFrame) -> List[str]:
        """
        Simulate a playoff round and return winners.
        
        Args:
            seeds: List of teams/seeds in the round
            round_name: Name of the round (e.g., 'Wild Card', 'Divisional', 'Conference')
            team_stats_df: DataFrame with team statistics
        
        Returns:
            List of winning teams
        """
        winners = []
        
        if round_name == 'Wild Card':
            # 2 vs 7, 3 vs 6, 4 vs 5
            matchups = [(2, 7), (3, 6), (4, 5)]
            for seed1, seed2 in matchups:
                team1 = next((s.team for s in seeds if s.seed == seed1), None)
                team2 = next((s.team for s in seeds if s.seed == seed2), None)
                
                if team1 and team2:
                    prob = self.predict_matchup_win_prob(team1, team2, team_stats_df)
                    winner = team1 if np.random.random() < prob else team2
                    winners.append(winner)
                    logger.info(f"  {team1} ({seed1}) vs {team2} ({seed2}): {winner} wins (prob: {prob:.3f})")
        
        elif round_name == 'Divisional':
            # 1 vs lowest remaining, 2 vs highest remaining (if applicable)
            # For simplicity, assume 1 vs 4/5 winner, 2 vs 3/6 winner
            if len(seeds) >= 4:
                # This is simplified - actual bracket depends on wild card results
                team1 = next((s.team for s in seeds if s.seed == 1), None)
                team2 = next((s.team for s in seeds if s.seed == 2), None)
                # Would need to match with wild card winners
                # For now, return top seeds
                if team1:
                    winners.append(team1)
                if team2:
                    winners.append(team2)
        
        elif round_name == 'Conference':
            # Championship game: two remaining teams
            if len(seeds) == 2:
                team1, team2 = seeds[0].team, seeds[1].team
                prob = self.predict_matchup_win_prob(team1, team2, team_stats_df)
                winner = team1 if np.random.random() < prob else team2
                winners.append(winner)
                logger.info(f"  {team1} vs {team2}: {winner} wins (prob: {prob:.3f})")
        
        return winners
    
    def simulate_full_playoffs(self, season: int, n_simulations: int = 10000) -> Dict:
        """
        Simulate the full playoff bracket multiple times.
        
        Args:
            season: Season year
            n_simulations: Number of simulations to run
        
        Returns:
            Dictionary with simulation results and probabilities
        """
        logger.info(f"Simulating {n_simulations} playoff brackets for season {season}")
        
        # Get seeding
        seeding = self.predict_seeding(season)
        df = self.load_team_stats(season)
        df = self.calculate_win_pct(df)
        df = self.calculate_weighted_win_pct(df)  # Add recent form weighting
        
        # Add engineered features BEFORE loading model (so they're available for prediction)
        df = self._add_engineered_features(df)
        
        # Load improved win probability model
        win_prob_model, features = self._load_win_prob_model(df)
        if win_prob_model is None:
            logger.warning("Using simple win_pct-based predictions (model unavailable)")
        
        # Track results
        super_bowl_winners = []
        conference_champions = {'AFC': [], 'NFC': []}
        
        for sim in range(n_simulations):
            if (sim + 1) % 100 == 0:
                logger.info(f"  Simulation {sim + 1}/{n_simulations}")
            
            # Simulate each conference
            conf_champions = {}
            for conf in ['AFC', 'NFC']:
                if conf not in seeding or len(seeding[conf]) < 7:
                    continue
                
                conf_seeds = seeding[conf]
                
                # Wild Card Round
                # Matchups: 2 vs 7, 3 vs 6, 4 vs 5
                wc_winners = []
                for seed1, seed2 in [(2, 7), (3, 6), (4, 5)]:
                    team1 = next((s.team for s in conf_seeds if s.seed == seed1), None)
                    team2 = next((s.team for s in conf_seeds if s.seed == seed2), None)
                    if team1 and team2:
                        # Higher seed (lower number) is home team
                        home_team = team1 if seed1 < seed2 else team2
                        prob = self.predict_matchup_win_prob(
                            team1, team2, df, home_team=home_team,
                            win_prob_model=win_prob_model, features=features
                        )
                        winner = team1 if np.random.random() < prob else team2
                        wc_winners.append((winner, seed1 if winner == team1 else seed2))
                
                # Divisional Round
                # 1 seed vs lowest remaining seed, 2 seed vs highest remaining seed
                div_winners = []
                seed1_team = next((s.team for s in conf_seeds if s.seed == 1), None)
                seed2_team = next((s.team for s in conf_seeds if s.seed == 2), None)
                
                if seed1_team and wc_winners:
                    # 1 seed plays lowest seed from wild card (1 seed is home)
                    lowest_wc = min(wc_winners, key=lambda x: x[1])
                    prob = self.predict_matchup_win_prob(
                        seed1_team, lowest_wc[0], df, home_team=seed1_team,
                        win_prob_model=win_prob_model, features=features
                    )
                    div_winner1 = seed1_team if np.random.random() < prob else lowest_wc[0]
                    div_winners.append(div_winner1)
                
                if seed2_team and len(wc_winners) >= 2:
                    # 2 seed plays highest seed from wild card (2 seed is home)
                    highest_wc = max([w for w in wc_winners if w[0] != lowest_wc[0]], 
                                    key=lambda x: x[1], default=lowest_wc)
                    prob = self.predict_matchup_win_prob(
                        seed2_team, highest_wc[0], df, home_team=seed2_team,
                        win_prob_model=win_prob_model, features=features
                    )
                    div_winner2 = seed2_team if np.random.random() < prob else highest_wc[0]
                    div_winners.append(div_winner2)
                
                # Conference Championship (higher seed is home team)
                if len(div_winners) == 2:
                    # Determine home team based on original seed
                    seed1_orig = next((s.seed for s in conf_seeds if s.team == div_winners[0]), 1)
                    seed2_orig = next((s.seed for s in conf_seeds if s.team == div_winners[1]), 2)
                    home_team = div_winners[0] if seed1_orig < seed2_orig else div_winners[1]
                    
                    prob = self.predict_matchup_win_prob(
                        div_winners[0], div_winners[1], df, home_team=home_team,
                        win_prob_model=win_prob_model, features=features
                    )
                    conf_champ = div_winners[0] if np.random.random() < prob else div_winners[1]
                    conf_champions[conf] = conf_champ
                    conference_champions[conf].append(conf_champ)
            
            # Super Bowl (neutral site, no home advantage)
            if 'AFC' in conf_champions and 'NFC' in conf_champions:
                prob = self.predict_matchup_win_prob(
                    conf_champions['AFC'], 
                    conf_champions['NFC'], 
                    df,
                    home_team=None,  # Neutral site
                    win_prob_model=win_prob_model, features=features
                )
                sb_winner = conf_champions['AFC'] if np.random.random() < prob else conf_champions['NFC']
                super_bowl_winners.append(sb_winner)
        
        # Calculate probabilities
        results = {
            'season': season,
            'seeding': {
                conf: [
                    {
                        'team': s.team,
                        'seed': s.seed,
                        'win_pct': s.win_pct,
                        'wins': s.wins,
                        'losses': s.losses
                    }
                    for s in seeds
                ]
                for conf, seeds in seeding.items()
            },
            'super_bowl_probabilities': {
                team: super_bowl_winners.count(team) / len(super_bowl_winners)
                for team in set(super_bowl_winners)
            } if super_bowl_winners else {},
            'conference_championship_probabilities': {
                conf: {
                    team: conference_champions[conf].count(team) / len(conference_champions[conf])
                    for team in set(conference_champions[conf])
                }
                for conf in conference_champions
                if conference_champions[conf]
            },
            'n_simulations': n_simulations
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save simulation results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


def main():
    """Main entry point for playoff predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict NFL playoff seeding and outcomes')
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year to predict (default: 2024)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of simulations to run (default: 1000)')
    parser.add_argument('--conference', type=str, choices=['AFC', 'NFC', None],
                       help='Conference to predict (default: both)')
    parser.add_argument('--output', type=str, default='output/playoff_predictions.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    team_stats_path = "data/processed/team_stats_with_fantasy_clean.csv"
    predictor = PlayoffPredictor(team_stats_path)
    
    # Run simulation
    results = predictor.simulate_full_playoffs(args.season, args.simulations)
    
    # Save results
    predictor.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print(f"PLAYOFF PREDICTIONS - SEASON {args.season}")
    print("="*60)
    
    for conf in ['AFC', 'NFC']:
        if conf in results['seeding']:
            print(f"\n{conf} SEEDING:")
            for seed_info in results['seeding'][conf]:
                print(f"  {seed_info['seed']}. {seed_info['team']} "
                      f"({seed_info['win_pct']:.3f}, {seed_info['wins']}-{seed_info['losses']})")
    
    if results['super_bowl_probabilities']:
        print("\nSUPER BOWL PROBABILITIES:")
        sorted_probs = sorted(results['super_bowl_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        for team, prob in sorted_probs[:10]:  # Top 10
            print(f"  {team}: {prob*100:.1f}%")
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

