"""
Improved Playoff Predictor with Enhanced Features

This is an improved version demonstrating the top improvements:
1. XGBoost win probability model
2. Home field advantage
3. Recent form weighting
4. Better feature engineering
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Import base class
from src.playoff_predictor import (
    PlayoffPredictor, PlayoffSeed, NFL_CONFERENCES, TEAM_NAME_MAPPING
)

logger = logging.getLogger(__name__)

# Stadium types for weather adjustments
DOME_TEAMS = ['ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'ARI', 'LAR', 'LV']
COLD_WEATHER_TEAMS = ['GB', 'CHI', 'MIN', 'BUF', 'NE', 'PIT', 'CLE', 'DEN']


class ImprovedPlayoffPredictor(PlayoffPredictor):
    """
    Enhanced playoff predictor with improved accuracy.
    
    Improvements:
    - XGBoost win probability model
    - Home field advantage
    - Recent form weighting
    - Better feature engineering
    """
    
    def _load_win_prob_model(self, team_stats_df: pd.DataFrame):
        """
        Load or create improved win probability model using XGBoost.
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            from itertools import combinations
            
            # Try XGBoost first
            try:
                from xgboost import XGBClassifier
                USE_XGBOOST = True
            except ImportError:
                logger.warning("XGBoost not available, using Random Forest")
                from sklearn.ensemble import RandomForestClassifier
                USE_XGBOOST = False
            
            # Use same feature selection as base class
            EXCLUDE = set(['team', 'season', 'season_type', 'win_pct', 'win', 'loss', 'tie', 
                          'record', 'win_off', 'loss_off', 'tie_off', 'record_off',
                          'win_def', 'loss_def', 'tie_def', 'record_def',
                          'win_pct_off', 'win_pct_def'])
            
            numeric_cols = team_stats_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in EXCLUDE]
            
            # Add engineered features
            team_stats_df = self._add_engineered_features(team_stats_df)
            engineered_features = [
                'offensive_efficiency', 'defensive_efficiency', 
                'turnover_diff', 'points_per_drive'
            ]
            features.extend([f for f in engineered_features if f in team_stats_df.columns])
            
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
                    StandardScaler(),
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
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            
            clf.fit(pairs_X, pairs_y)
            
            logger.info(f"Trained improved win probability model with {len(features)} features")
            return clf, features
            
        except Exception as e:
            logger.warning(f"Error creating improved win probability model: {e}")
            return None, None
    
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
            # Estimate drives from snaps (rough approximation)
            estimated_drives = df.get('offense_snaps', 60) / 10  # ~10 snaps per drive
            df['points_per_drive'] = df['total_off_points'] / estimated_drives.replace(0, 1)
        
        return df
    
    def calculate_weighted_win_pct(self, df: pd.DataFrame, 
                                   recent_weight: float = 0.6) -> pd.DataFrame:
        """
        Calculate win percentage with more weight on recent performance.
        """
        df = df.copy()
        
        # If we have multiple seasons, weight recent ones more
        if 'season' in df.columns:
            df = df.sort_values(['team', 'season'])
            df['win_pct_recent'] = df.groupby('team')['win_pct'].rolling(
                window=2, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Weighted average
            df['win_pct_weighted'] = (
                recent_weight * df['win_pct_recent'] + 
                (1 - recent_weight) * df['win_pct']
            )
        else:
            df['win_pct_weighted'] = df['win_pct']
        
        return df
    
    def predict_matchup_win_prob(self, team1: str, team2: str, 
                                 team_stats_df: pd.DataFrame,
                                 home_team: str = None,
                                 win_prob_model=None, features=None,
                                 month: int = 1) -> float:
        """
        Predict win probability with home field advantage and weather factors.
        """
        # Get base probability
        base_prob = super().predict_matchup_win_prob(
            team1, team2, team_stats_df, win_prob_model, features
        )
        
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
    
    def simulate_full_playoffs(self, season: int, n_simulations: int = 10000) -> Dict:
        """
        Simulate playoffs with improved features.
        Uses more simulations by default for better accuracy.
        """
        logger.info(f"Running improved simulation with {n_simulations} iterations")
        
        # Get seeding with weighted win_pct
        seeding = self.predict_seeding(season)
        df = self.load_team_stats(season)
        df = self.calculate_win_pct(df)
        df = self.calculate_weighted_win_pct(df)  # Add recent form weighting
        
        # Load improved win probability model
        win_prob_model, features = self._load_win_prob_model(df)
        if win_prob_model is None:
            logger.warning("Using simple win_pct-based predictions (model unavailable)")
        
        # Track results
        super_bowl_winners = []
        conference_champions = {'AFC': [], 'NFC': []}
        
        for sim in range(n_simulations):
            if (sim + 1) % 1000 == 0:
                logger.info(f"  Simulation {sim + 1}/{n_simulations}")
            
            # Simulate each conference
            conf_champions = {}
            for conf in ['AFC', 'NFC']:
                if conf not in seeding or len(seeding[conf]) < 7:
                    continue
                
                conf_seeds = seeding[conf]
                
                # Wild Card Round (higher seed is home team)
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
                div_winners = []
                seed1_team = next((s.team for s in conf_seeds if s.seed == 1), None)
                seed2_team = next((s.team for s in conf_seeds if s.seed == 2), None)
                
                if seed1_team and wc_winners:
                    lowest_wc = min(wc_winners, key=lambda x: x[1])
                    prob = self.predict_matchup_win_prob(
                        seed1_team, lowest_wc[0], df, home_team=seed1_team,
                        win_prob_model=win_prob_model, features=features
                    )
                    div_winner1 = seed1_team if np.random.random() < prob else lowest_wc[0]
                    div_winners.append(div_winner1)
                
                if seed2_team and len(wc_winners) >= 2:
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
        
        # Calculate probabilities (same as base class)
        seeding_dict = {
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
        }
        
        results = {
            'season': season,
            'seeding': seeding_dict,
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
            'n_simulations': n_simulations,
            'model_type': 'improved'  # Mark as improved version
        }
        
        return results


def main():
    """Example usage of improved predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved NFL playoff predictions')
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--simulations', type=int, default=10000)
    parser.add_argument('--output', type=str, default='output/playoff_predictions_improved.json')
    
    args = parser.parse_args()
    
    predictor = ImprovedPlayoffPredictor("data/processed/team_stats_with_fantasy_clean.csv")
    results = predictor.simulate_full_playoffs(args.season, args.simulations)
    predictor.save_results(results, args.output)
    
    print(f"\nImproved predictions saved to {args.output}")
    print(f"Super Bowl probabilities:")
    for team, prob in sorted(results['super_bowl_probabilities'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {team}: {prob*100:.1f}%")


if __name__ == '__main__':
    main()

