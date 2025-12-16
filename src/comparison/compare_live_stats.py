"""
Compare model predictions with live NFL statistics.

This script:
1. Loads saved model predictions
2. Fetches current NFL statistics
3. Compares predictions vs actuals
4. Generates comparison reports
"""
import pandas as pd
import numpy as np
import logging
import requests
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLStatsFetcher:
    """Fetch current NFL statistics from various sources."""
    
    def __init__(self):
        self.current_season = datetime.now().year
        # Adjust season if we're before September (previous season)
        if datetime.now().month < 9:
            self.current_season -= 1
    
    def fetch_espn_stats(self, position: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch stats from ESPN (web scraping approach).
        Note: This is a placeholder - you may need to use a different API.
        """
        try:
            # ESPN API endpoint (may require API key)
            # For now, we'll use a simpler approach with nflfastR or manual data entry
            logger.warning("ESPN API requires authentication. Using alternative method.")
            return None
        except Exception as e:
            logger.error(f"Error fetching ESPN stats: {e}")
            return None
    
    def fetch_nflfastr_stats(self, season: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch stats using nfl-data-py package (Python wrapper for nflfastR data).
        """
        try:
            # Try using nfl-data-py if available
            try:
                from .fetch_live_nfl_stats import fetch_season_stats
                logger.info(f"Fetching NFL stats for season {season or self.current_season}...")
                
                season_to_fetch = season or self.current_season
                stats = fetch_season_stats(season_to_fetch)
                
                if stats is not None and not stats.empty:
                    logger.info(f"Successfully fetched {len(stats)} rows from nfl-data-py")
                    return stats
                else:
                    logger.warning("nfl-data-py returned empty data")
                    return None
            except ImportError as e:
                logger.warning(f"nfl-data-py not available: {e}")
                logger.warning("Install with: pip install nfl-data-py")
                return None
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def fetch_pro_football_reference(self) -> Optional[pd.DataFrame]:
        """
        Scrape Pro Football Reference (requires beautifulsoup4).
        Note: Be respectful of their robots.txt and rate limits.
        """
        try:
            from bs4 import BeautifulSoup
            import time
            
            # This is a placeholder - actual implementation would scrape PFR
            logger.warning("Pro Football Reference scraping not fully implemented")
            return None
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return None
        except Exception as e:
            logger.error(f"Error scraping Pro Football Reference: {e}")
            return None


class PredictionComparator:
    """Compare model predictions with actual NFL statistics."""
    
    def __init__(self, predictions_dir: str = "output", season: int = None):
        self.predictions_dir = Path(predictions_dir)
        self.season = season or datetime.now().year
        if datetime.now().month < 9:
            self.season -= 1
        self.stats_fetcher = NFLStatsFetcher()
    
    def load_predictions(self, position: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load saved predictions from output directory.
        
        Parameters:
        -----------
        position : str, optional
            Specific position to load (QB, WR, RB, CB, LB, DT)
            
        Returns:
        --------
        dict : Dictionary of DataFrames with predictions
        """
        predictions = {}
        
        # Look for prediction files
        prediction_files = list(self.predictions_dir.glob("*.csv"))
        
        for file in prediction_files:
            try:
                df = pd.read_csv(file)
                # Try to identify what type of predictions these are
                if 'Predicted' in df.columns or 'predicted' in df.columns.lower():
                    key = file.stem
                    predictions[key] = df
                    logger.info(f"Loaded predictions from {file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Could not load {file.name}: {e}")
        
        return predictions
    
    def calculate_fantasy_points_standard(self, stats_df: pd.DataFrame, position: str) -> pd.DataFrame:
        """
        Calculate standard fantasy points from raw stats.
        
        Parameters:
        -----------
        stats_df : pd.DataFrame
            DataFrame with player statistics
        position : str
            Player position (QB, RB, WR, etc.)
            
        Returns:
        --------
        pd.DataFrame with fantasy points added
        """
        df = stats_df.copy()
        
        if position.upper() == 'QB':
            # Standard QB scoring: 4 pts per TD pass, 1 pt per 25 pass yards, -2 per INT
            df['fantasy_points'] = (
                (df.get('passing_tds', 0) * 4) +
                (df.get('passing_yards', 0) / 25) +
                (df.get('interceptions', 0) * -2) +
                (df.get('rushing_yards', 0) / 10) +
                (df.get('rushing_tds', 0) * 6)
            )
        elif position.upper() in ['RB', 'WR']:
            # Standard RB/WR scoring: 1 pt per 10 rush/rec yards, 6 pts per TD
            df['fantasy_points'] = (
                (df.get('rushing_yards', 0) / 10) +
                (df.get('receiving_yards', 0) / 10) +
                (df.get('rushing_tds', 0) * 6) +
                (df.get('receiving_tds', 0) * 6)
            )
        else:
            logger.warning(f"Fantasy point calculation not implemented for {position}")
            df['fantasy_points'] = 0
        
        return df
    
    def is_team_level_prediction(self, df: pd.DataFrame) -> bool:
        """Check if prediction DataFrame is team-level or player-level."""
        # Team-level predictions typically have 'team' column and no 'player_name'
        has_team = 'team' in df.columns
        has_player_name = 'player_name' in df.columns
        
        # If it has team but no player_name, it's likely team-level
        if has_team and not has_player_name:
            return True
        # If filename suggests team averages
        return False
    
    def match_players(self, predictions_df: pd.DataFrame, actuals_df: pd.DataFrame, 
                     pred_name_col: str = 'player_name', 
                     actual_name_col: str = 'player_name') -> pd.DataFrame:
        """
        Match players between predictions and actuals.
        
        Parameters:
        -----------
        predictions_df : pd.DataFrame
            Predictions with player names
        actuals_df : pd.DataFrame
            Actual stats with player names
        pred_name_col : str
            Column name for player name in predictions
        actual_name_col : str
            Column name for player name in actuals
            
        Returns:
        --------
        pd.DataFrame : Merged DataFrame with predictions and actuals
        """
        # Check if pred_name_col exists
        if pred_name_col not in predictions_df.columns:
            logger.warning(f"Column '{pred_name_col}' not found in predictions. Available columns: {list(predictions_df.columns)}")
            return pd.DataFrame()
        
        # Normalize player names for matching
        pred_names = predictions_df[pred_name_col].str.upper().str.strip()
        actual_names = actuals_df[actual_name_col].str.upper().str.strip()
        
        # Try exact match first
        merged = predictions_df.merge(
            actuals_df,
            left_on=pred_name_col,
            right_on=actual_name_col,
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        if len(merged) == 0:
            logger.warning("No exact matches found. Trying fuzzy matching...")
            # Could implement fuzzy matching here with fuzzywuzzy
            try:
                from fuzzywuzzy import fuzz, process
                # Fuzzy matching implementation
                logger.info("Fuzzy matching not fully implemented")
            except ImportError:
                logger.warning("fuzzywuzzy not installed for fuzzy matching")
        
        return merged
    
    def match_teams(self, predictions_df: pd.DataFrame, actuals_df: pd.DataFrame,
                   position: str = None) -> pd.DataFrame:
        """
        Match team-level predictions with aggregated actual stats.
        
        Parameters:
        -----------
        predictions_df : pd.DataFrame
            Team-level predictions (has 'team' column)
        actuals_df : pd.DataFrame
            Player-level actual stats
        position : str, optional
            Position to filter for (e.g., 'CB', 'QB')
            
        Returns:
        --------
        pd.DataFrame : Merged DataFrame with team-level predictions and actuals
        """
        if 'team' not in predictions_df.columns:
            logger.error("Team-level predictions must have 'team' column")
            return pd.DataFrame()
        
        # Filter actual stats by position if specified
        actuals_filtered = actuals_df.copy()
        logger.info(f"Starting with {len(actuals_filtered)} total players in actual stats")
        
        if position and 'position' in actuals_df.columns:
            logger.info(f"Filtering to position: {position.upper()}")
            logger.info(f"Available positions: {sorted(actuals_df['position'].unique())}")
            actuals_filtered = actuals_filtered[actuals_filtered['position'] == position.upper()]
            logger.info(f"Filtered actual stats to {position} position: {len(actuals_filtered)} players")
            
            if len(actuals_filtered) == 0:
                logger.error(f"No players found for position {position.upper()} in actual stats!")
                return pd.DataFrame()
        
        # Get team column name in actuals
        team_col = None
        for col in ['team', 'recent_team', 'team_abbr']:
            if col in actuals_filtered.columns:
                team_col = col
                break
        
        if team_col is None:
            logger.error("Could not find team column in actual stats")
            return pd.DataFrame()
        
        # Get fantasy points column
        fp_cols = [c for c in actuals_filtered.columns if 'fantasy' in c.lower() or 'fp' in c.lower()]
        if not fp_cols:
            logger.warning("No fantasy points column found in actual stats, calculating...")
            from .fetch_live_nfl_stats import calculate_fantasy_points_standard
            actuals_filtered = calculate_fantasy_points_standard(actuals_filtered, position)
            fp_cols = [c for c in actuals_filtered.columns if 'fantasy' in c.lower()]
        
        if not fp_cols:
            logger.error("Could not find or calculate fantasy points")
            return pd.DataFrame()
        
        fp_col = fp_cols[0]
        
        # Aggregate actual stats by team
        team_actuals = actuals_filtered.groupby(team_col)[fp_col].mean().reset_index()
        team_actuals.columns = [team_col, 'Actual']
        
        logger.info(f"Aggregated actual stats to {len(team_actuals)} teams")
        logger.info(f"Sample actual teams: {list(team_actuals[team_col].head(5))}")
        
        # Normalize team names for matching
        predictions_df = predictions_df.copy()
        predictions_df['team_normalized'] = predictions_df['team'].str.upper().str.strip()
        team_actuals['team_normalized'] = team_actuals[team_col].str.upper().str.strip()
        
        logger.info(f"Sample predicted teams: {list(predictions_df['team_normalized'].head(5))}")
        logger.info(f"Sample actual teams (normalized): {list(team_actuals['team_normalized'].head(5))}")
        
        # Merge
        merged = predictions_df.merge(
            team_actuals,
            left_on='team_normalized',
            right_on='team_normalized',
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        logger.info(f"Matched {len(merged)} teams out of {len(predictions_df)} predicted teams")
        
        if len(merged) == 0:
            logger.warning("No teams matched! Checking team name differences...")
            pred_teams = set(predictions_df['team_normalized'].unique())
            actual_teams = set(team_actuals['team_normalized'].unique())
            logger.warning(f"Predicted teams: {sorted(pred_teams)}")
            logger.warning(f"Actual teams: {sorted(actual_teams)}")
            logger.warning(f"Common teams: {sorted(pred_teams & actual_teams)}")
        
        return merged
    
    def compare_predictions(self, predictions: Dict[str, pd.DataFrame], 
                          actual_stats: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare predictions with actual statistics.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of prediction DataFrames
        actual_stats : pd.DataFrame, optional
            Actual statistics. If None, will try to fetch.
            
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        if actual_stats is None:
            logger.info("Fetching actual NFL statistics...")
            actual_stats = self.stats_fetcher.fetch_nflfastr_stats(self.season)
            
            if actual_stats is None:
                logger.error("Could not fetch actual statistics. Cannot compare.")
                return pd.DataFrame()
        
        comparison_results = []
        
        for pred_name, pred_df in predictions.items():
            logger.info(f"Comparing {pred_name}...")
            
            # Determine position from filename or data
            position = self._infer_position(pred_name, pred_df)
            
            # Check if this is team-level or player-level prediction
            is_team_level = self.is_team_level_prediction(pred_df)
            
            if is_team_level:
                logger.info(f"Detected team-level predictions for {pred_name}")
                logger.info(f"Prediction columns: {list(pred_df.columns)}")
                logger.info(f"Prediction shape: {pred_df.shape}")
                # Match teams
                matched = self.match_teams(pred_df, actual_stats, position)
                logger.info(f"After team matching: {len(matched)} matched teams")
            else:
                logger.info(f"Detected player-level predictions for {pred_name}")
                logger.info(f"Prediction columns: {list(pred_df.columns)}")
                # Calculate actual fantasy points if needed
                if 'fantasy_points' not in actual_stats.columns:
                    actual_stats = self.calculate_fantasy_points_standard(actual_stats, position)
                
                # Match players
                matched = self.match_players(pred_df, actual_stats)
                logger.info(f"After player matching: {len(matched)} matched players")
            
            if len(matched) > 0:
                logger.info(f"Found {len(matched)} matches for {pred_name}")
                logger.info(f"Matched columns: {list(matched.columns)}")
                # Find prediction column
                pred_cols = [c for c in matched.columns if 'predicted' in c.lower() and c != 'team_normalized']
                if not pred_cols:
                    logger.warning(f"Could not find prediction column in {pred_name}. Available: {list(matched.columns)}")
                    continue
                pred_col = pred_cols[0]
                
                # Find actual column
                actual_cols = [c for c in matched.columns if 'actual' in c.lower() and c != 'team_normalized']
                if not actual_cols:
                    # Try fantasy_points columns
                    actual_cols = [c for c in matched.columns if 'fantasy' in c.lower()]
                if not actual_cols:
                    logger.warning(f"Could not find actual column in {pred_name}. Available: {list(matched.columns)}")
                    continue
                actual_col = actual_cols[0]
                
                logger.info(f"Using columns: Predicted='{pred_col}', Actual='{actual_col}'")
                
                metrics = self._calculate_comparison_metrics(
                    matched[pred_col], 
                    matched[actual_col],
                    pred_name
                )
                comparison_results.append(metrics)
            else:
                logger.warning(f"No matched players for {pred_name}")
        
        if comparison_results:
            return pd.DataFrame(comparison_results)
        else:
            return pd.DataFrame()
    
    def _infer_position(self, filename: str, df: pd.DataFrame) -> str:
        """Infer position from filename or DataFrame."""
        filename_upper = filename.upper()
        if 'QB' in filename_upper or 'QUARTERBACK' in filename_upper:
            return 'QB'
        elif 'RB' in filename_upper or 'RUNNING' in filename_upper:
            return 'RB'
        elif 'WR' in filename_upper or 'WIDE' in filename_upper:
            return 'WR'
        elif 'CB' in filename_upper or 'CORNERBACK' in filename_upper:
            return 'CB'
        elif 'LB' in filename_upper or 'LINEBACKER' in filename_upper:
            return 'LB'
        elif 'DT' in filename_upper:
            return 'DT'
        else:
            return 'UNKNOWN'
    
    def _calculate_comparison_metrics(self, predicted: pd.Series, actual: pd.Series, 
                                     model_name: str) -> Dict:
        """Calculate comparison metrics between predictions and actuals."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Remove any NaN values
        mask = ~(predicted.isna() | actual.isna())
        pred_clean = predicted[mask]
        actual_clean = actual[mask]
        
        if len(pred_clean) == 0:
            return {'model': model_name, 'error': 'No valid data points'}
        
        mae = mean_absolute_error(actual_clean, pred_clean)
        mse = mean_squared_error(actual_clean, pred_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_clean, pred_clean)
        
        # Calculate percentage error
        pct_error = (mae / actual_clean.mean()) * 100 if actual_clean.mean() != 0 else 0
        
        # Count of predictions
        n_predictions = len(pred_clean)
        
        return {
            'model': model_name,
            'n_predictions': n_predictions,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Mean_Actual': actual_clean.mean(),
            'Mean_Predicted': pred_clean.mean(),
            'Pct_Error': pct_error
        }
    
    def generate_report(self, comparison_df: pd.DataFrame, output_file: str = None) -> str:
        """
        Generate a comparison report.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison results
        output_file : str, optional
            Path to save report
            
        Returns:
        --------
        str : Report text
        """
        if comparison_df.empty:
            return "No comparison data available."
        
        report_lines = [
            "=" * 80,
            "NFL PREDICTION vs LIVE STATS COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Season: {self.season}",
            "=" * 80,
            ""
        ]
        
        for _, row in comparison_df.iterrows():
            report_lines.extend([
                f"Model: {row['model']}",
                f"  Number of Predictions: {row['n_predictions']}",
                f"  Mean Absolute Error: {row['MAE']:.2f}",
                f"  Root Mean Squared Error: {row['RMSE']:.2f}",
                f"  R² Score: {row['R²']:.3f}",
                f"  Mean Actual: {row['Mean_Actual']:.2f}",
                f"  Mean Predicted: {row['Mean_Predicted']:.2f}",
                f"  Percentage Error: {row['Pct_Error']:.1f}%",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text


def main():
    """Main function to run comparison."""
    logger.info("Starting prediction vs live stats comparison...")
    
    comparator = PredictionComparator(season=2024)  # Adjust season as needed
    
    # Load predictions
    logger.info("Loading model predictions...")
    predictions = comparator.load_predictions()
    
    if not predictions:
        logger.error("No predictions found. Please run the model training scripts first.")
        return
    
    # Compare with actual stats
    logger.info("Comparing predictions with actual statistics...")
    comparison_df = comparator.compare_predictions(predictions)
    
    if not comparison_df.empty:
        # Generate and print report
        report = comparator.generate_report(
            comparison_df,
            output_file="output/prediction_comparison_report.txt"
        )
        print("\n" + report)
        
        # Save comparison DataFrame
        comparison_df.to_csv("output/prediction_comparison.csv", index=False)
        logger.info("Comparison results saved to output/prediction_comparison.csv")
    else:
        logger.warning("No comparison results generated. Check data availability.")


if __name__ == "__main__":
    main()

