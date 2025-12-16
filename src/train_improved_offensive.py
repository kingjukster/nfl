"""
Improved offensive model training with advanced features and models.

Usage:
    python src/train_improved_offensive.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from improved_models import (
    add_rolling_features, add_trend_features, add_opponent_features,
    train_improved_model, remove_correlated_features
)
from attacker import get_data as get_base_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_improved_data():
    """Get data with improved feature engineering."""
    # Get base data (this will download from Kaggle)
    logger.info("Fetching base data...")
    # We'll need to modify this to return the DataFrame instead of training models
    # For now, let's create a version that returns processed data
    
    import kagglehub
    path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
    path = path.replace("\\", "/")
    
    csv_path = Path(f"{path}/yearly_player_stats_offense.csv")
    df = pd.read_csv(csv_path)
    
    # Filter positions
    df = df[df['position'].isin(['QB', 'RB', 'WR'])]
    df = df[(df['games_missed'] <= 7) & (df['games_missed'].notna())]
    
    # Calculate games played
    if 'games_played' not in df.columns:
        df['games_played'] = 17 - df['games_missed']
    df['games_played'] = df['games_played'].replace(0, pd.NA)
    
    # Per-game features
    to_normalize = [
        "offense_snaps", "touches", "targets", "receptions",
        "rushing_yards", "receiving_yards", "yards_after_catch", "total_yards",
        "rush_touchdown", "receiving_touchdown", "total_tds",
        "rush_attempts", "rush_attempts_redzone", "targets_redzone"
    ]
    
    for col in to_normalize:
        if col in df.columns:
            df[f"{col}_pg"] = df[col] / df["games_played"]
    
    # Add improved features
    if 'season' in df.columns and 'player_id' in df.columns:
        df = add_trend_features(df, group_col='player_id')
    
    # Remove correlated features
    feature_cols = [c for c in df.columns if c not in [
        'position', 'team', 'player_id', 'player_name', 
        'fantasy_points_standard', 'fantasy_points_ppr'
    ]]
    df_features = df[feature_cols].select_dtypes(include=[np.number])
    df_features = remove_correlated_features(df_features, threshold=0.95)
    
    return df, df_features.columns.tolist()


def train_improved_offensive_models():
    """Train improved offensive models with advanced techniques."""
    logger.info("Training improved offensive models...")
    
    # Get data with improved features
    df, feature_cols = get_improved_data()
    
    target = 'fantasy_points_standard'
    positions = ['QB', 'RB', 'WR']
    models_dict = {}
    
    for position in positions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {position} model with improvements...")
        logger.info(f"{'='*60}")
        
        pos_df = df[df['position'] == position].copy()
        pos_df = pos_df.dropna(subset=[target])
        
        if len(pos_df) < 10:
            logger.warning(f"Insufficient data for {position}")
            continue
        
        # Prepare features
        X = pos_df[feature_cols].copy()
        y = pos_df[target].copy()
        
        # Handle categoricals
        X = pd.get_dummies(X, drop_first=True)
        
        # Remove correlated features
        X = remove_correlated_features(X, threshold=0.95)
        
        # Chronological split
        if 'season' in pos_df.columns:
            max_season = pos_df['season'].max()
            train_mask = pos_df['season'] < max_season
            test_mask = pos_df['season'] == max_season
            
            if test_mask.sum() > 0:
                X_train = X.loc[train_mask]
                X_test = X.loc[test_mask]
                y_train = y.loc[train_mask]
                y_test = y.loc[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())
        
        # Check model availability
        try:
            from improved_models import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
        except:
            XGBOOST_AVAILABLE = False
            LIGHTGBM_AVAILABLE = False
        
        # Try different models and compare
        model_types = []
        if XGBOOST_AVAILABLE:
            model_types.append('xgb')
        if LIGHTGBM_AVAILABLE:
            model_types.append('lgbm')
        model_types.append('rf')  # Always available
        
        best_model = None
        best_metrics = None
        best_type = None
        
        for model_type in model_types:
            logger.info(f"\nTrying {model_type.upper()} model...")
            try:
                model, scaler, metrics = train_improved_model(
                    X_train, y_train, X_test, y_test,
                    model_type=model_type,
                    tune_hyperparameters=True,
                    position_name=position
                )
                
                if best_metrics is None or metrics['R²'] > best_metrics['R²']:
                    best_model = model
                    best_metrics = metrics
                    best_type = model_type
                    best_scaler = scaler
                
                logger.info(f"{model_type.upper()} - R²: {metrics['R²']:.3f}, MAE: {metrics['MAE']:.2f}")
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
        
        if best_model:
            logger.info(f"\nBest model for {position}: {best_type.upper()}")
            logger.info(f"  R²: {best_metrics['R²']:.3f}")
            logger.info(f"  MAE: {best_metrics['MAE']:.2f}")
            logger.info(f"  RMSE: {best_metrics['RMSE']:.2f}")
            logger.info(f"  Improvement: {best_metrics['Improvement_%']:.1f}%")
            
            # Save model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"improved_{position.lower()}_model.pkl"
            scaler_path = models_dir / f"improved_{position.lower()}_scaler.pkl"
            
            joblib.dump(best_model, model_path)
            joblib.dump(best_scaler, scaler_path)
            logger.info(f"Saved to {model_path}")
            
            models_dict[position] = (best_model, best_scaler, best_metrics)
    
    return models_dict


def check_model_availability():
    """Check which advanced models are available."""
    global XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
    try:
        from xgboost import XGBRegressor
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        logger.warning("XGBoost not available. Install with: pip install xgboost")
    
    try:
        from lightgbm import LGBMRegressor
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
    
    return XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE


if __name__ == "__main__":
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE = check_model_availability()
    
    models = train_improved_offensive_models()
    print(f"\n[OK] Trained {len(models)} improved models!")

