import kagglehub
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_data():
    """Download, load, and process NFL offensive player statistics from Kaggle."""
    try:
        # Download latest version
        logger.info("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
        path = path.replace("\\", "/")  # For Windows compatibility
        
        csv_path = Path(f"{path}/yearly_player_stats_offense.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from dataset")
        
        if df.empty:
            raise ValueError("Dataset is empty")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Filter positions: QB, WR, RB only
    logger.info("Filtering positions: QB, WR, RB")
    df_QB = df[df['position'].isin(['QB'])].copy()
    df_WR = df[df['position'].isin(['WR'])].copy()
    df_RB = df[df['position'].isin(['RB'])].copy()
    
    # Log missing values
    logger.info(f"Missing fantasy_points_standard in QB: {df_QB['fantasy_points_standard'].isna().sum()}")
    logger.info(f"Missing fantasy_points_ppr in RB: {df_RB['fantasy_points_ppr'].isna().sum()}")
    logger.info(f"Missing fantasy_points_ppr in WR: {df_WR['fantasy_points_ppr'].isna().sum()}")
    
    # Remove players with > 7 games missed
    max_games_missed = 7
    df_QB = df_QB[(df_QB['games_missed'] <= max_games_missed) & (df_QB['games_missed'].notna())].copy()
    df_RB = df_RB[(df_RB['games_missed'] <= max_games_missed) & (df_RB['games_missed'].notna())].copy()
    df_WR = df_WR[(df_WR['games_missed'] <= max_games_missed) & (df_WR['games_missed'].notna())].copy()
    
    logger.info(f"After filtering: QB={len(df_QB)}, RB={len(df_RB)}, WR={len(df_WR)}")

    # Calculate games played (handle different season lengths)
    # Use max of games_played + games_missed to determine season length
    if 'games_played' not in df.columns:
        # Try to infer season length from data
        season_length = int((df['games_missed'].max() + 10) // 1)  # Rough estimate
        if season_length < 16:
            season_length = 16
        elif season_length > 18:
            season_length = 17
        else:
            season_length = 17  # Default for modern NFL
        
        df["games_played"] = season_length - df["games_missed"]
        logger.info(f"Using season length: {season_length} games")
    else:
        # If games_played exists, use it but validate
        df["games_played"] = df["games_played"].fillna(17 - df["games_missed"])

    # Avoid divide-by-zero
    df["games_played"] = df["games_played"].replace(0, pd.NA)

    # Features to normalize (totals) - REMOVED DUPLICATES
    to_normalize = [
        "offense_snaps", "touches", "targets", "receptions",
        "rushing_yards", "receiving_yards", "yards_after_catch", "total_yards",
        "rush_touchdown", "receiving_touchdown", "total_tds",
        "rush_attempts", "rush_attempts_redzone", "targets_redzone"
    ]

    # Create new per-game versions
    for col in to_normalize:
        df[f"{col}_pg"] = df[col] / df["games_played"]

    # Keep static / rate-based features as-is
    static_features = [
        "position", "age", "years_exp", "height", "weight",
        "team", "depth_team",
        "offense_pct", "team_offense_snaps",
        "yptarget", "yptouch", "ypc", "ypr",
        "rec_td_pct", "rush_td_pct", "td_pct",
        "team_targets_share", "team_rush_attempts_share", "team_receiving_yards_share",
        "games_missed",'fantasy_points_standard','fantasy_points_ppr',
    ]

    # Combine everything into a new averaged feature DataFrame
    avg_features = df[[*static_features, *[f"{c}_pg" for c in to_normalize]]].copy()

    # Split by position
    df_QB = avg_features[avg_features['position'] == 'QB'].copy()
    df_WR = avg_features[avg_features['position'] == 'WR'].copy()
    df_RB = avg_features[avg_features['position'] == 'RB'].copy()

    # Define target and features
    target = 'fantasy_points_standard'
    features = [col for col in avg_features.columns if col not in [
        'position', 'team', 'depth_team', target, 
        'fantasy_points_ppr', 'fantasy_points_half_ppr'
    ]]

    # Train all three models
    logger.info("Training models for QB, WR, RB...")
    QB_model, QB_scaler = train_position_model(df_QB, "Quarterback", target, features)
    WR_model, WR_scaler = train_position_model(df_WR, "Wide Receiver", target, features)
    RB_model, RB_scaler = train_position_model(df_RB, "Running Back", target, features)
    
    return {
        'QB': (QB_model, QB_scaler),
        'WR': (WR_model, WR_scaler),
        'RB': (RB_model, RB_scaler)
    }



def train_position_model(df_position, position_name, target, features, use_chronological_split=True):
    """
    Train a Random Forest model for a specific position.
    
    Parameters:
    -----------
    df_position : pd.DataFrame
        Data for the specific position
    position_name : str
        Name of the position (for logging)
    target : str
        Target column name
    features : list
        List of feature column names
    use_chronological_split : bool
        If True, use chronological split by season; otherwise use random split
        
    Returns:
    --------
    tuple : (model, scaler)
        Trained model and fitted scaler
    """
    # Drop rows missing the target
    initial_len = len(df_position)
    df_position = df_position.dropna(subset=[target])
    if len(df_position) < initial_len:
        logger.warning(f"{position_name}: Dropped {initial_len - len(df_position)} rows with missing target")
    
    if len(df_position) < 10:
        raise ValueError(f"Insufficient data for {position_name}: only {len(df_position)} rows")
    
    X = df_position[features].copy()
    y = df_position[target].copy()
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data - use chronological if season column exists
    if use_chronological_split and 'season' in df_position.columns:
        max_season = df_position['season'].max()
        train_mask = df_position['season'] < max_season
        test_mask = df_position['season'] == max_season
        
        if test_mask.sum() == 0:
            logger.warning(f"{position_name}: No test data for latest season, using random split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train = X.loc[train_mask]
            X_test = X.loc[test_mask]
            y_train = y.loc[train_mask]
            y_test = y.loc[test_mask]
            logger.info(f"{position_name}: Chronological split - Train: {len(X_train)}, Test: {len(X_test)} (season {max_season})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logger.info(f"{position_name}: Random split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model (Random Forest)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    # Comprehensive metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    
    # Baseline comparison
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0
    
    # Print results
    print(f"\n📊 {position_name} Model Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  MedAE: {medae:.2f}")
    print(f"  Improvement over baseline: {improvement:.1f}%")
    
    # Save model and scaler
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / f"{position_name.lower().replace(' ', '_')}_model.pkl"
    scaler_path = output_dir / f"{position_name.lower().replace(' ', '_')}_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved scaler to {scaler_path}")
    
    return model, scaler


if __name__ == "__main__":
    models = get_data()