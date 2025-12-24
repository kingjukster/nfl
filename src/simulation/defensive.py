import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
POS = "CB"  # change to "LB", "CB", "DT"
DEFENSE_CSV = "data/raw/yearly_player_stats_defense.csv"
TIERS_CSV = "data/processed/cornerback_tiers_2024.csv"
USE_GRID_SEARCH = True

# Load data with error handling
try:
    csv_path = Path(DEFENSE_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {DEFENSE_CSV}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {DEFENSE_CSV}")
    
    if df.empty:
        raise ValueError(f"Dataset is empty: {DEFENSE_CSV}")
    
    df = df[df['player_name'] != 'N/A']
    logger.info(f"After filtering N/A players: {len(df)} rows")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

POS = "CB"  # change to "LB", "CB", "DT"

data = df

dropped = ['team', 'player_id', 'player_name', 'college', 'conference', 'division',
           'career_fantasy_points_ppr', 'career_fantasy_points_standard',
           'season_average_fantasy_points_ppr', 'career_average_fantasy_points_ppr',
           'season_average_fantasy_points_standard', 'career_average_fantasy_points_standard',
           ]

data = data.drop(columns=dropped)
data['games_played_per_season'] = data['games_played_career'] / data['seasons_played']
data = data[data['games_played_per_season'] >= 10]
data = data.drop(columns=['games_played_per_season'])
data = data[data['games_missed'] <= 7]

delta_cols = [
    'delta_def_touchdown',
    'delta_fantasy_points_ppr',
    'delta_fantasy_points_standard',
    'delta_defense_snaps',
    'delta_team_defense_snaps',
    'delta_depth_team',
    'delta_games_missed',
    'delta_defense_pct',
    'delta_career_solo_tackle',
    'delta_career_assist_tackle',
    'delta_career_tackle_with_assist',
    'delta_career_sack',
    'delta_career_qb_hit',
    'delta_career_safety',
    'delta_career_interception',
    'delta_career_def_touchdown',
    'delta_career_defensive_two_point_attempt',
    'delta_career_fumble_forced',
    'delta_career_defensive_two_point_conv',
    'delta_career_defensive_extra_point_attempt',
    'delta_career_defensive_extra_point_conv',
    'delta_career_fantasy_points_ppr',
    'delta_career_fantasy_points_standard',
    'delta_career_defense_snaps',
    'delta_career_team_defense_snaps',
    'delta_season_average_solo_tackle',
    'delta_career_average_solo_tackle',
    'delta_season_average_assist_tackle',
    'delta_career_average_assist_tackle',
    'delta_season_average_tackle_with_assist',
    'delta_career_average_tackle_with_assist',
    'delta_season_average_sack',
    'delta_career_average_sack',
    'delta_season_average_qb_hit',
    'delta_career_average_qb_hit',
    'delta_season_average_safety',
    'delta_career_average_safety',
    'delta_season_average_interception',
    'delta_career_average_interception',
    'delta_season_average_def_touchdown',
    'delta_career_average_def_touchdown',
    'delta_season_average_defensive_two_point_attempt',
    'delta_career_average_defensive_two_point_attempt',
    'delta_season_average_fumble_forced',
    'delta_career_average_fumble_forced',
    'delta_season_average_defensive_two_point_conv',
    'delta_career_average_defensive_two_point_conv',
    'delta_season_average_defensive_extra_point_attempt',
    'delta_career_average_defensive_extra_point_attempt',
    'delta_season_average_defensive_extra_point_conv',
    'delta_career_average_defensive_extra_point_conv',
    'delta_season_average_fantasy_points_ppr',
    'delta_career_average_fantasy_points_ppr',
    'delta_season_average_fantasy_points_standard',
    'delta_career_average_fantasy_points_standard',
    'delta_season_average_defense_snaps',
    'delta_career_average_defense_snaps',
    'delta_season_average_team_defense_snaps',
    'delta_career_average_team_defense_snaps'
]

season_average_features = [
    'season_average_solo_tackle',
    'season_average_assist_tackle',
    'season_average_tackle_with_assist',
    'season_average_sack',
    'season_average_qb_hit',
    'season_average_safety',
    'season_average_interception',
    'season_average_def_touchdown',
    'season_average_defensive_two_point_attempt',
    'season_average_fumble_forced',
    'season_average_defensive_two_point_conv',
    'season_average_defensive_extra_point_attempt',
    'season_average_defensive_extra_point_conv',
    'season_average_defense_snaps',
    'season_average_team_defense_snaps'
]


"""
data['fantasy_points_ppr'] = (
    data['delta_career_solo_tackle'] * 0.5 +
    data['delta_career_assist_tackle'] * 0.15 +
    data['delta_career_sack'] * 1.0 +
    data['delta_career_interception'] * 4.0 +
    data['delta_career_fumble_forced'] * 3.0 +
    data['delta_career_def_touchdown'] * 6.0 +
    data['delta_career_safety'] * 2.0
)"""

# --- Opportunity scaling (unchanged logic) ---
snap_ratio = np.minimum(
    (data['defense_snaps'] / data['defense_snaps'].replace(0, np.nan).mean()).fillna(0),
    1.0
)
pct_ratio = data['defense_pct'].clip(lower=0, upper=1).fillna(0)
opp_raw = 0.6 * snap_ratio + 0.4 * pct_ratio
opp = (0.75 + 0.5 * opp_raw).clip(lower=0.75, upper=1.25)

data['fantasy_points_ppr'] = (
    data['season_average_solo_tackle'] * 0.30 +
    data['season_average_assist_tackle'] * 0.10 +
    data['season_average_sack'] * 1.50 +
    data['season_average_qb_hit'] * 0.50 +
    data['season_average_interception'] * 5.50 +
    data['season_average_fumble_forced'] * 3.00 +
    data['season_average_def_touchdown'] * 6.00 +
    data['season_average_safety'] * 2.00
) * opp

data = data.drop(columns=delta_cols)
data = data.drop(columns=season_average_features)
# Define features (X) and target (y)
pos_data = data[data['position'] == POS]

y = pos_data['fantasy_points_ppr'] # target variable
X = pos_data.drop(columns=['fantasy_points_ppr', 'fantasy_points_standard', 'position'])  # predictors

# chronological train/test split by season
max_season = X['season'].max()
train_mask = X['season'] < max_season
test_mask  = X['season'] == max_season
# Save player + season info before dropping
X_test_with_season = X.loc[test_mask][['season']].copy()

X_train = X.loc[train_mask].drop(columns=['season'], errors='ignore')
X_test  = X.loc[test_mask].drop(columns=['season'], errors='ignore')
y_train = y.loc[train_mask]
y_test  = y.loc[test_mask]

print(f"Training on seasons < {max_season}, testing on season {max_season}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
if USE_GRID_SEARCH:
    logger.info("Performing hyperparameter tuning with GridSearchCV...")
    param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
    ridge_cv = GridSearchCV(
        Ridge(),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    ridge_cv.fit(X_train_scaled, y_train)
    ridge = ridge_cv.best_estimator_
    logger.info(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    logger.info(f"Best CV score: {ridge_cv.best_score_:.3f}")
else:
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)

# Comprehensive evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Baseline comparison
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0

print(f"\n{POS} Model Performance:")
print(f"  MSE:  {mse:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAE:  {mae:.3f}")
print(f"  R²:   {r2:.3f}")
print(f"  MedAE: {medae:.3f}")
print(f"  Improvement over baseline: {improvement:.1f}%")

# Add predictions and actuals back into your test set
X_test_copy = X_test.copy()
X_test_copy['Actual'] = y_test.values
X_test_copy['Predicted'] = y_pred

player_info = df[['player_id', 'player_name', 'season']]
pos_info = player_info.loc[pos_data.index]

test_indices = X.index[test_mask]  # indices of the test fold
pos_info_test = pos_info.loc[test_indices]

X_test_copy = X_test_copy.join(pos_info_test[['player_name', 'season']])


# Choose which season to display
target_season = X['season'].max()

# Filter and sort by predicted fantasy points
top_players = X_test_copy.sort_values(by='Predicted', ascending=False).head(15)
print(f"\n Top 15 Predicted Defensive Players for {max_season}:")
print(top_players[['player_name', 'Predicted', 'Actual']].to_string(index=False))

# Compute average predicted POSITION fantasy points per team for the test season
team_info = df[['player_id', 'team', 'season']]
pos_team_info = team_info.loc[pos_data.index]
pos_team_test = pos_team_info.loc[test_indices]

X_test_copy = X_test_copy.join(pos_team_test['team'])

team_avg = (
    X_test_copy.groupby('team')[['Predicted', 'Actual']]
    .mean()
    .sort_values(by='Predicted', ascending=False)
)

print(f"\n Average Predicted {POS} Fantasy Points per Team for {int(max_season)}:")
print(team_avg.head(15).to_string())

# Save team averages
output_dir = Path("output/models")
output_dir.mkdir(parents=True, exist_ok=True)
team_avg_path = output_dir / f"avg_{POS}_fantasy_by_team_{int(max_season)}.csv"
team_avg.to_csv(team_avg_path)
print(f"\nSaved team averages to {team_avg_path}")

# Save model and scaler
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
model_path = models_dir / f"ridge_{POS.lower()}_model.pkl"
scaler_path = models_dir / f"ridge_{POS.lower()}_scaler.pkl"

joblib.dump(ridge, model_path)
joblib.dump(scaler, scaler_path)
logger.info(f"Saved model to {model_path}")
logger.info(f"Saved scaler to {scaler_path}")

# After building X_test_copy with player_name + Predicted:
# Try to load tiers file if it exists
tiers_path = Path(TIERS_CSV)
if tiers_path.exists():
    try:
        tiers = pd.read_csv(tiers_path)
        eval_df = X_test_copy.merge(tiers[['player_name','Tier','TierScore']], on='player_name', how='inner')
        
        if len(eval_df) > 0:
            from scipy.stats import spearmanr
            rho, p = spearmanr(eval_df['Predicted'], eval_df['TierScore'])
            print(f"\n📈 Spearman rank correlation (Predicted vs TierScore):")
            print(f"  rho = {rho:.3f}")
            print(f"  p-value = {p:.3g}")
        else:
            logger.warning("No matching players found between predictions and tiers file")
    except Exception as e:
        logger.warning(f"Could not load or process tiers file: {e}")
else:
    logger.info(f"Tiers file not found: {TIERS_CSV}, skipping correlation analysis")

