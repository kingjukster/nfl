import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def get_data():

    # Download latest version
    path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")

    import pandas as pd 
    path = path.replace("\\", "/")  # For Windows compatibility
    df = pd.read_csv(f"{path}\\yearly_player_stats_offense.csv")

    column_names_list = df.columns.tolist()
    #need to have players that are only qb,wr,rb 
    df_QB = df[df['position'].isin(['QB'])]
    df_WR = df[df['position'].isin(['WR'])]
    df_RB = df[df['position'].isin(['RB'])] 
    df_QB['fantasy_points_standard'].isna().sum()
    df_RB['fantasy_points_ppr'].isna().sum()
    df_WR['fantasy_points_ppr'].isna().sum()
    #remove if games_missed > 7

    df_QB = df_QB[df_QB['games_missed'] <= 7]
    df_RB = df_RB[df_RB['games_missed'] <= 7]
    df_WR = df_WR[df_WR['games_missed'] <= 7]
    df_QB = df_QB[df_QB['games_missed'].notna()]
    df_RB = df_RB[df_RB['games_missed'].notna()]
    df_WR = df_WR[df_WR['games_missed'].notna()]
    # Example: start with your existing df
    # df = pd.read_csv("player_stats.csv")

    # If not already present:
    df["games_played"] = 17 - df["games_missed"]

    # Avoid divide-by-zero
    df["games_played"] = df["games_played"].replace(0, pd.NA)

    # Features to normalize (totals)
    to_normalize = [
        "offense_snaps", "touches", "targets", "receptions",
        "rushing_yards", "receiving_yards", "yards_after_catch", "total_yards",
        "rush_touchdown", "receiving_touchdown", "total_tds",
        "rush_attempts", "rush_attempts_redzone", "targets_redzone",
        "touches", "total_yards", "rush_touchdown",
        "receiving_yards", "offense_snaps"
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

        # Optionally inspect
    #now time to use these features to predict fantasy points for qbs, wrs, rbs
    target_columns = ['fantasy_points_standard', 'fantasy_points_ppr',]   

    df_QB_targets = df_QB[target_columns]
    df_RB_targets = df_RB[target_columns]       
    df_WR_targets = df_WR[target_columns]

    # Example split by position
    df_QB = avg_features[avg_features['position'] == 'QB'].copy()
    df_WR = avg_features[avg_features['position'] == 'WR'].copy()
    df_RB = avg_features[avg_features['position'] == 'RB'].copy()

    # Define your target
    target = 'fantasy_points_standard'  # Replace this with your actual target column

    # Define features — exclude target and identifiers
    features = [col for col in avg_features.columns if col not in ['position', 'team', 'depth_team', target,'fantasy_points_ppr', 'fantasy_points_half_ppr']]

    # Function to train model for a given position


    # Train all three
    QB_model, QB_scaler = train_position_model(df_QB, "Quarterback",target,features)
    WR_model, WR_scaler = train_position_model(df_WR, "Wide Receiver",target,features)
    RB_model, RB_scaler = train_position_model(df_RB, "Running Back",target,features)



def train_position_model(df_position, position_name,target,features):
    # Drop rows missing the target
    df_position = df_position.dropna(subset=[target])
    
    X = df_position[features]
    y = df_position[target]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model (Random Forest for example)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 {position_name} model performance:")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}\n")
    
    return model, scaler