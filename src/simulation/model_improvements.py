"""
Model Improvement and Fine-Tuning Module

This module provides tools to improve playoff prediction accuracy through:
1. Advanced feature engineering
2. Hyperparameter tuning
3. Model validation
4. Feature importance analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def add_advanced_features(df: pd.DataFrame, game_results_df: pd.DataFrame = None, 
                         season: int = None, week: int = None) -> pd.DataFrame:
    """
    Add advanced features to improve model accuracy.
    
    Features added:
    - Recent form (last 4-6 games)
    - Strength of schedule (opponent win percentage)
    - Offensive/defensive efficiency (points per drive, yards per play)
    - Turnover differential
    - Red zone efficiency (if available)
    - Third down conversion rates (if available)
    - Head-to-head records (if game results available)
    - Momentum (winning/losing streaks)
    - Injury impact (if injury data available)
    
    Args:
        df: Team statistics DataFrame
        game_results_df: Game-by-game results (optional, for head-to-head)
        season: Season year (optional, for injury data)
        week: Week number (optional, for injury data)
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # 1. Recent form (last 4-6 games win percentage)
    if 'season' in df.columns and game_results_df is not None:
        df = _add_recent_form(df, game_results_df, window=6)
    
    # 2. Strength of schedule (already calculated, but ensure it's present)
    if 'strength_of_schedule' not in df.columns:
        df = _calculate_sos(df, game_results_df)
    
    # 3. Offensive efficiency metrics
    df = _add_efficiency_metrics(df)
    
    # 4. Turnover differential
    df = _add_turnover_metrics(df)
    
    # 5. Momentum/streak features
    if game_results_df is not None:
        df = _add_momentum_features(df, game_results_df)
    
    # 6. Home/away splits (if available)
    if game_results_df is not None:
        df = _add_home_away_splits(df, game_results_df)
    
    # 7. Injury impact (if injury data available)
    if season is not None:
        df = _add_injury_features(df, season, week)
    
    return df


def _add_recent_form(df: pd.DataFrame, game_results_df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Add recent form (win percentage in last N games)"""
    df = df.copy()
    
    if 'recent_form' in df.columns:
        return df
    
    recent_forms = []
    for _, row in df.iterrows():
        team = row['team']
        season = row.get('season', None)
        
        if season is None:
            recent_forms.append(0.5)
            continue
        
        # Get team's games (check for both 'team' column format and 'home_team'/'away_team' format)
        if 'team' in game_results_df.columns:
            # Format: team, opponent, result
            team_games = game_results_df[
                (game_results_df['team'] == team) &
                (game_results_df.get('season', season) == season)
            ].copy()
        elif 'home_team' in game_results_df.columns:
            # Format: home_team, away_team
            team_games = game_results_df[
                ((game_results_df['home_team'] == team) | (game_results_df['away_team'] == team)) &
                (game_results_df.get('season', season) == season)
            ].copy()
        else:
            recent_forms.append(row.get('win_pct', 0.5))
            continue
        
        if team_games.empty:
            recent_forms.append(row.get('win_pct', 0.5))
            continue
        
        # Sort by week (most recent last)
        if 'week' in team_games.columns:
            team_games = team_games.sort_values('week')
        
        # Get last N games
        recent_games = team_games.tail(window)
        
        # Calculate win percentage
        wins = 0
        for _, game in recent_games.iterrows():
            if 'result' in game_results_df.columns:
                # Format: team, opponent, result
                if game.get('result') == 'W':
                    wins += 1
            elif 'home_team' in game_results_df.columns:
                # Format: home_team, away_team, home_score, away_score
                home = game.get('home_team', '')
                away = game.get('away_team', '')
                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                
                if home == team and home_score > away_score:
                    wins += 1
                elif away == team and away_score > home_score:
                    wins += 1
        
        recent_form = wins / len(recent_games) if len(recent_games) > 0 else row.get('win_pct', 0.5)
        recent_forms.append(recent_form)
    
    df['recent_form'] = recent_forms
    return df


def _calculate_sos(df: pd.DataFrame, game_results_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate strength of schedule (opponent win percentage)"""
    df = df.copy()
    
    if 'strength_of_schedule' in df.columns:
        return df
    
    sos_values = []
    for _, row in df.iterrows():
        team = row['team']
        season = row.get('season', None)
        
        if season is None or game_results_df is None:
            sos_values.append(0.5)  # Neutral SOS
            continue
        
        # Get all opponents this team faced (check format)
        if 'team' in game_results_df.columns:
            # Format: team, opponent, result
            team_games = game_results_df[
                (game_results_df['team'] == team) &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            # Get opponent teams
            opponents = team_games['opponent'].unique().tolist()
        elif 'home_team' in game_results_df.columns:
            # Format: home_team, away_team
            team_games = game_results_df[
                ((game_results_df['home_team'] == team) | (game_results_df['away_team'] == team)) &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            # Get opponent teams
            opponents = []
            for _, game in team_games.iterrows():
                home = game.get('home_team', '')
                away = game.get('away_team', '')
                if home == team:
                    opponents.append(away)
                else:
                    opponents.append(home)
        else:
            sos_values.append(0.5)
            continue
        
        if not opponents:
            sos_values.append(0.5)
            continue
        
        # Calculate average opponent win percentage
        opponent_wps = []
        for opp in opponents:
            opp_row = df[(df['team'] == opp) & (df.get('season', season) == season)]
            if not opp_row.empty:
                opponent_wps.append(opp_row.iloc[0].get('win_pct', 0.5))
        
        sos = np.mean(opponent_wps) if opponent_wps else 0.5
        sos_values.append(sos)
    
    df['strength_of_schedule'] = sos_values
    return df


def _add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add offensive and defensive efficiency metrics"""
    df = df.copy()
    
    # Points per game (if not already present)
    if 'points_per_game' not in df.columns:
        if 'points_for' in df.columns and 'games_played' in df.columns:
            df['points_per_game'] = df['points_for'] / df['games_played'].clip(lower=1)
        elif 'total_off_points' in df.columns and 'games_played' in df.columns:
            df['points_per_game'] = df['total_off_points'] / df['games_played'].clip(lower=1)
    
    # Points allowed per game
    if 'points_allowed_per_game' not in df.columns:
        if 'points_against' in df.columns and 'games_played' in df.columns:
            df['points_allowed_per_game'] = df['points_against'] / df['games_played'].clip(lower=1)
        elif 'total_def_points' in df.columns and 'games_played' in df.columns:
            df['points_allowed_per_game'] = df['total_def_points'] / df['games_played'].clip(lower=1)
    
    # Yards per play (if yards and plays available)
    if 'offensive_yards' in df.columns and 'total_plays' in df.columns:
        df['yards_per_play_off'] = df['offensive_yards'] / df['total_plays'].clip(lower=1)
    
    if 'defensive_yards_allowed' in df.columns and 'def_total_plays' in df.columns:
        df['yards_per_play_def'] = df['defensive_yards_allowed'] / df['def_total_plays'].clip(lower=1)
    
    # Net points (points for - points against)
    if 'points_per_game' in df.columns and 'points_allowed_per_game' in df.columns:
        df['net_points_per_game'] = df['points_per_game'] - df['points_allowed_per_game']
    
    return df


def _add_turnover_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add turnover-related metrics"""
    df = df.copy()
    
    # Turnover differential
    if 'turnovers' in df.columns and 'def_turnovers' in df.columns:
        df['turnover_differential'] = df['def_turnovers'] - df['turnovers']
    elif 'interceptions' in df.columns and 'fumbles' in df.columns:
        # Try to find defensive turnovers
        if 'def_interceptions' in df.columns or 'def_fumbles' in df.columns:
            df['turnovers_forced'] = (
                df.get('def_interceptions', 0) + df.get('def_fumbles', 0)
            )
            df['turnovers_lost'] = (
                df.get('interceptions', 0) + df.get('fumbles', 0)
            )
            df['turnover_differential'] = df['turnovers_forced'] - df['turnovers_lost']
    
    # Turnover rate (per game)
    if 'turnover_differential' in df.columns and 'games_played' in df.columns:
        df['turnover_differential_per_game'] = df['turnover_differential'] / df['games_played'].clip(lower=1)
    
    return df


def _add_momentum_features(df: pd.DataFrame, game_results_df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum features (winning/losing streaks)"""
    df = df.copy()
    
    if 'current_streak' in df.columns:
        return df
    
    streaks = []
    for _, row in df.iterrows():
        team = row['team']
        season = row.get('season', None)
        
        if season is None:
            streaks.append(0)
            continue
        
        # Get team's games (check for both formats)
        if 'team' in game_results_df.columns:
            team_games = game_results_df[
                (game_results_df['team'] == team) &
                (game_results_df.get('season', season) == season)
            ].copy()
        elif 'home_team' in game_results_df.columns:
            team_games = game_results_df[
                ((game_results_df['home_team'] == team) | (game_results_df['away_team'] == team)) &
                (game_results_df.get('season', season) == season)
            ].copy()
        else:
            streaks.append(0)
            continue
        
        if team_games.empty:
            streaks.append(0)
            continue
        
        # Sort by week
        if 'week' in team_games.columns:
            team_games = team_games.sort_values('week')
        
        # Calculate current streak
        streak = 0
        for _, game in team_games.iterrows():
            if 'result' in game_results_df.columns:
                # Format: team, opponent, result
                won = game.get('result') == 'W'
            elif 'home_team' in game_results_df.columns:
                # Format: home_team, away_team, home_score, away_score
                home = game.get('home_team', '')
                away = game.get('away_team', '')
                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                won = (home == team and home_score > away_score) or (away == team and away_score > home_score)
            else:
                won = False
            
            if won:
                streak = max(0, streak) + 1
            else:
                streak = min(0, streak) - 1
        
        streaks.append(streak)
    
    df['current_streak'] = streaks
    return df


def _add_home_away_splits(df: pd.DataFrame, game_results_df: pd.DataFrame) -> pd.DataFrame:
    """Add home/away win percentage splits"""
    df = df.copy()
    
    home_wps = []
    away_wps = []
    
    for _, row in df.iterrows():
        team = row['team']
        season = row.get('season', None)
        
        if season is None:
            home_wps.append(0.5)
            away_wps.append(0.5)
            continue
        
        # Get home and away games (check format)
        if 'home_away' in game_results_df.columns:
            # Format: team, opponent, home_away, result
            home_games = game_results_df[
                (game_results_df['team'] == team) &
                (game_results_df['home_away'] == 'home') &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            away_games = game_results_df[
                (game_results_df['team'] == team) &
                (game_results_df['home_away'] == 'away') &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            # Calculate home win percentage
            if not home_games.empty:
                home_wins = sum(game.get('result') == 'W' for _, game in home_games.iterrows())
                home_wp = home_wins / len(home_games)
            else:
                home_wp = row.get('win_pct', 0.5)
            
            # Calculate away win percentage
            if not away_games.empty:
                away_wins = sum(game.get('result') == 'W' for _, game in away_games.iterrows())
                away_wp = away_wins / len(away_games)
            else:
                away_wp = row.get('win_pct', 0.5)
        elif 'home_team' in game_results_df.columns:
            # Format: home_team, away_team, home_score, away_score
            home_games = game_results_df[
                (game_results_df['home_team'] == team) &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            away_games = game_results_df[
                (game_results_df['away_team'] == team) &
                (game_results_df.get('season', season) == season)
            ].copy()
            
            # Calculate home win percentage
            if not home_games.empty:
                home_wins = sum(
                    (game.get('home_score', 0) > game.get('away_score', 0))
                    for _, game in home_games.iterrows()
                )
                home_wp = home_wins / len(home_games)
            else:
                home_wp = row.get('win_pct', 0.5)
            
            # Calculate away win percentage
            if not away_games.empty:
                away_wins = sum(
                    (game.get('away_score', 0) > game.get('home_score', 0))
                    for _, game in away_games.iterrows()
                )
                away_wp = away_wins / len(away_games)
            else:
                away_wp = row.get('win_pct', 0.5)
        else:
            # Can't determine home/away, use overall win_pct
            home_wp = row.get('win_pct', 0.5)
            away_wp = row.get('win_pct', 0.5)
        
        home_wps.append(home_wp)
        away_wps.append(away_wp)
    
    df['home_win_pct'] = home_wps
    df['away_win_pct'] = away_wps
    df['home_away_diff'] = [h - a for h, a in zip(home_wps, away_wps)]
    
    return df


def _add_injury_features(df: pd.DataFrame, season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Add injury impact features to team statistics.
    
    Args:
        df: Team statistics DataFrame
        season: Season year
        week: Week number (optional)
    
    Returns:
        DataFrame with injury features added
    """
    df = df.copy()
    
    # Check if injury features already exist
    if 'injury_score' in df.columns and 'key_players_out' in df.columns:
        return df
    
    try:
        from src.data.fetching.fetch_injury_reports import calculate_team_injury_impact, fetch_injury_reports
        
        # Fetch injury data
        injury_df = fetch_injury_reports(season, week)
        
        if injury_df is None or injury_df.empty:
            logger.debug("No injury data available, skipping injury features")
            df['injury_score'] = 0.0
            df['key_players_out'] = 0
            df['total_players_out'] = 0
            return df
        
        # Calculate injury impact for each team
        injury_scores = []
        key_players_out = []
        total_players_out = []
        
        for _, row in df.iterrows():
            team = row.get('team', '')
            if not team:
                injury_scores.append(0.0)
                key_players_out.append(0)
                total_players_out.append(0)
                continue
            
            impact = calculate_team_injury_impact(injury_df, team, week)
            injury_scores.append(impact['injury_score'])
            key_players_out.append(impact['key_players_out'])
            total_players_out.append(impact['total_players_out'])
        
        df['injury_score'] = injury_scores
        df['key_players_out'] = key_players_out
        df['total_players_out'] = total_players_out
        
        logger.debug(f"Added injury features for {len(df)} teams")
        
    except ImportError:
        logger.debug("Injury tracking module not available")
        df['injury_score'] = 0.0
        df['key_players_out'] = 0
        df['total_players_out'] = 0
    except Exception as e:
        logger.debug(f"Could not add injury features: {e}")
        df['injury_score'] = 0.0
        df['key_players_out'] = 0
        df['total_players_out'] = 0
    
    return df


def tune_xgboost_hyperparameters(X_train, y_train, cv_folds: int = 5, n_jobs: int = -1):
    """
    Tune XGBoost hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs
    
    Returns:
        Best parameters and best score
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.warning("XGBoost not available for hyperparameter tuning")
        return None, None
    
    # Parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.15],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1))
    ])
    
    # Use TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Grid search
    logger.info("Starting XGBoost hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_log_loss',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_


def evaluate_model_performance(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (optional, for overfitting check)
        y_train: Training labels (optional)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
    
    # Test predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'test_log_loss': log_loss(y_test, y_pred_proba),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None
    }
    
    # Check for overfitting
    if X_train is not None and y_train is not None:
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        metrics['train_log_loss'] = log_loss(y_train, y_train_pred_proba)
        metrics['overfitting'] = metrics['test_log_loss'] - metrics['train_log_loss']
    
    return metrics


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model (XGBoost, Random Forest, etc.)
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    try:
        # Try to get feature importance from pipeline
        if hasattr(model, 'named_steps'):
            # It's a pipeline, get the classifier step
            if 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                else:
                    logger.warning("Classifier in pipeline does not support feature importance")
                    return pd.DataFrame()
            else:
                # Try to find any step with feature_importances_
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'feature_importances_'):
                        importances = step.feature_importances_
                        break
                else:
                    logger.warning("No step in pipeline supports feature importance")
                    return pd.DataFrame()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    except Exception as e:
        logger.warning(f"Error getting feature importance: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_improved_win_prob_model(team_stats_df: pd.DataFrame, 
                                   game_results_df: pd.DataFrame = None,
                                   tune_hyperparameters: bool = False) -> Tuple:
    """
    Create an improved win probability model with advanced features.
    
    Args:
        team_stats_df: Team statistics DataFrame
        game_results_df: Game results DataFrame (optional)
        tune_hyperparameters: Whether to tune hyperparameters
    
    Returns:
        Tuple of (model, features, metrics)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from itertools import combinations
    
    try:
        from xgboost import XGBClassifier
        USE_XGBOOST = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        USE_XGBOOST = False
    
    # Add advanced features
    logger.info("Adding advanced features...")
    # Get season from team_stats_df if available
    season = team_stats_df['season'].iloc[0] if 'season' in team_stats_df.columns else None
    enhanced_df = add_advanced_features(team_stats_df, game_results_df, season=season)
    
    # Feature selection
    EXCLUDE = set(['team', 'season', 'season_type', 'win_pct', 'win', 'loss', 'tie', 
                  'record', 'win_off', 'loss_off', 'tie_off', 'record_off',
                  'win_def', 'loss_def', 'tie_def', 'record_def',
                  'win_pct_off', 'win_pct_def', 'win_pct_recent', 'win_pct_weighted'])
    
    numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in EXCLUDE]
    
    if not features:
        logger.warning("No features available")
        return None, None, None
    
    logger.info(f"Using {len(features)} features: {features[:10]}...")
    
    # Build pairwise training data
    pairs_X, pairs_y = [], []
    for season, G in enhanced_df.groupby('season'):
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
        logger.warning("No pairwise data")
        return None, None, None
    
    pairs_X = np.asarray(pairs_X)
    pairs_y = np.asarray(pairs_y)
    
    # Train model
    if USE_XGBOOST:
        if tune_hyperparameters:
            best_params, best_score = tune_xgboost_hyperparameters(pairs_X, pairs_y)
            if best_params:
                # Extract parameters (remove 'classifier__' prefix)
                params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
                clf = make_pipeline(
                    StandardScaler(),
                    XGBClassifier(**params, random_state=42, eval_metric='logloss', n_jobs=-1)
                )
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                 subsample=0.8, random_state=42, n_jobs=-1, eval_metric='logloss')
                )
        else:
            clf = make_pipeline(
                StandardScaler(),
                XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, 
                            subsample=0.8, colsample_bytree=0.9, random_state=42, 
                            n_jobs=-1, eval_metric='logloss')
            )
    else:
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        )
    
    clf.fit(pairs_X, pairs_y)
    
    # Evaluate
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, pairs_X, pairs_y, cv=5, scoring='neg_log_loss')
    metrics = {
        'cv_log_loss_mean': -cv_scores.mean(),
        'cv_log_loss_std': cv_scores.std()
    }
    
    logger.info(f"Model trained. CV Log Loss: {metrics['cv_log_loss_mean']:.4f} ± {metrics['cv_log_loss_std']:.4f}")
    
    return clf, features, metrics

