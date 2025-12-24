"""
Model Tuning and Improvement Script

This script allows you to:
1. Tune hyperparameters for the win probability model
2. Add advanced features
3. Evaluate model performance
4. Compare different model configurations
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.simulation.predictor import PlayoffPredictor
from src.simulation.model_improvements import (
    add_advanced_features,
    tune_xgboost_hyperparameters,
    create_improved_win_prob_model,
    evaluate_model_performance,
    get_feature_importance
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Tune and improve playoff prediction model'
    )
    parser.add_argument('--season', type=int, default=2024,
                       help='Season to use for evaluation (default: 2024)')
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning (slow but thorough)')
    parser.add_argument('--add-features', action='store_true',
                       help='Add advanced features to model')
    parser.add_argument('--compare', action='store_true',
                       help='Compare baseline vs improved model')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    team_stats_path = "data/processed/team_stats_with_fantasy_clean.csv"
    
    logger.info("="*70)
    logger.info("MODEL TUNING AND IMPROVEMENT")
    logger.info("="*70)
    logger.info(f"Season: {args.season}")
    logger.info(f"Tune hyperparameters: {args.tune}")
    logger.info(f"Add advanced features: {args.add_features}")
    logger.info("="*70)
    
    # Load predictor
    predictor = PlayoffPredictor(team_stats_path)
    
    # Load team stats
    logger.info("Loading team statistics...")
    team_stats_df = predictor.load_team_stats(args.season)
    
    if team_stats_df.empty:
        logger.error(f"No data found for season {args.season}")
        return
    
    logger.info(f"Loaded {len(team_stats_df)} teams")
    
    # Load game results if available
    game_results_df = None
    try:
        game_results_df = predictor.load_game_results(args.season)
        if not game_results_df.empty:
            logger.info(f"Loaded {len(game_results_df)} game results")
    except Exception as e:
        logger.warning(f"Could not load game results: {e}")
    
    # Add advanced features if requested
    if args.add_features:
        logger.info("\n" + "="*70)
        logger.info("ADDING ADVANCED FEATURES")
        logger.info("="*70)
        
        enhanced_df = add_advanced_features(team_stats_df, game_results_df)
        
        # Show new features
        new_features = set(enhanced_df.columns) - set(team_stats_df.columns)
        if new_features:
            logger.info(f"Added {len(new_features)} new features:")
            for feat in sorted(new_features):
                logger.info(f"  - {feat}")
        else:
            logger.info("No new features could be added (may need game results)")
        
        team_stats_df = enhanced_df
    
    # Create improved model
    logger.info("\n" + "="*70)
    logger.info("CREATING IMPROVED WIN PROBABILITY MODEL")
    logger.info("="*70)
    
    model, features, metrics = create_improved_win_prob_model(
        team_stats_df,
        game_results_df,
        tune_hyperparameters=args.tune
    )
    
    if model is None:
        logger.error("Failed to create model")
        return
    
    logger.info(f"Model created successfully!")
    logger.info(f"Features: {len(features)}")
    if metrics:
        logger.info(f"CV Log Loss: {metrics.get('cv_log_loss_mean', 'N/A'):.4f}")
    
    # Get feature importance
    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE")
    logger.info("="*70)
    
    try:
        importance_df = get_feature_importance(model, features)
        if not importance_df.empty:
            logger.info("\nTop 15 Most Important Features:")
            for i, row in importance_df.head(15).iterrows():
                logger.info(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
            
            # Save feature importance to models subdirectory
            models_output_dir = Path(args.output_dir) / 'models'
            models_output_dir.mkdir(parents=True, exist_ok=True)
            importance_file = models_output_dir / f'feature_importance_{args.season}.csv'
            importance_df.to_csv(importance_file, index=False)
            logger.info(f"\nFeature importance saved to {importance_file}")
    except Exception as e:
        logger.warning(f"Could not get feature importance: {e}")
    
    # Compare with baseline if requested
    if args.compare:
        logger.info("\n" + "="*70)
        logger.info("COMPARING BASELINE VS IMPROVED MODEL")
        logger.info("="*70)
        
        # Create baseline model (current implementation)
        baseline_model, baseline_features = predictor._load_win_prob_model(team_stats_df)
        
        if baseline_model:
            logger.info("Baseline model: Current implementation")
            logger.info(f"  Features: {len(baseline_features) if baseline_features else 'N/A'}")
        
        logger.info("Improved model: Enhanced with advanced features")
        logger.info(f"  Features: {len(features)}")
        if metrics:
            logger.info(f"  CV Log Loss: {metrics.get('cv_log_loss_mean', 'N/A'):.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("TUNING COMPLETE")
    logger.info("="*70)
    logger.info("\nTo use the improved model, update PlayoffPredictor to:")
    logger.info("  1. Use add_advanced_features() before training")
    logger.info("  2. Use the tuned hyperparameters")
    logger.info("  3. Include the new features in feature selection")


if __name__ == '__main__':
    main()

