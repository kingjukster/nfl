"""
Enhanced playoff predictions using historical data.

This script demonstrates how to use the fetched historical data
to improve playoff predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.fetching.load_historical_data import (
    load_pbp_data, derive_team_stats_from_pbp, 
    enhance_team_stats_with_pbp, get_team_season_stats_from_pbp
)
from src.simulation.predictor import PlayoffPredictor
from src.visualization.playoff_bracket_visualizer import PlayoffBracketVisualizer
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_enhanced_team_stats(season: int, use_pbp: bool = True) -> pd.DataFrame:
    """
    Create enhanced team statistics combining existing data with PBP-derived stats.
    
    Parameters:
    -----------
    season : int
        Season year
    use_pbp : bool
        Whether to enhance with play-by-play data
        
    Returns:
    --------
    pd.DataFrame : Enhanced team statistics
    """
    # Load existing team stats
    existing_path = Path("data/processed/team_stats_with_fantasy_clean.csv")
    
    if existing_path.exists():
        logger.info(f"Loading existing team stats from {existing_path}")
        team_stats = pd.read_csv(existing_path)
        
        # Filter to season if column exists
        if 'season' in team_stats.columns:
            team_stats = team_stats[team_stats['season'] == season]
    else:
        logger.warning(f"Existing team stats not found at {existing_path}")
        team_stats = pd.DataFrame()
    
    # Enhance with PBP data if requested
    if use_pbp:
        logger.info(f"Enhancing team stats with play-by-play data for {season}...")
        try:
            pbp = load_pbp_data(seasons=[season])
            if not pbp.empty:
                pbp_stats = derive_team_stats_from_pbp(pbp, season)
                
                if not pbp_stats.empty:
                    if not team_stats.empty:
                        # Merge
                        merge_cols = ['team', 'season'] if 'season' in team_stats.columns else ['team']
                        team_stats = team_stats.merge(
                            pbp_stats,
                            on=merge_cols,
                            how='outer',
                            suffixes=('', '_pbp')
                        )
                        logger.info(f"Merged PBP stats with existing stats")
                    else:
                        # Use PBP stats only
                        team_stats = pbp_stats
                        logger.info(f"Using PBP-derived stats only")
                else:
                    logger.warning("No team stats derived from PBP")
            else:
                logger.warning(f"No PBP data available for {season}")
        except Exception as e:
            logger.error(f"Error enhancing with PBP data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    return team_stats


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced playoff predictions using historical data'
    )
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year to predict (default: 2024)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of simulations (default: 1000 for testing, use 10000 for production)')
    parser.add_argument('--use-pbp', action='store_true', default=True,
                       help='Enhance predictions with play-by-play data (default: True)')
    parser.add_argument('--no-pbp', action='store_true',
                       help='Do not use play-by-play data')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--save-enhanced-stats', action='store_true',
                       help='Save enhanced team stats to file')
    
    args = parser.parse_args()
    
    use_pbp = args.use_pbp and not args.no_pbp
    
    logger.info("="*70)
    logger.info("Enhanced Playoff Predictions with Historical Data")
    logger.info("="*70)
    logger.info(f"Season: {args.season}")
    logger.info(f"Using PBP data: {use_pbp}")
    logger.info(f"Simulations: {args.simulations}")
    logger.info("="*70)
    
    # Create enhanced team stats
    enhanced_stats = create_enhanced_team_stats(args.season, use_pbp=use_pbp)
    
    if enhanced_stats.empty:
        logger.error("No team statistics available. Cannot make predictions.")
        return
    
    # Save enhanced stats if requested
    if args.save_enhanced_stats:
        output_path = Path(args.output_dir) / f"enhanced_team_stats_{args.season}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_stats.to_csv(output_path, index=False)
        logger.info(f"Saved enhanced stats to {output_path}")
    
    # Create temporary CSV for predictor
    temp_stats_path = Path("data/processed") / f"temp_team_stats_{args.season}.csv"
    temp_stats_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_stats.to_csv(temp_stats_path, index=False)
    
    try:
        # Initialize predictor with enhanced stats
        logger.info("Initializing playoff predictor...")
        predictor = PlayoffPredictor(str(temp_stats_path))
        
        # Run simulation
        logger.info(f"Running {args.simulations} simulations...")
        results = predictor.simulate_full_playoffs(args.season, args.simulations)
        
        # Save results to playoffs subdirectory
        base_output_dir = Path(args.output_dir)
        output_dir = base_output_dir / 'playoffs'
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"enhanced_playoff_predictions_{args.season}.json"
        predictor.save_results(results, str(results_file))
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*70)
        print(f"  ENHANCED NFL PLAYOFF PREDICTIONS - SEASON {args.season}")
        print("="*70)
        
        for conf in ['AFC', 'NFC']:
            if conf in results['seeding']:
                print(f"\n{conf} PLAYOFF SEEDING:")
                print("-" * 70)
                for seed_info in results['seeding'][conf]:
                    print(f"  {seed_info['seed']:2d}. {seed_info['team']:3s} | "
                          f"Win%: {seed_info['win_pct']:.3f} | "
                          f"Record: {seed_info['wins']:.0f}-{seed_info['losses']:.0f}")
        
        if results['super_bowl_probabilities']:
            print("\n" + "="*70)
            print("SUPER BOWL WIN PROBABILITIES:")
            print("-" * 70)
            sorted_probs = sorted(results['super_bowl_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for i, (team, prob) in enumerate(sorted_probs[:10], 1):
                bar_length = int(prob * 50)
                bar = '#' * bar_length
                print(f"  {i:2d}. {team:3s} | {prob*100:5.1f}% | {bar}")
        
        print("\n" + "="*70)
        print(f"Enhanced predictions complete!")
        print(f"Results: {results_file}")
        print("="*70)
        
    finally:
        # Clean up temp file
        if temp_stats_path.exists():
            temp_stats_path.unlink()
            logger.debug(f"Cleaned up temporary file: {temp_stats_path}")


if __name__ == "__main__":
    main()

