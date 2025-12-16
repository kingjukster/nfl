"""
Entry point script for NFL Playoff Predictions

This script orchestrates the complete playoff prediction workflow:
1. Predicts playoff seeding
2. Simulates playoff bracket
3. Generates visualizations
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.playoff_predictor import PlayoffPredictor
from src.playoff_bracket_visualizer import PlayoffBracketVisualizer
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Predict NFL playoff seeding and Super Bowl winner'
    )
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year to predict (default: 2024)')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of simulations to run (default: 10000)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'playoff_predictions_{args.season}.json'
    
    # Initialize predictor
    team_stats_path = "data/processed/team_stats_with_fantasy_clean.csv"
    logger.info(f"Loading team statistics from {team_stats_path}")
    
    predictor = PlayoffPredictor(team_stats_path)
    
    # Run simulation
    logger.info(f"Running {args.simulations} simulations for season {args.season}...")
    results = predictor.simulate_full_playoffs(args.season, args.simulations)
    
    # Save results
    predictor.save_results(results, str(results_file))
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print(f"  NFL PLAYOFF PREDICTIONS - SEASON {args.season}")
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
        for i, (team, prob) in enumerate(sorted_probs[:10], 1):  # Top 10
            bar_length = int(prob * 50)  # Scale to 50 chars
            bar = '#' * bar_length  # Use # instead of Unicode character
            print(f"  {i:2d}. {team:3s} | {prob*100:5.1f}% | {bar}")
    
    # Conference championship probabilities
    if results.get('conference_championship_probabilities'):
        print("\n" + "="*70)
        print("CONFERENCE CHAMPIONSHIP PROBABILITIES:")
        print("-" * 70)
        for conf in ['AFC', 'NFC']:
            if conf in results['conference_championship_probabilities']:
                print(f"\n{conf}:")
                conf_probs = results['conference_championship_probabilities'][conf]
                sorted_conf = sorted(conf_probs.items(), key=lambda x: x[1], reverse=True)
                for team, prob in sorted_conf[:5]:  # Top 5
                    print(f"  {team:3s} | {prob*100:5.1f}%")
    
    print("\n" + "="*70)
    print(f"Results saved to: {results_file}")
    print("="*70)
    
    # Generate visualizations
    if not args.no_viz:
        logger.info("Generating visualizations...")
        try:
            visualizer = PlayoffBracketVisualizer(results_dict=results)
            
            # Summary chart
            visualizer.create_summary_chart(
                output_path=str(output_dir / f'super_bowl_probabilities_{args.season}.png')
            )
            logger.info("Created Super Bowl probabilities chart")
            
            # Bracket diagrams
            for conf in ['AFC', 'NFC']:
                if conf in results['seeding']:
                    visualizer.create_bracket_diagram(
                        conf,
                        output_path=str(output_dir / f'{conf.lower()}_playoff_bracket_{args.season}.png')
                    )
                    logger.info(f"Created {conf} bracket diagram")
            
            print(f"\nVisualizations saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
            logger.warning("Continuing without visualizations...")


if __name__ == '__main__':
    main()

