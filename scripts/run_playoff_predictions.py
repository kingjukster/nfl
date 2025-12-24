"""
Entry point script for NFL Playoff Predictions

This script orchestrates the complete playoff prediction workflow:
1. Predicts playoff seeding
2. Simulates playoff bracket
3. Generates visualizations
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.predictor import PlayoffPredictor
from src.visualization.playoff_bracket_visualizer import PlayoffBracketVisualizer
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
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of simulations to run (default: 1000 for testing, use 10000 for production)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths - use organized subdirectories
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / 'playoffs'
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = base_output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
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
                output_path=str(viz_dir / f'super_bowl_probabilities_{args.season}.png')
            )
            logger.info("Created Super Bowl probabilities chart")
            
            # Bracket diagrams
            for conf in ['AFC', 'NFC']:
                if conf in results['seeding']:
                    visualizer.create_bracket_diagram(
                        conf,
                        output_path=str(viz_dir / f'{conf.lower()}_playoff_bracket_{args.season}.png')
                    )
                    logger.info(f"Created {conf} bracket diagram")
            
            # Win probability heatmap
            visualizer.create_win_probability_heatmap(
                predictor=predictor,
                season=args.season,
                output_path=str(viz_dir / f'win_probability_heatmap_{args.season}.png')
            )
            logger.info("Created win probability heatmap")
            
            # Combined ESPN-style bracket with head-to-head probabilities
            bracket_output_path = str(viz_dir / f'playoff_bracket_{args.season}.png')
            visualizer.create_combined_bracket_diagram(
                predictor=predictor,
                season=args.season,
                output_path=bracket_output_path
            )
            logger.info(f"Created playoff bracket diagram: {bracket_output_path}")
            
            # QB playoff performance visualizations (bar chart + limited radar)
            qb_radar_path = str(viz_dir / f'qb_playoff_radar_top_{args.season}.png')
            visualizer.create_qb_playoff_radar(
                season=args.season,
                output_path=qb_radar_path,
                max_qbs=3
            )
            logger.info(f"Created QB playoff visualizations (bar chart + radar): {qb_radar_path}")
            
            print(f"\nResults saved to: {output_dir}")
            print(f"Visualizations saved to: {viz_dir}")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
            logger.warning("Continuing without visualizations...")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

