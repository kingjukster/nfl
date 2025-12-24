"""
Entry point script for generating win probability heatmap.
Uses the improved playoff bracket visualizer system.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.simulation.predictor import PlayoffPredictor
from src.visualization.playoff_bracket_visualizer import PlayoffBracketVisualizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate win probability heatmap using improved playoff predictor system'
    )
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year to predict (default: 2024)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--use-playoff-teams', action='store_true', default=True,
                       help='Use actual playoff teams (default: True)')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create predictor
        logger.info(f"Creating playoff predictor for season {args.season}")
        team_stats_path = "data/processed/team_stats_with_fantasy_clean.csv"
        predictor = PlayoffPredictor(team_stats_path)
        
        # Get playoff seeding (this determines which teams to include)
        logger.info("Predicting playoff seeding...")
        seeding = predictor.predict_seeding(args.season)
        
        # Create results dict for visualizer
        results = {
            'season': args.season,
            'seeding': {
                'AFC': [{'team': s.team, 'seed': s.seed} for s in seeding.get('AFC', [])],
                'NFC': [{'team': s.team, 'seed': s.seed} for s in seeding.get('NFC', [])]
            },
            'super_bowl_probabilities': {}  # Will be calculated by visualizer if needed
        }
        
        # Create visualizer
        visualizer = PlayoffBracketVisualizer(results_dict=results)
        
        # Generate heatmap using the better system
        output_path = viz_dir / f'win_probability_heatmap_{args.season}.png'
        logger.info(f"Generating win probability heatmap...")
        visualizer.create_win_probability_heatmap(
            predictor=predictor,
            season=args.season,
            output_path=str(output_path)
        )
        
        logger.info(f"Heatmap saved to {output_path}")
        print(f"\n[SUCCESS] Win probability heatmap generated successfully!")
        print(f"  Saved to: {output_path}")
        print(f"  Uses: {len(results['seeding']['AFC']) + len(results['seeding']['NFC'])} playoff teams")
        print(f"  Model: XGBoost with advanced features")
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

