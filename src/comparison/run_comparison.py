"""
Simple script to run the full comparison workflow.

Usage:
    python run_comparison.py
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.comparison.compare_live_stats import PredictionComparator
from src.comparison.fetch_live_nfl_stats import fetch_season_stats, calculate_fantasy_points_standard, save_live_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the complete comparison workflow."""
    print("=" * 80)
    print("NFL PREDICTION vs LIVE STATS COMPARISON")
    print("=" * 80)
    print()
    
    # Step 1: Fetch live stats
    print("Step 1: Fetching live NFL statistics...")
    current_season = datetime.now().year
    if datetime.now().month < 9:
        current_season -= 1
    
    live_stats = fetch_season_stats(current_season)
    
    if live_stats is None:
        print("[ERROR] Could not fetch live stats. Please check:")
        print("   1. Internet connection")
        print("   2. nfl-data-py installation: pip install nfl-data-py")
        print("   3. Try running: python fetch_live_nfl_stats.py")
        return
    
    # Calculate fantasy points
    live_stats = calculate_fantasy_points_standard(live_stats)
    
    # Save live stats
    stats_file = save_live_stats(live_stats, current_season)
    print(f"[OK] Live stats saved to {stats_file}")
    print()
    
    # Step 2: Load and compare predictions
    print("Step 2: Loading predictions and comparing...")
    comparator = PredictionComparator(season=current_season)
    
    # Load predictions
    predictions = comparator.load_predictions()
    
    if not predictions:
        print("[ERROR] No predictions found. Please:")
        print("   1. Run attacker.py to generate offensive predictions")
        print("   2. Run MLNFL.py to generate defensive predictions")
        print("   3. Check that output files are in the output/ directory")
        return
    
    print(f"[OK] Loaded {len(predictions)} prediction files")
    print()
    
    # Compare
    print("Step 3: Comparing predictions with live stats...")
    comparison_df = comparator.compare_predictions(predictions, live_stats)
    
    if comparison_df.empty:
        print("[ERROR] No comparison results. Possible issues:")
        print("   1. Player names don't match between predictions and live stats")
        print("   2. Missing required columns in prediction files")
        print("   3. Check logs for detailed error messages")
        return
    
    # Step 4: Generate report
    print("Step 4: Generating comparison report...")
    output_dir = Path("output/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "prediction_comparison_report.txt"
    report = comparator.generate_report(
        comparison_df,
        output_file=str(report_file)
    )
    
    # Save comparison CSV
    comparison_file = output_dir / "prediction_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"[OK] Comparison results saved to {comparison_file}")
    print()
    
    # Print summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(report)
    print()
    print("Detailed results saved to:")
    print(f"   - {comparison_file}")
    print(f"   - {report_file}")
    print()


if __name__ == "__main__":
    main()

