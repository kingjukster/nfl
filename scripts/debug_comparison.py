"""
Debug script to inspect prediction and actual data for comparison issues.
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_predictions():
    """Inspect prediction files."""
    output_dir = Path("output")
    prediction_files = list(output_dir.glob("*.csv"))
    
    print("=" * 80)
    print("PREDICTION FILES INSPECTION")
    print("=" * 80)
    
    for file in prediction_files:
        print(f"\n📄 File: {file.name}")
        try:
            df = pd.read_csv(file)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First few rows:")
            print(df.head(3).to_string())
            
            # Check if team-level or player-level
            if 'team' in df.columns and 'player_name' not in df.columns:
                print(f"  ✅ Team-level predictions")
                print(f"  Teams: {sorted(df['team'].unique())}")
            elif 'player_name' in df.columns:
                print(f"  ✅ Player-level predictions")
                print(f"  Sample players: {list(df['player_name'].head(5))}")
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")


def inspect_live_stats():
    """Inspect live stats file."""
    data_dir = Path("data")
    stats_files = list(data_dir.glob("live_nfl_stats_*.csv"))
    
    print("\n" + "=" * 80)
    print("LIVE STATS INSPECTION")
    print("=" * 80)
    
    for file in stats_files:
        print(f"\n📄 File: {file.name}")
        try:
            df = pd.read_csv(file)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            if 'position' in df.columns:
                print(f"  Positions available: {sorted(df['position'].unique())}")
                print(f"  Position counts:")
                print(df['position'].value_counts().to_string())
                
                # Check for defensive positions
                defensive_positions = ['CB', 'LB', 'DT', 'DE', 'S', 'SS', 'FS', 'ILB', 'OLB', 'MLB']
                defensive_players = df[df['position'].isin(defensive_positions)]
                print(f"\n  Defensive players: {len(defensive_players)}")
                if len(defensive_players) > 0:
                    print(f"  Defensive positions: {sorted(defensive_players['position'].unique())}")
                else:
                    print(f"  ⚠️  No defensive players found in live stats!")
            
            if 'team' in df.columns:
                print(f"\n  Teams: {sorted(df['team'].unique())}")
            elif 'recent_team' in df.columns:
                print(f"\n  Recent teams: {sorted(df['recent_team'].unique())}")
            
            # Check fantasy points
            fp_cols = [c for c in df.columns if 'fantasy' in c.lower()]
            if fp_cols:
                print(f"\n  Fantasy point columns: {fp_cols}")
                for col in fp_cols:
                    print(f"    {col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
            
            print(f"\n  First few rows:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")


def compare_team_names():
    """Compare team names between predictions and live stats."""
    print("\n" + "=" * 80)
    print("TEAM NAME COMPARISON")
    print("=" * 80)
    
    # Get prediction teams
    output_dir = Path("output")
    prediction_files = list(output_dir.glob("*team*.csv"))
    
    pred_teams = set()
    for file in prediction_files:
        try:
            df = pd.read_csv(file)
            if 'team' in df.columns:
                pred_teams.update(df['team'].str.upper().str.strip().unique())
        except:
            pass
    
    # Get actual teams
    data_dir = Path("data")
    stats_files = list(data_dir.glob("live_nfl_stats_*.csv"))
    
    actual_teams = set()
    for file in stats_files:
        try:
            df = pd.read_csv(file)
            for col in ['team', 'recent_team', 'team_abbr']:
                if col in df.columns:
                    actual_teams.update(df[col].str.upper().str.strip().unique())
                    break
        except:
            pass
    
    print(f"\nPredicted teams ({len(pred_teams)}):")
    print(f"  {sorted(pred_teams)}")
    
    print(f"\nActual teams ({len(actual_teams)}):")
    print(f"  {sorted(actual_teams)}")
    
    common = pred_teams & actual_teams
    print(f"\nCommon teams ({len(common)}):")
    print(f"  {sorted(common)}")
    
    only_pred = pred_teams - actual_teams
    only_actual = actual_teams - pred_teams
    
    if only_pred:
        print(f"\n⚠️  Only in predictions ({len(only_pred)}):")
        print(f"  {sorted(only_pred)}")
    
    if only_actual:
        print(f"\n⚠️  Only in actual stats ({len(only_actual)}):")
        print(f"  {sorted(only_actual)}")


if __name__ == "__main__":
    inspect_predictions()
    inspect_live_stats()
    compare_team_names()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

