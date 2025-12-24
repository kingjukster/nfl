"""
Example scripts showing how to use historical data in your project.

Run these examples to see how to integrate historical data into your workflows.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.fetching.load_historical_data import (
    load_pbp_data, derive_team_stats_from_pbp,
    get_team_season_stats_from_pbp, load_rosters, load_schedules
)
import pandas as pd


def example_1_load_pbp():
    """Example 1: Load play-by-play data"""
    print("="*60)
    print("Example 1: Loading Play-by-Play Data")
    print("="*60)
    
    # Load recent seasons
    pbp = load_pbp_data(start_year=2020, end_year=2023)
    print(f"\nLoaded {len(pbp):,} play-by-play records")
    print(f"Columns: {list(pbp.columns)[:10]}...")
    print(f"Seasons: {sorted(pbp['season'].unique()) if 'season' in pbp.columns else 'N/A'}")
    
    return pbp


def example_2_derive_team_stats():
    """Example 2: Derive team statistics from PBP"""
    print("\n" + "="*60)
    print("Example 2: Deriving Team Statistics from PBP")
    print("="*60)
    
    # Load PBP for a specific season
    pbp = load_pbp_data(seasons=[2023])
    
    if not pbp.empty:
        # Derive team stats
        team_stats = derive_team_stats_from_pbp(pbp, season=2023)
        print(f"\nDerived stats for {len(team_stats)} teams")
        print("\nSample team stats:")
        print(team_stats[['team', 'offensive_yards', 'passing_yards', 'rushing_yards', 'total_tds']].head(10))
    
    return team_stats if not pbp.empty else pd.DataFrame()


def example_3_get_team_stats():
    """Example 3: Get comprehensive stats for a specific team"""
    print("\n" + "="*60)
    print("Example 3: Get Team Stats for Specific Team")
    print("="*60)
    
    team = "KC"  # Kansas City Chiefs
    season = 2023
    
    stats = get_team_season_stats_from_pbp(team, season)
    
    if stats:
        print(f"\n{team} {season} Statistics:")
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value}")
    else:
        print(f"No stats found for {team} in {season}")
    
    return stats


def example_4_compare_seasons():
    """Example 4: Compare team performance across seasons"""
    print("\n" + "="*60)
    print("Example 4: Compare Team Performance Across Seasons")
    print("="*60)
    
    team = "KC"
    seasons = [2020, 2021, 2022, 2023]
    
    comparison = []
    for season in seasons:
        stats = get_team_season_stats_from_pbp(team, season)
        if stats:
            comparison.append(stats)
    
    if comparison:
        df = pd.DataFrame(comparison)
        print(f"\n{team} Performance Across Seasons:")
        print(df[['season', 'wins', 'losses', 'win_pct', 'offensive_yards', 'total_tds']].to_string(index=False))
    
    return pd.DataFrame(comparison) if comparison else pd.DataFrame()


def example_5_use_in_predictions():
    """Example 5: Use historical data to enhance predictions"""
    print("\n" + "="*60)
    print("Example 5: Using Historical Data for Predictions")
    print("="*60)
    
    # Load historical PBP
    pbp = load_pbp_data(start_year=2019, end_year=2023)
    
    if not pbp.empty:
        # Calculate team averages over multiple seasons
        team_avgs = pbp.groupby('posteam').agg({
            'yards_gained': 'mean',
            'touchdown': 'mean',
        }).reset_index()
        
        print(f"\nTeam averages over {len(pbp['season'].unique())} seasons:")
        print(team_avgs.head(10))
        
        # This could be used as features in predictions
        print("\nThese averages can be used as features in your models!")
    
    return team_avgs if not pbp.empty else pd.DataFrame()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Historical Data Usage Examples")
    print("="*60)
    
    try:
        # Example 1
        pbp = example_1_load_pbp()
        
        # Example 2
        team_stats = example_2_derive_team_stats()
        
        # Example 3
        kc_stats = example_3_get_team_stats()
        
        # Example 4
        comparison = example_4_compare_seasons()
        
        # Example 5
        team_avgs = example_5_use_in_predictions()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Use load_pbp_data() in your model training")
        print("2. Use derive_team_stats_from_pbp() for playoff predictions")
        print("3. Use get_team_season_stats_from_pbp() for team analysis")
        print("4. See docs/USING_HISTORICAL_DATA.md for more examples")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

