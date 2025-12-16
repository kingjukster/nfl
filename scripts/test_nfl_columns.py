"""Quick test to see what columns nfl-data-py returns."""
import nfl_data_py as nfl
import pandas as pd

# Fetch a small sample
print("Fetching 2024 weekly data...")
data = nfl.import_weekly_data([2024])

print(f"\nShape: {data.shape}")
print(f"\nAll columns ({len(data.columns)}):")
for i, col in enumerate(data.columns, 1):
    print(f"  {i:2d}. {col}")

# Check for team-related columns
team_cols = [c for c in data.columns if 'team' in c.lower()]
print(f"\nTeam-related columns: {team_cols}")

# Show a sample row
print(f"\nSample row:")
print(data.iloc[0][['player_name', 'position'] + team_cols].to_string())

