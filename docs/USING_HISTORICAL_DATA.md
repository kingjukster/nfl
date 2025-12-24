# Using Historical Data in Your Project

This guide shows how to integrate the fetched historical NFL data (1999-2025) into your existing prediction models and analysis.

## Quick Start

```python
from src.load_historical_data import load_pbp_data, derive_team_stats_from_pbp, enhance_team_stats_with_pbp

# Load play-by-play data
pbp = load_pbp_data(start_year=2020, end_year=2023)

# Derive team statistics from PBP
team_stats = derive_team_stats_from_pbp(pbp, season=2023)

# Enhance existing team stats
enhanced_stats = enhance_team_stats_with_pbp(existing_stats_df, season=2023)
```

## Integration Examples

### 1. Enhance Playoff Predictions with Historical Context

**File: `src/playoff_predictor.py`**

```python
from src.load_historical_data import load_pbp_data, derive_team_stats_from_pbp

class PlayoffPredictor:
    def load_team_stats(self, season: int) -> pd.DataFrame:
        """Load team statistics with historical enhancement"""
        # Load existing stats
        df = pd.read_csv(self.team_stats_path)
        df = df[df['season'] == season] if 'season' in df.columns else df
        
        # Enhance with PBP-derived stats
        pbp = load_pbp_data(seasons=[season])
        if not pbp.empty:
            pbp_stats = derive_team_stats_from_pbp(pbp, season)
            # Merge to enhance existing stats
            df = df.merge(pbp_stats, on=['team', 'season'], how='left', suffixes=('', '_pbp'))
        
        return df
```

### 2. Train Models with Extended Historical Data

**File: `src/attacker.py`**

```python
from src.load_historical_data import load_pbp_data, load_rosters

def get_data_with_historical():
    """Get data combining Kaggle and historical PBP data"""
    # Get Kaggle data (1999-2022)
    kaggle_data = get_data()  # Existing function
    
    # Get historical PBP data (1999-2025)
    pbp = load_pbp_data(start_year=1999)
    
    # Extract player stats from PBP
    # Group by player and aggregate
    player_stats = pbp.groupby(['player_id', 'season']).agg({
        'yards_gained': 'sum',
        'touchdown': 'sum',
        # ... other aggregations
    })
    
    # Merge with Kaggle data
    combined = pd.concat([kaggle_data, player_stats], ignore_index=True)
    
    return combined
```

### 3. Use Historical Data for Validation

**File: `src/playoff_validator.py`**

```python
from src.load_historical_data import load_pbp_data, load_schedules

def validate_predictions(season: int):
    """Validate predictions against actual historical results"""
    # Load actual game results
    schedules = load_schedules(seasons=[season])
    pbp = load_pbp_data(seasons=[season])
    
    # Calculate actual playoff results
    # Compare with predictions
    # ...
```

### 4. Create Advanced Features from PBP Data

```python
from src.load_historical_data import load_pbp_data
import pandas as pd

def create_advanced_features(season: int):
    """Create advanced features from play-by-play data"""
    pbp = load_pbp_data(seasons=[season])
    
    # Calculate team-level advanced metrics
    features = []
    
    for team in pbp['posteam'].unique():
        if pd.isna(team):
            continue
        
        team_pbp = pbp[pbp['posteam'] == team]
        
        # EPA (Expected Points Added) - if available
        if 'epa' in team_pbp.columns:
            team_epa = team_pbp['epa'].mean()
        
        # Success rate (yards gained >= 40% of yards to go)
        team_pbp['success'] = (
            (team_pbp['yards_gained'] >= team_pbp['ydstogo'] * 0.4) |
            (team_pbp['down'] == 1)  # First downs are always successful
        )
        success_rate = team_pbp['success'].mean()
        
        # Explosive plays (20+ yard gains)
        explosive_rate = (team_pbp['yards_gained'] >= 20).mean()
        
        features.append({
            'team': team,
            'season': season,
            'avg_epa': team_epa if 'epa' in team_pbp.columns else None,
            'success_rate': success_rate,
            'explosive_play_rate': explosive_rate
        })
    
    return pd.DataFrame(features)
```

## Practical Use Cases

### Use Case 1: Better Team Stats for Playoff Predictions

**Problem:** Current team stats may be incomplete or outdated.

**Solution:** Enhance with PBP-derived stats:

```python
# In run_playoff_predictions.py
from src.load_historical_data import enhance_team_stats_with_pbp

# Load existing stats
team_stats = pd.read_csv("data/processed/team_stats_with_fantasy_clean.csv")

# Enhance with PBP data
enhanced = enhance_team_stats_with_pbp(team_stats, season=2024)

# Use enhanced stats for predictions
predictor = PlayoffPredictor(enhanced_stats_path)
```

### Use Case 2: Extended Training Data for Player Models

**Problem:** Kaggle data only goes to 2022.

**Solution:** Combine with historical PBP data:

```python
# Extract player stats from PBP
pbp = load_pbp_data(start_year=1999, end_year=2025)

# Aggregate by player and season
player_stats = pbp.groupby(['player_id', 'player_name', 'season', 'team']).agg({
    'passing_yards': 'sum',
    'rushing_yards': 'sum',
    'receiving_yards': 'sum',
    'touchdown': 'sum',
    # ... more stats
})

# Combine with Kaggle data
# Train models on extended dataset
```

### Use Case 3: Historical Validation

**Problem:** Want to test predictions on past seasons.

**Solution:** Use historical data for backtesting:

```python
from src.load_historical_data import load_pbp_data, load_schedules

def backtest_predictions(start_season=2010, end_season=2023):
    """Test predictions on historical seasons"""
    results = []
    
    for season in range(start_season, end_season + 1):
        # Load actual results
        schedules = load_schedules(seasons=[season])
        
        # Make predictions
        predictor = PlayoffPredictor(...)
        predictions = predictor.simulate_full_playoffs(season)
        
        # Compare with actual
        accuracy = compare_predictions_vs_actual(predictions, schedules)
        results.append({'season': season, 'accuracy': accuracy})
    
    return pd.DataFrame(results)
```

## Data Structure

### Play-by-Play Data Columns

Key columns in the PBP dataset:
- `game_id`: Unique game identifier
- `season`: Season year
- `week`: Week number
- `posteam`: Team with possession
- `defteam`: Defending team
- `down`: Down number
- `ydstogo`: Yards to go
- `yards_gained`: Yards gained on play
- `touchdown`: 1 if touchdown, 0 otherwise
- `epa`: Expected Points Added (if available)
- `wp`: Win Probability (if available)
- `play_type`: Type of play (pass, run, etc.)

### Roster Data

- `player_id`: Unique player identifier
- `player_name`: Player name
- `team`: Team abbreviation
- `position`: Player position
- `season`: Season year

### Schedule Data

- `game_id`: Unique game identifier
- `season`: Season year
- `week`: Week number
- `home_team`: Home team
- `away_team`: Away team
- `home_score`: Home team score
- `away_score`: Away team score

## Performance Tips

1. **Load only what you need**: Use `seasons` parameter to load specific years
2. **Use aggregated files**: The aggregated CSV is faster than loading individual years
3. **Cache results**: Save processed data to avoid re-processing
4. **Filter early**: Filter by season/team before aggregating

## Example: Complete Integration

```python
# run_enhanced_predictions.py
from src.load_historical_data import load_pbp_data, derive_team_stats_from_pbp
from src.playoff_predictor import PlayoffPredictor
import pandas as pd

def main():
    season = 2024
    
    # Load historical PBP data
    print("Loading historical play-by-play data...")
    pbp = load_pbp_data(seasons=[season])
    
    # Derive team stats
    print("Deriving team statistics...")
    pbp_team_stats = derive_team_stats_from_pbp(pbp, season)
    
    # Load existing team stats
    existing_stats = pd.read_csv("data/processed/team_stats_with_fantasy_clean.csv")
    existing_stats = existing_stats[existing_stats['season'] == season]
    
    # Merge
    enhanced_stats = existing_stats.merge(
        pbp_team_stats,
        on=['team', 'season'],
        how='outer',
        suffixes=('', '_pbp')
    )
    
    # Save enhanced stats
    enhanced_stats.to_csv(f"data/processed/enhanced_team_stats_{season}.csv", index=False)
    
    # Use for predictions
    predictor = PlayoffPredictor(f"data/processed/enhanced_team_stats_{season}.csv")
    results = predictor.simulate_full_playoffs(season)
    
    print("Predictions complete!")

if __name__ == "__main__":
    main()
```

## Next Steps

1. **Update existing scripts** to optionally use historical data
2. **Create enhanced features** from PBP data (EPA, success rate, etc.)
3. **Extend model training** with more historical data
4. **Validate predictions** against historical results
5. **Build analytics** on top of the rich PBP dataset

See `src/load_historical_data.py` for all available functions and their documentation.

