# Historical Data Quick Start Guide

## What You Have

After running the fetch script, you now have:
- **1.2+ million play-by-play records** (1999-2025)
- **66,000+ roster records**
- **7,000+ schedule records**
- All stored in `data/historical/`

## Quick Integration Examples

### 1. Load and Use Play-by-Play Data

```python
from src.load_historical_data import load_pbp_data

# Load data for recent seasons
pbp = load_pbp_data(start_year=2020, end_year=2023)

# Filter to specific team
kc_plays = pbp[pbp['posteam'] == 'KC']

# Calculate team stats
kc_yards = kc_plays['yards_gained'].sum()
kc_tds = (kc_plays['touchdown'] == 1).sum()
```

### 2. Enhance Playoff Predictions

```python
# Use the enhanced predictions script
python run_enhanced_predictions.py --season 2024 --use-pbp
```

This automatically:
- Loads your existing team stats
- Enhances them with PBP-derived metrics
- Uses enhanced stats for predictions

### 3. Derive Team Statistics

```python
from src.load_historical_data import derive_team_stats_from_pbp

# Load PBP for a season
pbp = load_pbp_data(seasons=[2023])

# Derive team stats
team_stats = derive_team_stats_from_pbp(pbp, season=2023)

# Use in your models
print(team_stats[['team', 'offensive_yards', 'total_tds', 'turnovers']])
```

### 4. Get Specific Team Stats

```python
from src.load_historical_data import get_team_season_stats_from_pbp

# Get comprehensive stats for a team
stats = get_team_season_stats_from_pbp('KC', 2023)
print(f"KC 2023: {stats['wins']}-{stats['losses']}, {stats['offensive_yards']} yards")
```

### 5. Use in Existing Models

**For Playoff Predictions:**
```python
from src.playoff_predictor import PlayoffPredictor
from src.load_historical_data import enhance_team_stats_with_pbp

# Load and enhance stats
team_stats = pd.read_csv("data/processed/team_stats_with_fantasy_clean.csv")
enhanced = enhance_team_stats_with_pbp(team_stats, season=2024)

# Save and use
enhanced.to_csv("data/processed/enhanced_stats_2024.csv", index=False)
predictor = PlayoffPredictor("data/processed/enhanced_stats_2024.csv")
```

**For Player Models:**
```python
from src.load_historical_data import load_pbp_data

# Load historical PBP
pbp = load_pbp_data(start_year=1999, end_year=2025)

# Extract player stats
player_stats = pbp.groupby(['player_id', 'player_name', 'season']).agg({
    'yards_gained': 'sum',
    'touchdown': 'sum',
    # ... more aggregations
})

# Combine with Kaggle data for extended training
```

## Common Use Cases

### Use Case 1: Better Team Stats
**Problem:** Team stats file is incomplete  
**Solution:** Enhance with PBP data
```python
enhanced = enhance_team_stats_with_pbp(existing_stats, season=2024)
```

### Use Case 2: More Training Data
**Problem:** Kaggle data only goes to 2022  
**Solution:** Extract player stats from PBP
```python
pbp = load_pbp_data(start_year=2023, end_year=2025)
player_stats = extract_player_stats_from_pbp(pbp)
```

### Use Case 3: Historical Validation
**Problem:** Want to test on past seasons  
**Solution:** Use historical schedules and results
```python
schedules = load_schedules(start_year=2010, end_year=2023)
# Compare predictions vs actual results
```

### Use Case 4: Advanced Analytics
**Problem:** Want EPA, success rate, etc.  
**Solution:** Calculate from PBP data
```python
pbp = load_pbp_data(seasons=[2023])
if 'epa' in pbp.columns:
    team_epa = pbp.groupby('posteam')['epa'].mean()
```

## File Locations

- **Play-by-play**: `data/historical/aggregated/pbp_data_1999_present.csv`
- **Rosters**: `data/historical/nflfastr/rosters/rosters_1999_2025.csv`
- **Schedules**: `data/historical/nflfastr/schedules/schedules_1999_2025.csv`
- **Individual years**: `data/historical/nflfastr/pbp/pbp_{year}.csv`

## Next Steps

1. **Run examples**: `python examples/use_historical_data.py`
2. **Try enhanced predictions**: `python run_enhanced_predictions.py --season 2024`
3. **Read full guide**: `docs/USING_HISTORICAL_DATA.md`
4. **Integrate into your models**: See examples above

## Tips

- Start with recent seasons (2020-2024) to test
- Use aggregated files for faster loading
- Filter by season early to reduce memory
- Save processed data to avoid re-processing

