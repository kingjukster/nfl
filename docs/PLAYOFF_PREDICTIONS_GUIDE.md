# NFL Playoff Predictions Guide

## 🎯 Overview

This system predicts:
1. **Playoff Seeding** - Which teams make the playoffs and their seeds (1-7)
2. **Matchup Win Probabilities** - Probability of each team winning in head-to-head matchups
3. **Complete Bracket Simulation** - Simulates the entire playoff bracket from Wild Card to Super Bowl
4. **Super Bowl Probabilities** - Final probability of each team winning the Super Bowl

## 🚀 Quick Start

### Basic Usage

```bash
# Predict playoffs for 2024 season (default: 1000 simulations)
python run_playoff_predictions.py --season 2024

# Run more simulations for better accuracy
python run_playoff_predictions.py --season 2024 --simulations 10000

# Skip visualizations (faster)
python run_playoff_predictions.py --season 2024 --no-viz
```

### Output Files

After running, you'll get:
- `output/playoff_predictions_{season}.json` - Complete results in JSON format
- `output/super_bowl_probabilities_{season}.png` - Bar chart of Super Bowl probabilities
- `output/afc_playoff_bracket_{season}.png` - AFC bracket diagram
- `output/nfc_playoff_bracket_{season}.png` - NFC bracket diagram

## 📊 How It Works

### 1. Playoff Seeding

The system determines playoff seeding based on:
- **Division Winners (Seeds 1-4)**: Top team in each division by win percentage
- **Wild Card Teams (Seeds 5-7)**: Next 3 best teams by win percentage

Tiebreakers:
- Win percentage (primary)
- Points for (secondary)
- Points against (tertiary)

### 2. Matchup Predictions

Uses the same win probability model as `heatMap2.py`:
- Trains a Gaussian Naive Bayes model on pairwise team comparisons
- Uses team statistics to predict head-to-head win probability
- Falls back to simple win_pct comparison if model unavailable

### 3. Bracket Simulation

Simulates the complete playoff bracket:
- **Wild Card Round**: Seeds 2-7 play (2 vs 7, 3 vs 6, 4 vs 5)
- **Divisional Round**: Seed 1 vs lowest remaining, Seed 2 vs highest remaining
- **Conference Championship**: Two remaining teams
- **Super Bowl**: AFC champion vs NFC champion

### 4. Monte Carlo Simulation

Runs multiple simulations (default: 1000) to calculate:
- Super Bowl win probabilities
- Conference championship probabilities
- Expected bracket outcomes

## 📈 Understanding Results

### Seeding Output

```
AFC PLAYOFF SEEDING:
  1. KC  | Win%: 0.750 | Record: 12-4
  2. BUF | Win%: 0.688 | Record: 11-5
  ...
```

### Super Bowl Probabilities

```
SUPER BOWL WIN PROBABILITIES:
  1. KC  | 25.3% | ████████████████████████████
  2. BUF | 18.7% | ████████████████████
  ...
```

Higher probability = more likely to win Super Bowl based on simulations.

## 🔧 Advanced Usage

### Using the Python API

```python
from src.playoff_predictor import PlayoffPredictor

# Initialize predictor
predictor = PlayoffPredictor("data/processed/team_stats_with_fantasy_clean.csv")

# Predict seeding
seeding = predictor.predict_seeding(season=2024)

# Simulate playoffs
results = predictor.simulate_full_playoffs(season=2024, n_simulations=5000)

# Access results
print(results['super_bowl_probabilities'])
print(results['conference_championship_probabilities'])
```

### Customizing Simulations

```python
# Run more simulations for better accuracy
results = predictor.simulate_full_playoffs(season=2024, n_simulations=10000)

# Predict specific conference
seeding = predictor.predict_seeding(season=2024, conference='AFC')
```

### Visualizations

```python
from src.playoff_bracket_visualizer import PlayoffBracketVisualizer

# Load results
visualizer = PlayoffBracketVisualizer(results_path='output/playoff_predictions_2024.json')

# Create visualizations
visualizer.create_summary_chart(output_path='output/sb_probs.png')
visualizer.create_bracket_diagram('AFC', output_path='output/afc_bracket.png')
```

## 🎯 Integration with Existing Models

The playoff predictor integrates with:
- **Team Win Probability Model** (`heatMap2.py`) - Uses the same pairwise comparison model
- **Team Statistics** - Uses regular season team stats from `team_stats_with_fantasy_clean.csv`

## 📋 Requirements

- Team statistics CSV with columns:
  - `team` - Team abbreviation
  - `season` - Season year
  - `season_type` - 'REG' for regular season
  - `win_pct` or `win`/`loss`/`tie` columns
  - Various team statistics (offense, defense, etc.)

## 🔍 Troubleshooting

### No Teams Found for Conference

**Problem**: "No teams found for AFC/NFC in season X"

**Solution**: 
- Check that team names in your data match NFL abbreviations
- Verify the season has data in your CSV
- Check that `season_type == 'REG'` for regular season data

### Missing Win Probability Model

**Problem**: "Using simple win_pct-based predictions"

**Solution**:
- This is a fallback - the system will still work
- Ensure your team stats CSV has sufficient numeric features
- Check that you have data for multiple seasons (needed for training)

### Low Simulation Count

**Problem**: Probabilities seem unstable or unrealistic

**Solution**:
- Increase `--simulations` to 5000 or 10000
- More simulations = more stable probabilities
- Takes longer but gives better results

## 📊 Example Output

```
======================================================================
  NFL PLAYOFF PREDICTIONS - SEASON 2024
======================================================================

AFC PLAYOFF SEEDING:
----------------------------------------------------------------------
   1. KC  | Win%: 0.750 | Record: 12-4
   2. BUF | Win%: 0.688 | Record: 11-5
   3. BAL | Win%: 0.625 | Record: 10-6
   4. HOU | Win%: 0.563 | Record: 9-7
   5. MIA | Win%: 0.625 | Record: 10-6
   6. CLE | Win%: 0.563 | Record: 9-7
   7. PIT | Win%: 0.500 | Record: 8-8

NFC PLAYOFF SEEDING:
----------------------------------------------------------------------
   1. SF  | Win%: 0.750 | Record: 12-4
   2. DAL | Win%: 0.688 | Record: 11-5
   3. DET | Win%: 0.625 | Record: 10-6
   4. TB  | Win%: 0.563 | Record: 9-7
   5. PHI | Win%: 0.625 | Record: 10-6
   6. LAR | Win%: 0.563 | Record: 9-7
   7. GB  | Win%: 0.500 | Record: 8-8

======================================================================
SUPER BOWL WIN PROBABILITIES:
----------------------------------------------------------------------
   1. KC  | 25.3% | ████████████████████████████
   2. SF  | 22.1% | ███████████████████████
   3. BUF | 18.7% | ████████████████████
   4. DAL | 12.4% | █████████████
   5. BAL |  8.2% | ████████
   ...
```

## 🎓 Next Steps

1. **Improve Win Probability Model**: Use XGBoost or ensemble methods
2. **Add Home Field Advantage**: Factor in home/away for playoff games
3. **Injury Adjustments**: Account for key player injuries
4. **Weather Factors**: Consider weather for outdoor games
5. **Historical Playoff Performance**: Weight teams by playoff experience

## 📚 Related Documentation

- `docs/IMPROVING_PREDICTIONS.md` - How to improve model accuracy
- `src/heatMap2.py` - Win probability model implementation
- `README.md` - Project overview

