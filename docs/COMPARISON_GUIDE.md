# Comparing Predictions with Live NFL Stats

This guide explains how to compare your model predictions with current/live NFL statistics.

## 🎯 Overview

The comparison system allows you to:
1. Fetch current NFL statistics
2. Load your model predictions
3. Match players between predictions and actuals
4. Calculate comparison metrics
5. Generate detailed reports

## 📦 Installation

Install additional dependencies for fetching live stats:

```bash
pip install nfl-data-py requests beautifulsoup4 fuzzywuzzy python-Levenshtein
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Fetch Live NFL Statistics

```bash
python fetch_live_nfl_stats.py
```

This will:
- Fetch current season NFL statistics
- Calculate fantasy points
- Save to `data/live_nfl_stats_{season}.csv`

### Step 2: Compare Predictions

```bash
python compare_live_stats.py
```

This will:
- Load your saved predictions from `output/` directory
- Match players with live stats
- Calculate comparison metrics
- Generate a report in `output/prediction_comparison_report.txt`

## 📊 What Gets Compared

### Offensive Players (QB, RB, WR)
- **Predicted fantasy points** vs **Actual fantasy points**
- Metrics calculated:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
  - Percentage Error

### Defensive Players (CB, LB, DT)
- **Predicted fantasy points** vs **Actual fantasy points**
- Same metrics as offensive players

### Team Averages
- **Predicted team averages** vs **Actual team averages**
- Position-specific comparisons

## 📁 File Structure

```
output/
├── prediction_comparison.csv          # Detailed comparison data
├── prediction_comparison_report.txt  # Human-readable report
└── [your prediction files].csv      # Original predictions

data/
└── live_nfl_stats_{season}.csv       # Fetched live statistics
```

## 🔧 Manual Comparison

If you have your own data source:

1. **Prepare your actual stats CSV** with columns:
   - `player_name` (or similar)
   - `fantasy_points_standard` (or `fantasy_points`)
   - `position` (optional)
   - Other relevant stats

2. **Load and compare**:
```python
from compare_live_stats import PredictionComparator
import pandas as pd

comparator = PredictionComparator()
predictions = comparator.load_predictions()

# Load your actual stats
actual_stats = pd.read_csv("your_actual_stats.csv")

# Compare
comparison = comparator.compare_predictions(predictions, actual_stats)
print(comparison)
```

## 📈 Understanding the Metrics

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual
- Lower is better
- Example: MAE of 5 means predictions are off by 5 points on average

### RMSE (Root Mean Squared Error)
- Similar to MAE but penalizes larger errors more
- Lower is better

### R² (R-squared)
- Proportion of variance explained
- Range: -∞ to 1 (1 is perfect, 0 is baseline, negative is worse than baseline)
- Higher is better

### Percentage Error
- MAE as percentage of mean actual value
- Lower is better
- Example: 10% means predictions are off by 10% on average

## 🎨 Customization

### Change Season
```python
comparator = PredictionComparator(season=2023)  # Compare with 2023 stats
```

### Custom Prediction Directory
```python
comparator = PredictionComparator(predictions_dir="my_predictions")
```

### Filter by Position
```python
predictions = comparator.load_predictions(position="QB")
```

## 🔍 Troubleshooting

### "No predictions found"
- Make sure you've run the model training scripts first
- Check that prediction files are in the `output/` directory
- Verify CSV files have 'Predicted' or 'predicted' columns

### "Could not fetch actual statistics"
- Check internet connection
- Install nfl-data-py: `pip install nfl-data-py`
- Try fetching stats manually: `python fetch_live_nfl_stats.py`

### "No matched players"
- Player names might not match exactly
- Install fuzzywuzzy for fuzzy matching: `pip install fuzzywuzzy python-Levenshtein`
- Check that both datasets have player name columns

### "nfl-data-py not installed"
```bash
pip install nfl-data-py
```

## 📝 Example Output

```
================================================================================
NFL PREDICTION vs LIVE STATS COMPARISON REPORT
Generated: 2024-01-15 14:30:00
Season: 2024
================================================================================

Model: avg_CB_fantasy_by_team_2024
  Number of Predictions: 32
  Mean Absolute Error: 2.45
  Root Mean Squared Error: 3.12
  R² Score: 0.856
  Mean Actual: 45.23
  Mean Predicted: 44.89
  Percentage Error: 5.4%
```

## 🔗 Data Sources

### nfl-data-py
- **Source**: nflfastR data
- **Coverage**: 1999-present
- **Update Frequency**: Weekly during season
- **Installation**: `pip install nfl-data-py`

### Alternative Sources
- **ESPN API** (requires API key)
- **Pro Football Reference** (web scraping)
- **NFL.com** (web scraping)
- **Manual CSV upload**

## 💡 Tips

1. **Run after each week** during the season to track model performance
2. **Save comparison results** to track improvement over time
3. **Compare multiple seasons** to validate model stability
4. **Use percentage error** to understand relative accuracy
5. **Check R² score** to see how well predictions explain variance

## 🎯 Next Steps

1. Fetch current season stats
2. Run your models to generate predictions
3. Compare predictions with live stats
4. Analyze results and iterate on models
5. Track performance over time

---

For questions or issues, check the logs or review the comparison script code.

