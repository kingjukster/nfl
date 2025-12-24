# IST_5520-Football-Predictions

NFL game prediction project using machine learning to predict team wins and player fantasy points.

## 🎯 Project Overview

This project uses NFL statistics to:
- Predict fantasy points for offensive players (QB, RB, WR)
- Predict fantasy points for defensive players (CB, LB, DT)
- Predict team win probabilities using pairwise comparisons
- **Predict playoff seeding and Super Bowl winner** 🏆

## 📁 Project Structure

```
IST_5520-Footbal-Predictions/
├── src/                      # Main source code
│   ├── attacker.py          # Offensive player predictions (QB, RB, WR)
│   ├── MLNFL.py             # Defensive player predictions (CB, LB, DT)
│   ├── heatMap2.py          # Team win probability heatmap
│   ├── config.py            # Configuration settings
│   ├── utils.py             # Utility functions (backward compatible)
│   ├── constants/           # 🆕 Modular constants
│   │   ├── teams.py         # Team colors, logos, name mappings
│   │   └── conferences.py   # Conference/division structure
│   ├── data/                # 🆕 Modular data utilities
│   │   ├── loaders.py       # Safe CSV/JSON loading
│   │   └── processors.py    # Data processing functions
│   ├── visualization/        # 🆕 Modular visualization utilities
│   │   ├── team_visualization.py  # Team colors/logos for plots
│   │   └── styling.py            # Matplotlib styling utilities
│   ├── playoff_predictor.py # Playoff seeding and bracket simulation
│   ├── playoff_bracket_visualizer.py  # Bracket visualizations
│   ├── qb_playoff_stats.py  # QB playoff statistics
│   ├── qb_playoff_visualizer.py  # QB playoff visualizations
│   └── comparison/          # Comparison with live stats
│       ├── compare_live_stats.py
│       ├── fetch_live_nfl_stats.py
│       └── run_comparison.py
│
├── notebooks/                # Jupyter notebooks
│   ├── attacker.ipynb
│   ├── defence_points.ipynb
│   ├── heatmapnotebook.ipynb
│   ├── test_data.ipynb
│   └── summaryStats.ipynb
│
├── scripts/                  # Entry point scripts and utilities
│   ├── run_offensive_models.py      # Train offensive models
│   ├── run_defensive_models.py      # Train defensive models
│   ├── run_heatmap.py               # Generate win probability heatmap
│   ├── run_playoff_predictions.py   # Predict playoff seeding and Super Bowl
│   ├── run_comparison.py            # Compare predictions with live stats
│   ├── run_fetch_historical_data.py # Fetch historical NFL data
│   ├── run_enhanced_predictions.py  # Enhanced predictions with historical data
│   ├── tune_playoff_model.py        # Tune playoff model hyperparameters
│   ├── setup_paths.py               # Setup script paths
│   ├── debug_comparison.py          # Debug comparison script
│   └── test_nfl_columns.py          # Test script
│
├── docs/                     # Documentation
│   ├── COMPARISON_GUIDE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── IMPROVEMENTS_RECOMMENDATIONS.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   └── QUICK_FIXES.md
│
├── data/                     # Data files
│   ├── raw/                 # Raw data from Kaggle
│   ├── processed/           # Processed/cleaned data
│   └── live_nfl_stats_*.csv # Fetched live stats
│
├── models/                   # Saved ML models (auto-created)
├── output/                   # Output files (auto-created)
├── logs/                     # Log files (auto-created)
├── examples/                 # Example scripts
│   └── use_historical_data.py
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the scripts:
```bash
# Train offensive player models (QB, RB, WR)
python scripts/run_offensive_models.py

# Train defensive player models (CB, LB, DT)
python scripts/run_defensive_models.py

# Generate win probability heatmap
python scripts/run_heatmap.py

# Predict playoff seeding and Super Bowl
python scripts/run_playoff_predictions.py --season 2024

# Compare predictions with live NFL stats
python scripts/run_comparison.py

# Fetch historical NFL data (Super Bowl era: 1966-present)
python scripts/run_fetch_historical_data.py --start-year 1966 --data-types all --aggregate

# Enhanced predictions using historical data
python scripts/run_enhanced_predictions.py --season 2024 --use-pbp

# See examples of using historical data
python examples/use_historical_data.py
```

## 📊 Scripts

### run_offensive_models.py
Trains Random Forest models to predict fantasy points for:
- **Quarterbacks (QB)**
- **Wide Receivers (WR)**
- **Running Backs (RB)**

**Features:**
- Downloads data from Kaggle automatically
- Uses chronological train/test split
- Comprehensive evaluation metrics
- Saves models to `models/` directory

**Output:**
- Model performance metrics (MAE, RMSE, R², MedAE)
- Saved models: `quarterback_model.pkl`, `wide_receiver_model.pkl`, `running_back_model.pkl`

### run_defensive_models.py
Trains Ridge regression models to predict defensive player fantasy points.

**Features:**
- Hyperparameter tuning with GridSearchCV
- Chronological train/test split
- Position-specific models (CB, LB, DT)
- Team-level aggregations

**Configuration:**
- Edit `src/MLNFL.py` - Change `POS` variable: "CB", "LB", or "DT"
- Set `USE_GRID_SEARCH = True` for hyperparameter tuning

**Output:**
- Top predicted players
- Team averages by position (saved to `output/`)
- Saved model: `ridge_{position}_model.pkl`

### run_heatmap.py
Creates win probability heatmaps using Naive Bayes classifier.

**Features:**
- Pairwise team comparisons
- Win probability matrix visualization
- Saves heatmap to `output/` directory

**Output:**
- Win probability heatmap (saved as PNG)
- Model summary statistics

### run_playoff_predictions.py
Predicts playoff seeding and simulates the complete playoff bracket from Wild Card to Super Bowl.

**Features:**
- Determines playoff seeds (1-7) for AFC and NFC based on regular season performance
- Simulates Wild Card, Divisional, Conference Championship, and Super Bowl rounds
- Uses Monte Carlo simulation (default: 1000 runs) to calculate probabilities
- Integrates with win probability model from `heatMap2.py`
- Generates bracket visualizations

**Usage:**
```bash
# Basic prediction (1000 simulations)
python scripts/run_playoff_predictions.py --season 2024

# More simulations for better accuracy
python scripts/run_playoff_predictions.py --season 2024 --simulations 10000

# Skip visualizations (faster)
python scripts/run_playoff_predictions.py --season 2024 --no-viz
```

**Output:**
- Playoff seeding predictions (AFC and NFC)
- Super Bowl win probabilities
- Conference championship probabilities
- Bracket diagrams (PNG images)
- Complete simulation results (JSON)

**See `docs/PLAYOFF_PREDICTIONS_GUIDE.md` for detailed documentation.**

### run_comparison.py
Compares model predictions with live NFL statistics.

**Features:**
- Fetches current NFL stats automatically
- Matches players/teams between predictions and actuals
- Calculates comprehensive comparison metrics
- Generates detailed reports

**Output:**
- Comparison metrics (MAE, RMSE, R², etc.)
- Comparison report (text and CSV)

## ⚙️ Configuration

Edit `config.py` to customize:
- Model parameters (alpha, n_estimators, etc.)
- File paths
- Data filtering thresholds
- Logging settings

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- kagglehub >= 0.1.0

## 🔧 Recent Improvements

All recommended improvements have been implemented:

✅ **Bug Fixes**
- Fixed duplicate items in feature lists
- Removed duplicate imports
- Fixed train/test split methodology

✅ **Error Handling**
- File existence checks
- Data validation
- Graceful error messages

✅ **Model Improvements**
- Chronological train/test splits
- GridSearchCV hyperparameter tuning
- Comprehensive evaluation metrics
- Model persistence

✅ **Code Quality**
- Logging throughout
- Better documentation
- Configuration management
- Utility functions

✅ **Modular Architecture** 🆕
- Centralized constants (`src/constants/`) - Team colors, mappings, conferences
- Reusable data utilities (`src/data/`) - Safe loading, processing functions
- Shared visualization utilities (`src/visualization/`) - Styling, team visualization
- DRY principle - No code duplication
- Backward compatible - Existing code continues to work

## 🚀 Improving Predictions

Want to improve your model's predictive accuracy? See:

- **`docs/IMPROVING_PREDICTIONS.md`** - Comprehensive improvement guide for player predictions
- **`docs/IMPROVING_PLAYOFF_PREDICTIONS.md`** - Complete guide for playoff predictions (Top 10 improvements)
- **`docs/ADDITIONAL_IMPROVEMENTS.md`** - **NEW!** 15 additional improvements beyond the basics
- **`docs/QUICK_PLAYOFF_IMPROVEMENTS.md`** - Quick wins for playoff accuracy
- **`docs/PREDICTION_IMPROVEMENTS_SUMMARY.md`** - Quick summary
- **`docs/QUICK_IMPROVEMENTS.md`** - Fast improvements for player models

### Next Steps for Improvement

1. **Historical Validation** - Test on past seasons to measure accuracy (`src/playoff_validator.py`)
2. **Ensemble Models** - Combine multiple models for better accuracy
3. **Full Tiebreaker Rules** - Implement complete NFL tiebreaker hierarchy
4. **Playoff Experience** - Factor in teams' playoff history
5. **Strength of Schedule** - Adjust for schedule difficulty

See `docs/ADDITIONAL_IMPROVEMENTS.md` for complete list and implementation guides.

### Quick Start for Better Predictions

1. **Install advanced models** (5 min):
   ```bash
   pip install xgboost lightgbm
   ```

2. **Try improved training** (10 min):
   ```bash
   python src/train_improved_offensive.py
   ```

3. **Compare results** - See which model performs best!

See `docs/IMPLEMENTATION_SUMMARY.md` for details on completed improvements.

## 📝 Data Sources

### Current Data Sources

- **Kaggle Dataset**: "philiphyde1/nfl-stats-1999-2022"
  - Automatically downloaded via kagglehub
  - Contains yearly player statistics (offense and defense)
  - Covers 1999-2022

- **nfl-data-py**: Current/live NFL data
  - Used for recent seasons and live statistics
  - Provides schedules, game results, and team records

### Historical Data Sources (Super Bowl Era: 1966-Present)

The project now supports fetching comprehensive historical NFL data from the start of the Super Bowl era:

- **Pro-Football-Reference (PFR)**: Season-level data (1966-present)
  - Team offense/defense statistics
  - Season standings and records
  - Game-by-game results
  - Player season totals

- **nflfastR/nflverse**: Play-by-play data (1999-present)
  - Detailed play-by-play records
  - EPA (Expected Points Added)
  - Win probability metrics
  - Advanced analytics

**Fetching Historical Data:**

```bash
# Fetch all historical data from 1966 to present
python scripts/run_fetch_historical_data.py --start-year 1966 --data-types all

# Fetch only team stats from PFR
python scripts/run_fetch_historical_data.py --start-year 1966 --data-types team --source pfr

# Fetch play-by-play data only (1999+)
python scripts/run_fetch_historical_data.py --start-year 1999 --data-types pbp --source nflfastr

# Fetch specific year range and aggregate
python scripts/run_fetch_historical_data.py --start-year 2020 --end-year 2023 --data-types all --aggregate
```

**Data Storage:**
- Individual year files: `data/historical/pfr/` and `data/historical/nflfastr/`
- Aggregated datasets: `data/historical/aggregated/`
  - `team_season_stats_1966_present.csv`
  - `game_results_1966_present.csv`
  - `player_season_stats_1966_present.csv`
  - `pbp_data_1999_present.csv`

**Note:** Historical data fetching respects rate limits and includes resume capability to skip already-fetched years.

## 🏆 Playoff Predictions

The system can predict:
- **Playoff Seeding**: Which teams make playoffs and their seeds (1-7)
- **Matchup Win Probabilities**: Head-to-head win probabilities
- **Complete Bracket Simulation**: Simulates entire playoff bracket
- **Super Bowl Probabilities**: Final probability of each team winning

**Quick Start:**
```bash
# Predict 2024 playoffs (1000 simulations)
python scripts/run_playoff_predictions.py --season 2024

# More simulations for better accuracy
python scripts/run_playoff_predictions.py --season 2024 --simulations 10000
```

**See `docs/PLAYOFF_PREDICTIONS_GUIDE.md` for complete documentation.**

### run_enhanced_predictions.py
Enhanced playoff predictions using historical play-by-play data.

**Features:**
- Enhances team statistics with play-by-play derived metrics
- Uses historical context for better predictions
- Combines existing stats with PBP data

**Usage:**
```bash
# Enhanced predictions with PBP data
python scripts/run_enhanced_predictions.py --season 2024 --use-pbp

# Save enhanced stats for later use
python scripts/run_enhanced_predictions.py --season 2024 --save-enhanced-stats
```

**See `docs/USING_HISTORICAL_DATA.md` for complete guide on using historical data.**

## 📈 Model Performance

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **MedAE** (Median Absolute Error)
- **Baseline Comparison** (improvement percentage)

## 🗂️ Output Files

- **Models**: Saved to `models/` directory as `.pkl` files
- **Predictions**: Team averages saved as CSV files
- **Visualizations**: Heatmaps saved to `output/` directory
- **Logs**: Logging output to console (and optionally to files)

## 🔗 Related Files

- `defence_points.ipynb` ← `MLNFL.py` (notebook version)
- `heatmapnotebook.ipynb` ← `heatMap2.py` (notebook version)
- `attacker.ipynb` ← `attacker.py` (notebook version)
- `test_data.ipynb` - Data processing and preparation

## 📚 Documentation

All documentation is in the `docs/` folder. Key documents include:
- `docs/MODULAR_STRUCTURE.md` - 🆕 **Modular code architecture guide**
- `docs/IMPROVEMENTS_RECOMMENDATIONS.md` - Detailed improvement guide
- `docs/QUICK_FIXES.md` - Quick fix reference
- `docs/IMPROVEMENTS_SUMMARY.md` - Prioritized improvements
- `docs/IMPLEMENTATION_SUMMARY.md` - What was implemented
- `docs/PLAYOFF_PREDICTIONS_GUIDE.md` - Complete playoff prediction guide
- `docs/USING_HISTORICAL_DATA.md` - Guide for using historical data

## 🏗️ Modular Architecture

The codebase uses a modular structure for better maintainability and reusability:

### Constants Module (`src/constants/`)
Centralized NFL constants:
- **Team colors, logos, name mappings** - Single source of truth
- **Conference/division structure** - NFL organizational data
- **Utility functions** - `get_team_color()`, `normalize_team_name()`, etc.

### Data Module (`src/data/`)
Reusable data utilities:
- **Safe loading** - `load_csv_safe()`, `load_json_safe()` with error handling
- **Data processing** - `chronological_split()`, `normalize_team_names_in_df()`
- **Validation** - `validate_dataframe()`, `ensure_columns_exist()`

### Visualization Module (`src/visualization/`)
Shared visualization utilities:
- **Team visualization** - `get_team_color()`, `load_team_logo()`
- **Styling** - `setup_plot_style()`, `apply_professional_styling()`
- **Figure saving** - `save_figure_safe()` with error handling

**Example Usage:**
```python
from src.constants import get_team_color, normalize_team_name
from src.data import load_csv_safe, chronological_split
from src.visualization import setup_plot_style, apply_professional_styling

# Get team color
color = get_team_color('KC')  # Returns '#E31837'

# Load data safely
df = load_csv_safe('data/file.csv')

# Setup plot styling
setup_plot_style('seaborn-v0_8-darkgrid')
apply_professional_styling(fig, ax, title='My Chart')
```

**See `docs/MODULAR_STRUCTURE.md` for complete documentation.**

## 🤝 Contributing

1. Follow the existing code style
2. Add logging for new features
3. Include error handling
4. Update documentation

## 📄 License

[Add your license here]

---

**Note**: Most needed CSV files are already created. The scripts will download additional data from Kaggle as needed.
