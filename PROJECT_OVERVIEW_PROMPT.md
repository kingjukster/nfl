# NFL Prediction Project - Overview for AI Assistant

## Project Overview

This is a comprehensive NFL (National Football League) prediction project that uses machine learning to predict:
- **Fantasy points for offensive players** (Quarterbacks, Running Backs, Wide Receivers)
- **Fantasy points for defensive players** (Cornerbacks, Linebackers, Defensive Tackles)
- **Team win probabilities** using pairwise comparisons
- **Playoff seeding and Super Bowl winner** through Monte Carlo simulations

## Project Structure

```
IST_5520-Footbal-Predictions/
├── src/                      # Main source code
│   ├── attacker.py          # Offensive player predictions (QB, RB, WR)
│   ├── MLNFL.py             # Defensive player predictions (CB, LB, DT)
│   ├── heatMap2.py          # Team win probability heatmap generator
│   ├── playoff_predictor.py # Main playoff prediction and simulation engine
│   ├── playoff_bracket_visualizer.py # Visualizes playoff brackets
│   ├── playoff_validator.py # Validates predictions against historical data
│   ├── config.py            # Configuration settings (paths, model params)
│   ├── utils.py             # Utility functions
│   ├── comparison/          # Comparison with live NFL stats
│   │   ├── compare_live_stats.py
│   │   ├── fetch_live_nfl_stats.py
│   │   └── run_comparison.py
│   └── [other modules for data fetching, processing, etc.]
│
├── scripts/                  # Entry point scripts (all executable)
│   ├── run_offensive_models.py      # Train QB, RB, WR models
│   ├── run_defensive_models.py      # Train CB, LB, DT models
│   ├── run_heatmap.py               # Generate win probability heatmap
│   ├── run_playoff_predictions.py   # Main playoff prediction script
│   ├── run_enhanced_predictions.py  # Enhanced predictions with historical data
│   ├── run_comparison.py            # Compare predictions with live stats
│   ├── run_fetch_historical_data.py # Fetch historical NFL data (1966-present)
│   ├── tune_playoff_model.py        # Tune playoff model hyperparameters
│   └── setup_paths.py               # Path setup utility
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── attacker.ipynb
│   ├── defence_points.ipynb
│   ├── heatmapnotebook.ipynb
│   └── [other notebooks]
│
├── data/                     # Data files
│   ├── raw/                 # Raw data from Kaggle
│   ├── processed/           # Processed/cleaned data
│   ├── historical/          # Historical NFL data (1966-present)
│   │   ├── pfr/            # Pro-Football-Reference data
│   │   ├── nflfastr/       # nflfastR play-by-play data
│   │   └── aggregated/     # Combined datasets
│   └── live_nfl_stats_*.csv # Fetched live stats
│
├── models/                   # Saved ML models (auto-created)
│   ├── quarterback_model.pkl
│   ├── wide_receiver_model.pkl
│   ├── running_back_model.pkl
│   └── ridge_*.pkl files for defensive positions
│
├── output/                   # All output files (organized by type)
│   ├── playoffs/            # Playoff prediction JSON files
│   ├── validation/          # Validation results
│   ├── analysis/            # Analysis reports
│   ├── comparison/          # Comparison results (CSV, TXT)
│   ├── models/              # Model outputs (feature importance, team averages)
│   └── visualizations/      # All charts/images (heatmaps, brackets, etc.)
│
├── docs/                     # Documentation
├── examples/                 # Example usage scripts
├── logs/                     # Log files
└── requirements.txt          # Python dependencies
```

## Key Technologies & Libraries

- **Machine Learning**: scikit-learn (Random Forest, Ridge Regression, Naive Bayes, XGBoost)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Data Sources**: 
  - Kaggle dataset (1999-2022)
  - nfl-data-py / nflverse (live and historical data)
  - Pro-Football-Reference (scraped historical data 1966-present)

## Main Workflows

### 1. Player Fantasy Point Predictions

**Offensive Players (QB, RB, WR):**
- Script: `scripts/run_offensive_models.py`
- Source: `src/attacker.py`
- Model: Random Forest
- Output: Saved models in `models/` directory

**Defensive Players (CB, LB, DT):**
- Script: `scripts/run_defensive_models.py`
- Source: `src/MLNFL.py`
- Model: Ridge Regression with GridSearchCV
- Output: Models in `models/`, team averages in `output/models/`

### 2. Team Win Probability Heatmap

- Script: `scripts/run_heatmap.py`
- Source: `src/heatMap2.py`
- Model: Naive Bayes classifier
- Output: Heatmap saved to `output/visualizations/win_probability_heatmap_*.png`

### 3. Playoff Predictions (Main Feature)

- Script: `scripts/run_playoff_predictions.py`
- Source: `src/playoff_predictor.py`, `src/playoff_bracket_visualizer.py`
- Process:
  1. Determines playoff seeding (1-7) for AFC and NFC based on regular season performance
  2. Simulates complete playoff bracket (Wild Card → Divisional → Conference → Super Bowl)
  3. Uses Monte Carlo simulation (default: 1000 runs, recommend 10000 for production)
  4. Calculates win probabilities for each team
- Output: 
  - JSON results in `output/playoffs/playoff_predictions_*.json`
  - Visualizations in `output/visualizations/`:
    - Super Bowl probabilities chart
    - AFC/NFC bracket diagrams
    - Win probability heatmap

### 4. Historical Data Integration

- Script: `scripts/run_fetch_historical_data.py`
- Sources: Pro-Football-Reference (1966-present), nflfastR (1999-present)
- Purpose: Enhances predictions with comprehensive historical context
- Output: Stored in `data/historical/`

### 5. Validation & Analysis

- Scripts: Validation and analysis modules in `src/`
- Purpose: Validates predictions against historical seasons
- Output: Results in `output/validation/` and `output/analysis/`

### 6. Comparison with Live Stats

- Script: `scripts/run_comparison.py`
- Source: `src/comparison/`
- Purpose: Compares model predictions with actual live NFL statistics
- Output: CSV and report in `output/comparison/`

## Configuration

All configuration is centralized in `src/config.py`:
- Data paths
- Model parameters
- Output directories (organized structure)
- Historical data settings

The config automatically creates all necessary directories, including organized output subdirectories.

## Output Organization

**Important**: All outputs are organized into labeled subdirectories:
- `output/playoffs/` - Playoff prediction JSON files
- `output/validation/` - Validation results JSON
- `output/analysis/` - Analysis reports JSON
- `output/comparison/` - Comparison CSV and text reports
- `output/models/` - Model outputs (feature importance, team averages)
- `output/visualizations/` - All images/charts (PNG files)

This organization makes it easy to find specific output types and prevents clutter.

## Key Design Patterns

1. **Path Management**: All scripts use `Path(__file__).parent.parent` to correctly reference project root from `scripts/` folder
2. **Modular Architecture**: Core logic in `src/`, entry points in `scripts/`
3. **Configuration-Driven**: Centralized config in `src/config.py`
4. **Comprehensive Logging**: Logging throughout for debugging and tracking
5. **Error Handling**: Try-catch blocks and graceful error messages
6. **Data Validation**: Checks for file existence and data quality

## Common Tasks

**To train offensive models:**
```bash
python scripts/run_offensive_models.py
```

**To train defensive models:**
```bash
python scripts/run_defensive_models.py
```

**To generate win probability heatmap:**
```bash
python scripts/run_heatmap.py
```

**To predict playoffs (main use case):**
```bash
python scripts/run_playoff_predictions.py --season 2024 --simulations 10000
```

**To compare predictions with live stats:**
```bash
python scripts/run_comparison.py
```

**To fetch historical data:**
```bash
python scripts/run_fetch_historical_data.py --start-year 1966 --data-types all --aggregate
```

## Important Notes for AI Assistants

1. **Path References**: Scripts in `scripts/` use `Path(__file__).parent.parent` to get project root. Source files in `src/` use relative paths from project root.

2. **Output Structure**: Always save outputs to the appropriate subdirectory:
   - JSON results → `output/playoffs/`, `output/validation/`, or `output/analysis/`
   - CSV data → `output/comparison/` or `output/models/`
   - Images/charts → `output/visualizations/`

3. **Model Files**: Saved models go in `models/` directory (root level, not in output)

4. **Data Files**: 
   - Raw data → `data/raw/`
   - Processed data → `data/processed/`
   - Historical data → `data/historical/`
   - Live stats → `data/` (root level)

5. **Configuration**: When adding new paths or settings, update `src/config.py` to maintain consistency

6. **Script Structure**: Entry point scripts in `scripts/` should:
   - Import from `src/` modules
   - Use organized output subdirectories
   - Include proper path handling for `scripts/` location

## Current State

The project is fully functional with:
- ✅ Complete file organization (clean structure)
- ✅ Organized output directories (labeled subfolders)
- ✅ Working playoff prediction system
- ✅ Historical data integration
- ✅ Validation and analysis tools
- ✅ Comparison with live stats
- ✅ Comprehensive documentation

The codebase follows Python best practices and is ready for further development or analysis.

