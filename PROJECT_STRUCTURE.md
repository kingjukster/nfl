# Project Structure

## 📁 Directory Layout

```
IST_5520-Footbal-Predictions/
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── attacker.py              # Offensive player predictions (QB, RB, WR)
│   ├── MLNFL.py                 # Defensive player predictions (CB, LB, DT)
│   ├── heatMap2.py              # Team win probability heatmap
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   └── comparison/              # Comparison with live NFL stats
│       ├── __init__.py
│       ├── compare_live_stats.py
│       ├── fetch_live_nfl_stats.py
│       └── run_comparison.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── attacker.ipynb
│   ├── defence_points.ipynb
│   ├── heatmapnotebook.ipynb
│   ├── test_data.ipynb
│   └── summaryStats.ipynb
│
├── scripts/                      # Utility and test scripts
│   ├── debug_comparison.py      # Debug comparison issues
│   └── test_nfl_columns.py      # Test nfl-data-py columns
│
├── docs/                         # Documentation
│   ├── COMPARISON_GUIDE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── IMPROVEMENTS_RECOMMENDATIONS.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   ├── QUICK_FIXES.md
│   ├── ORGANIZATION_PLAN.md
│   └── REORGANIZATION_SUMMARY.md
│
├── data/                         # Data files
│   ├── raw/                      # Raw data from Kaggle
│   │   ├── yearly_player_stats_*.csv
│   │   ├── weekly_player_stats_*.csv
│   │   └── team_stats.csv
│   ├── processed/                # Processed/cleaned data
│   │   ├── team_stats_with_fantasy_clean.csv
│   │   ├── merged_file.csv
│   │   └── nfl_team_off_def_combined.csv
│   └── live_nfl_stats_*.csv      # Fetched live stats
│
├── models/                       # Saved ML models (auto-created)
│   ├── quarterback_model.pkl
│   ├── wide_receiver_model.pkl
│   ├── running_back_model.pkl
│   └── ridge_cb_model.pkl
│
├── output/                       # Output files (auto-created)
│   ├── avg_*_fantasy_by_team_*.csv
│   ├── prediction_comparison.csv
│   ├── prediction_comparison_report.txt
│   └── win_probability_heatmap_*.png
│
├── run_offensive_models.py       # Entry point: Train offensive models
├── run_defensive_models.py       # Entry point: Train defensive models
├── run_heatmap.py                # Entry point: Generate heatmap
├── run_comparison.py             # Entry point: Compare with live stats
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
└── PROJECT_STRUCTURE.md          # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Models
```bash
# Train offensive models (QB, RB, WR)
python run_offensive_models.py

# Train defensive models (CB, LB, DT)
python run_defensive_models.py

# Generate win probability heatmap
python run_heatmap.py

# Compare predictions with live stats
python run_comparison.py
```

## 📂 Directory Descriptions

### `src/`
Main source code for the project. Contains all Python modules.

### `src/comparison/`
Module for comparing predictions with live NFL statistics.

### `notebooks/`
Jupyter notebooks for exploration and analysis.

### `scripts/`
Utility scripts for debugging and testing.

### `docs/`
All project documentation and guides.

### `data/raw/`
Raw data files from Kaggle or other sources.

### `data/processed/`
Cleaned and processed data files ready for modeling.

### `models/`
Saved machine learning models (auto-created).

### `output/`
Generated output files: predictions, reports, visualizations (auto-created).

## 🔗 File Relationships

- `notebooks/attacker.ipynb` ↔ `src/attacker.py`
- `notebooks/defence_points.ipynb` ↔ `src/MLNFL.py`
- `notebooks/heatmapnotebook.ipynb` ↔ `src/heatMap2.py`
- `notebooks/test_data.ipynb` → Data processing

## 📝 Notes

- All entry point scripts are in the root directory for easy access
- Data files are organized by type (raw vs processed)
- Models and output are auto-created directories
- Documentation is centralized in `docs/`

