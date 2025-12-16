# Repository Organization Plan

## New Structure

```
IST_5520-Footbal-Predictions/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── attacker.py              # Offensive player predictions
│   ├── MLNFL.py                 # Defensive player predictions
│   ├── heatMap2.py              # Team win probability heatmap
│   ├── config.py                # Configuration
│   ├── utils.py                 # Utility functions
│   └── comparison/              # Comparison scripts
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
├── scripts/                      # Utility/test scripts
│   ├── debug_comparison.py
│   └── test_nfl_columns.py
│
├── docs/                         # Documentation
│   ├── COMPARISON_GUIDE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── IMPROVEMENTS_RECOMMENDATIONS.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   └── QUICK_FIXES.md
│
├── data/                         # Data files
│   ├── raw/                      # Raw data files
│   │   ├── yearly_player_stats_*.csv
│   │   ├── weekly_player_stats_*.csv
│   │   └── team_stats.csv
│   ├── processed/                # Processed data files
│   │   ├── team_stats_with_fantasy_clean.csv
│   │   ├── merged_file.csv
│   │   └── nfl_team_off_def_combined.csv
│   └── live_nfl_stats_*.csv      # Live stats (fetched)
│
├── models/                       # Saved ML models (auto-created)
├── output/                       # Output files (auto-created)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Files to Move

### Source Code → src/
- attacker.py
- MLNFL.py
- heatMap2.py
- config.py
- utils.py

### Comparison → src/comparison/
- compare_live_stats.py
- fetch_live_nfl_stats.py
- run_comparison.py

### Notebooks → notebooks/
- attacker.ipynb
- defence_points.ipynb
- heatmapnotebook.ipynb
- test_data.ipynb
- summaryStats.ipynb

### Documentation → docs/
- COMPARISON_GUIDE.md
- IMPLEMENTATION_SUMMARY.md
- IMPROVEMENTS_RECOMMENDATIONS.md
- IMPROVEMENTS_SUMMARY.md
- QUICK_FIXES.md

### Scripts → scripts/
- debug_comparison.py
- test_nfl_columns.py

### Data → data/raw/ or data/processed/
- All CSV files organized by type

