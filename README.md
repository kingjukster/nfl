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
│   ├── utils.py             # Utility functions
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
├── scripts/                  # Utility and test scripts
│   ├── debug_comparison.py
│   └── test_nfl_columns.py
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
│
├── run_offensive_models.py   # Entry point: Train offensive models
├── run_defensive_models.py   # Entry point: Train defensive models
├── run_heatmap.py            # Entry point: Generate heatmap
├── run_comparison.py         # Entry point: Compare with live stats
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
python run_offensive_models.py

# Train defensive player models (CB, LB, DT)
python run_defensive_models.py

# Generate win probability heatmap
python run_heatmap.py

# Predict playoff seeding and Super Bowl
python run_playoff_predictions.py --season 2024

# Compare predictions with live NFL stats
python run_comparison.py
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
python run_playoff_predictions.py --season 2024

# More simulations for better accuracy
python run_playoff_predictions.py --season 2024 --simulations 10000

# Skip visualizations (faster)
python run_playoff_predictions.py --season 2024 --no-viz
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

## 🚀 Improving Predictions

Want to improve your model's predictive accuracy? See:

- **`docs/IMPROVING_PREDICTIONS.md`** - Comprehensive improvement guide for player predictions
- **`docs/IMPROVING_PLAYOFF_PREDICTIONS.md`** - Complete guide for playoff predictions (Top 10 improvements)
- **`docs/ADDITIONAL_IMPROVEMENTS.md`** - **NEW!** 15 additional improvements beyond the basics
- **`QUICK_PLAYOFF_IMPROVEMENTS.md`** - Quick wins for playoff accuracy
- **`docs/PREDICTION_IMPROVEMENTS_SUMMARY.md`** - Quick summary
- **`QUICK_IMPROVEMENTS.md`** - Fast improvements for player models

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

- **Kaggle Dataset**: "philiphyde1/nfl-stats-1999-2022"
  - Automatically downloaded via kagglehub
  - Contains yearly player statistics (offense and defense)

## 🏆 Playoff Predictions

The system can predict:
- **Playoff Seeding**: Which teams make playoffs and their seeds (1-7)
- **Matchup Win Probabilities**: Head-to-head win probabilities
- **Complete Bracket Simulation**: Simulates entire playoff bracket
- **Super Bowl Probabilities**: Final probability of each team winning

**Quick Start:**
```bash
# Predict 2024 playoffs (1000 simulations)
python run_playoff_predictions.py --season 2024

# More simulations for better accuracy
python run_playoff_predictions.py --season 2024 --simulations 10000
```

**See `docs/PLAYOFF_PREDICTIONS_GUIDE.md` for complete documentation.**

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

- `IMPROVEMENTS_RECOMMENDATIONS.md` - Detailed improvement guide
- `QUICK_FIXES.md` - Quick fix reference
- `IMPROVEMENTS_SUMMARY.md` - Prioritized improvements
- `IMPLEMENTATION_SUMMARY.md` - What was implemented

## 🤝 Contributing

1. Follow the existing code style
2. Add logging for new features
3. Include error handling
4. Update documentation

## 📄 License

[Add your license here]

---

**Note**: Most needed CSV files are already created. The scripts will download additional data from Kaggle as needed.
