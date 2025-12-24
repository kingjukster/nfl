# Modular Code Structure

This document describes the modular, reusable code structure implemented in the NFL Prediction Project.

## Overview

The codebase has been refactored into a modular structure with clear separation of concerns:

- **Constants**: Centralized NFL constants (teams, conferences, mappings)
- **Data Utilities**: Reusable data loading and processing functions
- **Visualization Utilities**: Shared visualization styling and team visualization functions
- **Backward Compatibility**: Existing code continues to work through wrapper functions

## Module Structure

### `src/constants/`

Centralized location for all NFL-related constants.

#### `src/constants/teams.py`
- `TEAM_COLORS`: Dictionary of team primary colors (hex codes)
- `TEAM_LOGO_ALIASES`: Mapping for logo file names (e.g., JAX → JAG.gif)
- `TEAM_NAME_MAPPING`: Comprehensive team name normalization mapping
- `STANDARD_TEAM_ABBREVIATIONS`: Set of standard NFL team abbreviations
- `get_team_color(team)`: Get team color with fallback
- `get_team_logo_path(team)`: Get path to team logo file
- `normalize_team_name(team)`: Normalize team name to standard abbreviation

#### `src/constants/conferences.py`
- `NFL_CONFERENCES`: Conference and division structure
- `DOME_TEAMS`: List of dome stadium teams
- `COLD_WEATHER_TEAMS`: List of cold weather teams
- `get_team_conference(team)`: Get team's conference (AFC/NFC)
- `get_team_division(team)`: Get team's division

### `src/data/`

Reusable data loading and processing utilities.

#### `src/data/loaders.py`
- `load_csv_safe(file_path, required=True, **kwargs)`: Safe CSV loading with error handling
- `load_json_safe(file_path, required=True)`: Safe JSON loading with error handling
- `validate_dataframe(df, min_rows=1, required_columns=None)`: Validate dataframe
- `ensure_columns_exist(df, columns, fill_value=None)`: Ensure columns exist

#### `src/data/processors.py`
- `normalize_team_names_in_df(df, team_column='team')`: Normalize team names in dataframe
- `chronological_split(X, y, test_season=None, season_col='season')`: Chronological train/test split
- `filter_by_season(df, season, season_col='season')`: Filter dataframe by season

### `src/visualization/`

Shared visualization utilities.

#### `src/visualization/team_visualization.py`
- `get_team_color(team)`: Get team color for visualization
- `load_team_logo(team, size=None)`: Load team logo as numpy array
- `apply_team_colors(teams)`: Get list of team colors for multiple teams

#### `src/visualization/styling.py`
- `setup_plot_style(style, fallback_styles=None)`: Setup matplotlib style with fallbacks
- `apply_professional_styling(fig, ax, title=None, subtitle=None, background_color='#F8F9FA')`: Apply professional styling
- `save_figure_safe(fig, output_path, dpi=300, bbox_inches='tight')`: Save figure with error handling

## Usage Examples

### Using Constants

```python
from src.constants import get_team_color, get_team_conference, normalize_team_name

# Get team color
color = get_team_color('KC')  # Returns '#E31837'

# Get team conference
conference = get_team_conference('KC')  # Returns 'AFC'

# Normalize team name
normalized = normalize_team_name('Kansas City')  # Returns 'KC'
```

### Using Data Utilities

```python
from src.data import load_csv_safe, chronological_split

# Load CSV safely
df = load_csv_safe('data/file.csv', required=True)

# Chronological split
X_train, X_test, y_train, y_test = chronological_split(
    X, y, test_season=2023
)
```

### Using Visualization Utilities

```python
from src.visualization import setup_plot_style, apply_professional_styling, get_team_color

# Setup plot style
setup_plot_style('seaborn-v0_8-darkgrid')

# Create figure and apply styling
fig, ax = plt.subplots()
apply_professional_styling(fig, ax, title='My Chart', subtitle='Subtitle')

# Get team colors
colors = [get_team_color(team) for team in teams]
```

## Refactored Files

The following files have been updated to use the new modular structure:

1. **src/qb_playoff_visualizer.py**
   - Uses `src.constants.get_team_color`
   - Uses `src.visualization` utilities for styling

2. **src/playoff_bracket_visualizer.py**
   - Uses `src.constants.TEAM_LOGO_ALIASES`
   - Uses `src.visualization.load_team_logo`

3. **src/playoff_predictor.py**
   - Uses `src.constants` for NFL_CONFERENCES, DOME_TEAMS, etc.
   - Uses `src.constants.normalize_team_name`

4. **src/normalize_historical_data.py**
   - Uses `src.constants` for team name mapping
   - Uses centralized `normalize_team_name` function

5. **src/utils.py**
   - Maintains backward compatibility
   - Wraps new modular data utilities

## Benefits

1. **DRY Principle**: No code duplication - constants and utilities defined once
2. **Maintainability**: Changes to team colors/mappings only need to be made in one place
3. **Reusability**: Functions can be easily imported and used across modules
4. **Testability**: Modular functions are easier to unit test
5. **Consistency**: All modules use the same team colors, mappings, and utilities
6. **Backward Compatibility**: Existing code continues to work through wrapper functions

## Future Improvements

- Add unit tests for each module
- Create additional utility modules (e.g., `src/models/` for model utilities)
- Add type hints throughout
- Create API documentation with Sphinx

