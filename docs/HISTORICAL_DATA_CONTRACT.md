# Historical Data Foundation Contract

## Three-Layer Separation Rationale

### Layer A: Team Season Statistics (1966-Present)

**Purpose**: Deterministic team-level aggregation for playoff simulations and validation.

**Why Separate**:
- **Playoff seeding requires complete team records**: Every team-season must exist for accurate historical validation. Missing seasons break Monte Carlo simulations.
- **Deterministic aggregation**: Team stats are the most reliable historical data. PFR provides consistent team-level data back to 1966.
- **Single source of truth**: Playoff simulations depend on exact win/loss records, point differentials, and playoff outcomes. This layer guarantees completeness.
- **Era-agnostic core metrics**: Wins, losses, points_for, points_against are comparable across eras without normalization.

**Integration Points**:
- `playoff_predictor.py`: Seeding calculations, strength of schedule
- `playoff_validator.py`: Historical accuracy validation
- `heatMap2.py`: Team strength priors

### Layer B: Player Season Statistics (1966-Present, Where Available)

**Purpose**: Fantasy model priors and era-adjusted baselines.

**Why Separate**:
- **Incomplete early data**: Defensive player stats are sparse pre-1990s. Offensive stats are more complete but still have gaps.
- **Era awareness mandatory**: Fantasy scoring rules, season lengths, and stat definitions changed. Cannot force cross-era normalization.
- **Position-specific requirements**: Offensive (QB/RB/WR/TE) and defensive (CB/LB/DT/S) players have fundamentally different stat structures.
- **Append-only semantics**: Player data accumulates; never overwrite historical records.

**Integration Points**:
- `attacker.py`: Offensive fantasy point predictions
- `MLNFL.py`: Defensive fantasy point predictions
- Career baseline calculations for priors

### Layer C: Play-by-Play & Advanced Metrics (1999-Present)

**Purpose**: Modern ML signal (EPA, win probability, drive efficiency).

**Why Isolated**:
- **Data availability**: nflfastR play-by-play data starts in 1999. Pre-1999 play-by-play does not exist in machine-readable format.
- **Different granularity**: Play-by-play is orders of magnitude larger than season aggregates. Joining with Layers A/B only when explicitly required prevents data bloat.
- **Modern ML features**: EPA, win probability, and drive efficiency are era-specific metrics that don't translate to pre-1999.

**Integration Points**:
- Advanced win probability models (when available)
- Drive-level efficiency features
- Situational analysis (down/distance, time remaining)

**Why NOT Backfill Pre-1999**:
- No source data exists. Pre-1999 play-by-play records are not digitized.
- Attempting to synthesize play-by-play from box scores introduces massive bias.
- Layer A (team season stats) provides sufficient signal for pre-1999 validation.

---

## Column-Level Aggregation Contracts

### Layer A: `team_seasons_1966_present.csv`

**Primary Key**: `(season, team)` - exactly one row per team per season

**Required Columns** (minimum schema):
```python
{
    'season': int,                    # 1966-2025 (no gaps)
    'team': str,                      # Standard 3-letter abbreviation (normalized)
    'wins': int,                      # Regular season wins (0-17)
    'losses': int,                    # Regular season losses (0-17)
    'ties': int,                      # Regular season ties (0-2)
    'points_for': int,                # Total points scored (regular season)
    'points_against': int,            # Total points allowed (regular season)
    'point_diff': int,                # points_for - points_against
    'offensive_yards': int,           # Total offensive yards (nullable pre-1970s)
    'defensive_yards': int,           # Total yards allowed (nullable pre-1970s)
    'turnovers_forced': int,          # Takeaways (nullable pre-1980s)
    'turnovers_committed': int,       # Giveaways (nullable pre-1980s)
    'playoff_round_reached': str      # 'None', 'Wild Card', 'Divisional', 'Conference', 'Super Bowl', 'Won Super Bowl'
}
```

**Aggregation Rules**:
- Sum all regular season games (exclude playoffs)
- Use PFR team stats as primary source
- If multiple sources exist for same season, prefer most complete (fewest nulls)
- Normalize team names using `src/normalize_historical_data.py::normalize_team_names()`
- `playoff_round_reached`: Derive from playoff bracket results (not from team stats files)

**Deterministic Guarantees**:
- All seasons 1966 → current year must exist
- All 32 teams (or appropriate count for era) must exist per season
- No duplicate (season, team) pairs
- Missing years logged as errors (not warnings)

### Layer B: `player_seasons_offense.csv`

**Primary Key**: `(player_id, season, team)` - one row per player-season-team combination

**Required Columns**:
```python
{
    'player_id': str,                 # Unique identifier (PFR player_id or generated)
    'player_name': str,               # Full name
    'position': str,                  # 'QB', 'RB', 'WR', 'TE'
    'team': str,                      # Standard 3-letter abbreviation
    'season': int,                    # 1966-2025 (where available)
    'era_bucket': str,                # 'pre_1978', '1978_2001', '2002_2020', '2021_present'
    'games_played': int,              # Games played (0-17)
    'games_started': int,              # Games started (nullable)
    'passing_yards': float,           # Nullable (RB/WR/TE)
    'passing_tds': int,               # Nullable
    'interceptions': int,             # Nullable
    'rushing_yards': float,           # Nullable (QB/WR/TE)
    'rushing_tds': int,               # Nullable
    'receiving_yards': float,         # Nullable (QB/RB)
    'receiving_tds': int,             # Nullable
    'receptions': int,                # Nullable
    'fantasy_points': float,          # Standard scoring (calculated)
    'fantasy_points_ppr': float       # PPR scoring (calculated, nullable pre-2000s)
}
```

**Aggregation Rules**:
- Sum all regular season games
- If player played for multiple teams in one season, create separate rows per team
- Calculate `fantasy_points` using standard scoring:
  - QB: (passing_tds * 4) + (passing_yards / 25) - (interceptions * 2) + (rushing_yards / 10) + (rushing_tds * 6)
  - RB/WR/TE: (rushing_yards / 10) + (receiving_yards / 10) + (rushing_tds * 6) + (receiving_tds * 6) + (receptions * 0 for standard, * 1 for PPR)
- `era_bucket` assignment:
  - `pre_1978`: season < 1978 (14 games)
  - `1978_2001`: 1978 <= season < 2002 (16 games, pre-32 teams)
  - `2002_2020`: 2002 <= season < 2021 (16 games, 32 teams)
  - `2021_present`: season >= 2021 (17 games)
- Append-only: Never overwrite existing records. New data appends to file.

### Layer B: `player_seasons_defense.csv`

**Primary Key**: `(player_id, season, team)` - one row per player-season-team combination

**Required Columns**:
```python
{
    'player_id': str,                 # Unique identifier
    'player_name': str,               # Full name
    'position': str,                  # 'CB', 'LB', 'DT', 'S', 'DE'
    'team': str,                      # Standard 3-letter abbreviation
    'season': int,                    # 1966-2025 (where available, sparse pre-1990s)
    'era_bucket': str,                # Same as offense
    'games_played': int,              # Games played
    'games_started': int,              # Games started (nullable)
    'solo_tackles': int,              # Nullable (sparse pre-1990s)
    'assist_tackles': int,            # Nullable
    'sacks': float,                   # Nullable (sparse pre-1980s)
    'interceptions': int,             # Nullable
    'forced_fumbles': int,            # Nullable
    'fumble_recoveries': int,         # Nullable
    'defensive_tds': int,             # Nullable
    'safeties': int,                  # Nullable
    'fantasy_points': float,          # Standard IDP scoring (calculated)
    'fantasy_points_ppr': float       # PPR IDP scoring (calculated)
}
```

**Aggregation Rules**:
- Same team-splitting logic as offense
- Calculate `fantasy_points` using standard IDP scoring:
  - Solo tackle: 1.0, Assist: 0.5, Sack: 2.0, INT: 3.0, FF: 2.0, FR: 2.0, TD: 6.0, Safety: 2.0
- Accept incomplete early data: Log warnings for missing seasons but do not fail
- `era_bucket` same as offense

### Layer C: `pbp_data_1999_present.csv` (Existing)

**Location**: `data/historical/nflfastr/` (already correct per spec)

**Constraints**:
- Do NOT aggregate into `data/historical/aggregated/` (keep isolated)
- Only join with Layers A/B when explicitly required by model
- No backfilling pre-1999

---

## Enforcement Logic Recommendations

### Postcondition Validation (run_fetch_historical_data.py)

After aggregation, enforce these postconditions:

```python
def validate_layer_a_completeness(df: pd.DataFrame, start_year: int = 1966) -> Dict[str, Any]:
    """
    Validate Layer A completeness.
    
    Returns:
    -------
    Dict with validation results:
    {
        'missing_seasons': List[int],
        'missing_teams': Dict[int, List[str]],  # season -> list of missing teams
        'duplicate_records': List[Tuple[int, str]],
        'null_required_fields': Dict[str, int]  # column -> count of nulls
    }
    """
    current_year = datetime.now().year
    expected_seasons = set(range(start_year, current_year + 1))
    actual_seasons = set(df['season'].unique())
    missing_seasons = sorted(expected_seasons - actual_seasons)
    
    # Check for missing teams per season
    missing_teams = {}
    for season in expected_seasons:
        season_df = df[df['season'] == season]
        expected_team_count = get_expected_team_count(season)  # 26 pre-1995, 30 1995-2001, 32 2002+
        if len(season_df) < expected_team_count:
            missing_teams[season] = get_missing_teams(season, season_df)
    
    # Check duplicates
    duplicates = df[df.duplicated(subset=['season', 'team'], keep=False)]
    
    # Check required fields
    required_fields = ['season', 'team', 'wins', 'losses', 'points_for', 'points_against']
    null_counts = {col: df[col].isna().sum() for col in required_fields}
    
    return {
        'missing_seasons': missing_seasons,
        'missing_teams': missing_teams,
        'duplicate_records': duplicates[['season', 'team']].to_dict('records') if not duplicates.empty else [],
        'null_required_fields': null_counts
    }

def validate_layer_b_append_safety(existing_file: Path, new_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Ensure append-only semantics for Layer B.
    
    Returns:
    -------
    Dict with validation results:
    {
        'overlapping_records': int,  # Records that would be overwritten
        'new_records': int,
        'era_bucket_coverage': Dict[str, int]  # era_bucket -> record count
    }
    """
    if not existing_file.exists():
        return {'new_records': len(new_data), 'overlapping_records': 0}
    
    existing = pd.read_csv(existing_file)
    key_cols = ['player_id', 'season', 'team']
    
    # Check for overlaps
    existing_keys = set(zip(existing['player_id'], existing['season'], existing['team']))
    new_keys = set(zip(new_data['player_id'], new_data['season'], new_data['team']))
    overlapping = existing_keys & new_keys
    
    return {
        'overlapping_records': len(overlapping),
        'new_records': len(new_keys - existing_keys),
        'era_bucket_coverage': new_data['era_bucket'].value_counts().to_dict()
    }

def enforce_deterministic_aggregation(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    """
    Ensure deterministic output order.
    
    Always sort by sort_cols before writing to CSV.
    This guarantees identical output for identical input.
    """
    return df.sort_values(sort_cols).reset_index(drop=True)
```

### Script Enforcement in `run_fetch_historical_data.py`

```python
# After aggregation
if args.aggregate:
    validation_results = validate_layer_a_completeness(team_seasons_df)
    
    if validation_results['missing_seasons']:
        logger.error(f"MISSING SEASONS: {validation_results['missing_seasons']}")
        logger.error("Layer A contract violated: All seasons 1966-present must exist")
        sys.exit(1)  # Fail hard
    
    if validation_results['missing_teams']:
        for season, teams in validation_results['missing_teams'].items():
            logger.error(f"Season {season} missing teams: {teams}")
        sys.exit(1)
    
    if validation_results['duplicate_records']:
        logger.error(f"DUPLICATE RECORDS: {validation_results['duplicate_records']}")
        sys.exit(1)
    
    # Log null counts but don't fail (some fields are nullable)
    for col, null_count in validation_results['null_required_fields'].items():
        if null_count > 0:
            logger.warning(f"Column '{col}' has {null_count} null values")
    
    logger.info("✓ Layer A validation passed")
```

---

## ML-Specific Guidance

### Era Bias Mitigation

**Problem**: Models trained on modern data (2002+) will overfit to 32-team, salary-cap era patterns. Pre-2002 data has different competitive dynamics.

**Solutions**:

1. **Era-stratified validation**:
   ```python
   # In validation, split by era_bucket
   train_eras = ['2002_2020', '2021_present']
   test_eras = ['pre_1978', '1978_2001']  # Test on different era
   ```

2. **Era-aware features**:
   ```python
   # Add era indicators as features
   df['is_pre_salary_cap'] = (df['season'] < 1994).astype(int)
   df['is_32_team_era'] = (df['season'] >= 2002).astype(int)
   df['games_per_season'] = df['season'].apply(get_season_length)  # 14/16/17
   ```

3. **Regularization by era**:
   ```python
   # Weight samples by era recency (more weight to recent)
   sample_weights = df['season'].apply(lambda y: 1.0 if y >= 2002 else 0.5)
   model.fit(X, y, sample_weight=sample_weights)
   ```

### Validation Integrity

**Problem**: Using same era for training and validation creates false confidence.

**Solution**: Time-based cross-validation with era awareness:
```python
# Never validate on future data
for test_season in range(2010, 2025):
    train_seasons = [s for s in range(1966, test_season)]
    # Train on train_seasons, validate on test_season
    # This respects temporal order and era boundaries
```

### Fantasy Point Normalization

**Problem**: Fantasy scoring rules changed. PPR wasn't standard pre-2000s.

**Solution**: Calculate both standard and PPR, but use era-appropriate default:
```python
# For pre-2000s, use standard scoring as primary
# For 2000s+, use PPR as primary
df['fantasy_points_primary'] = df.apply(
    lambda row: row['fantasy_points_ppr'] if row['season'] >= 2000 
    else row['fantasy_points_standard'],
    axis=1
)
```

### Playoff Round Encoding

**Problem**: `playoff_round_reached` must be deterministic and comparable.

**Solution**: Use ordered categorical encoding:
```python
PLAYOFF_ROUNDS = {
    'None': 0,
    'Wild Card': 1,
    'Divisional': 2,
    'Conference': 3,
    'Super Bowl': 4,
    'Won Super Bowl': 5
}
# This allows numeric comparisons for "how far did team go"
```

---

## Summary

The three-layer separation is architecturally sound. Layer A provides deterministic team-level foundation. Layer B handles incomplete player data with era awareness. Layer C isolates modern play-by-play without forcing backfill.

**Key Enforcement Points**:
1. Layer A must have all seasons 1966→present (fail if missing)
2. Layer B is append-only (never overwrite)
3. Layer C stays isolated (no aggregation into `aggregated/`)
4. All outputs go to `data/historical/aggregated/` (no ad-hoc locations)
5. Deterministic sorting before CSV write

