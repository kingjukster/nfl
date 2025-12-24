# Historical Data Fetching Guide

## Overview

This project supports fetching NFL historical data from the Super Bowl era (1966) to present using multiple data sources.

## Data Sources

### Pro-Football-Reference (PFR) - 1966 to Present

**Note:** PFR has anti-scraping measures that may block automated requests with 403 Forbidden errors. This is common and expected.

**Available Data:**
- Team season statistics
- Season standings
- Game-by-game results
- Player season totals

**Limitations:**
- May require manual intervention if blocked
- Rate limiting recommended (1-2 seconds between requests)
- Some years may have incomplete data

### nflfastR/nflverse - 1999 to Present

**Available Data:**
- Play-by-play data
- Roster information
- Schedule data
- Advanced analytics (EPA, win probability)

**Advantages:**
- More reliable API access
- Higher quality data for modern era
- Better for analytics and modeling

## Usage

### Fetch All Historical Data

```bash
# Fetch from 1966 to present (may encounter PFR blocking)
python run_fetch_historical_data.py --start-year 1966 --data-types all --aggregate

# Fetch only nflfastR data (1999+, more reliable)
python run_fetch_historical_data.py --start-year 1999 --data-types all --source nflfastr --aggregate
```

### Fetch Specific Data Types

```bash
# Team stats only
python run_fetch_historical_data.py --start-year 1966 --data-types team --source pfr

# Play-by-play data (1999+)
python run_fetch_historical_data.py --start-year 1999 --data-types pbp --source nflfastr
```

## Handling 403 Forbidden Errors

If you encounter 403 Forbidden errors from PFR:

### Option 1: Use nflfastR Only (Recommended for 1999+)

```bash
python run_fetch_historical_data.py --start-year 1999 --data-types all --source nflfastr
```

### Option 2: Manual Download from PFR

1. Visit https://www.pro-football-reference.com/years/
2. Download data for each year manually
3. Place files in `data/historical/pfr/` with naming:
   - `team_stats_{year}.csv`
   - `standings_{year}.csv`
   - `game_results_{year}.csv`

### Option 3: Use VPN or Different Network

Some networks/IPs may be blocked. Try:
- Using a VPN
- Different network connection
- Running from a different location

### Option 4: Increase Request Delays

Edit `src/config.py`:
```python
pfr_request_delay: float = 3.0  # Increase from 1.0 to 3.0 seconds
```

## Data Storage Structure

```
data/historical/
├── pfr/
│   ├── team_stats/
│   ├── player_stats/
│   └── game_results/
├── nflfastr/
│   ├── pbp/
│   └── rosters/
└── aggregated/
    ├── team_season_stats_1966_present.csv
    ├── game_results_1966_present.csv
    ├── player_season_stats_1966_present.csv
    └── pbp_data_1999_present.csv
```

## Recommended Approach

For best results:

1. **1999-Present**: Use nflfastR (most reliable)
   ```bash
   python run_fetch_historical_data.py --start-year 1999 --source nflfastr --data-types all
   ```

2. **1966-1998**: 
   - Try PFR automated fetching
   - If blocked, manually download from PFR
   - Or use existing Kaggle data (1999-2022) as baseline

3. **Aggregate**: Combine all sources
   ```bash
   python run_fetch_historical_data.py --aggregate
   ```

## Troubleshooting

### Issue: All PFR requests return 403

**Solution**: PFR is blocking your IP/network. Use nflfastR for 1999+ data or manually download PFR data.

### Issue: pandas read_html() fails

**Solution**: 
- Ensure `lxml` and `html5lib` are installed: `pip install lxml html5lib`
- Try different network/VPN
- Use manual download method

### Issue: nflfastR package not found

**Solution**: Install with `pip install nflreadr` or `pip install nflfastr`

### Issue: Data files not found after fetching

**Solution**: Check `data/historical/` directory. Files are saved incrementally. Use `--no-resume` to re-fetch.

## Alternative Data Sources

If PFR continues to block:

1. **Stathead** (Sports Reference): Exportable data, may require subscription
2. **NFL.com Official API**: Limited historical data
3. **Kaggle Datasets**: Search for NFL historical datasets
4. **Manual Collection**: Download from PFR manually and organize

## Best Practices

1. **Start Small**: Test with a few years first (e.g., 2020-2024)
2. **Use Resume**: Default behavior skips already-fetched years
3. **Aggregate After**: Run with `--aggregate` flag to create unified datasets
4. **Check Logs**: Review `logs/historical_data_fetch.log` for issues
5. **Validate Data**: Check aggregated files for completeness

## Example Workflow

```bash
# 1. Fetch nflfastR data (most reliable)
python run_fetch_historical_data.py --start-year 1999 --source nflfastr --data-types all

# 2. Try PFR for older years (may fail due to blocking)
python run_fetch_historical_data.py --start-year 1966 --end-year 1998 --source pfr --data-types team

# 3. Aggregate everything
python run_fetch_historical_data.py --aggregate

# 4. Verify results
ls -lh data/historical/aggregated/
```

