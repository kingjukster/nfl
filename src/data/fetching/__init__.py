"""
Data fetching utilities.

Functions for fetching NFL data from various sources.
"""

from src.data.fetching.fetch_game_results import fetch_game_results
from src.data.fetching.fetch_injury_reports import fetch_injury_reports
from src.data.fetching.fetch_nflfastr_data import fetch_all_nflfastr_data as fetch_nflfastr_data
from src.data.fetching.fetch_pfr_data import fetch_all_pfr_data as fetch_pfr_data
from src.data.fetching.fetch_team_records import fetch_team_records
# Note: load_historical_data, aggregate_historical_data, and normalize_historical_data
# should be imported directly from their respective modules

__all__ = [
    'fetch_game_results',
    'fetch_injury_reports',
    'fetch_nflfastr_data',
    'fetch_pfr_data',
    'fetch_team_records',
]

