"""
Fetch NFL historical data from Pro-Football-Reference (PFR).

This module provides functions to scrape season-level team stats, player stats,
game results, and standings from PFR for the Super Bowl era (1966-present).
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from bs4 import BeautifulSoup
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_pfr_url(year: int, page_type: str = "index") -> str:
    """Construct PFR URL for a given year and page type."""
    base = data_config.pfr_base_url
    if page_type == "index":
        return f"{base}/years/{year}/"
    elif page_type == "team_stats":
        return f"{base}/years/{year}/"
    elif page_type == "standings":
        return f"{base}/years/{year}/"
    return f"{base}/years/{year}/"


def _fetch_with_retry(url: str, max_retries: int = 3, delay: float = 1.0) -> Optional[requests.Response]:
    """Fetch URL with retry logic and rate limiting."""
    # More realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    # Use session to maintain cookies
    session = requests.Session()
    session.headers.update(headers)
    
    for attempt in range(max_retries):
        try:
            time.sleep(data_config.pfr_request_delay)
            response = session.get(url, timeout=30, allow_redirects=True)
            
            # Check for 403 specifically
            if response.status_code == 403:
                logger.warning(f"403 Forbidden for {url}. PFR may be blocking automated requests.")
                logger.warning("Consider using pandas read_html() directly or manual data download.")
                # Try pandas read_html as fallback
                try:
                    logger.info(f"Attempting pandas read_html() for {url}")
                    df = pd.read_html(url, attrs={'id': 'team_stats'})
                    if df:
                        logger.info("Successfully fetched using pandas read_html()")
                        # Return a mock response object with the HTML
                        class MockResponse:
                            def __init__(self, html):
                                self.text = html
                                self.status_code = 200
                        return MockResponse(pd.io.html._get_html(url))
                except Exception as e:
                    logger.debug(f"pandas read_html also failed: {e}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None
    return None


def _parse_html_table(html_content: str, table_id: str = None) -> Optional[pd.DataFrame]:
    """Parse HTML table from PFR page."""
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try to find table by ID first
        if table_id:
            table = soup.find('table', id=table_id)
            if table:
                df = pd.read_html(str(table))[0]
                return df
        
        # Otherwise, try pandas read_html (works well with PFR)
        tables = pd.read_html(html_content)
        if tables:
            # Return the first table, or search for specific ones
            return tables[0]
    except Exception as e:
        logger.warning(f"Error parsing table: {e}")
        return None
    return None


def fetch_pfr_season_team_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Fetch team offense/defense stats for a season from PFR.
    
    Uses pandas read_html() directly as it often works better with PFR's anti-scraping.
    
    Parameters:
    -----------
    year : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Team statistics, or None if fetch fails
    """
    try:
        url = _get_pfr_url(year, "team_stats")
        logger.info(f"Fetching team stats for {year} from {url}")
        
        # Try pandas read_html first (often works better with PFR)
        try:
            logger.debug("Attempting pandas read_html() directly...")
            tables = pd.read_html(url, attrs={'id': 'team_stats'}, flavor='lxml')
            if tables and len(tables) > 0:
                df = tables[0]
                df['season'] = year
                # Clean up multi-index headers
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(0)
                logger.info(f"Successfully fetched team stats for {year} using pandas read_html()")
                return df
        except Exception as e:
            logger.debug(f"pandas read_html with table ID failed: {e}")
        
        # Try reading all tables
        try:
            logger.debug("Attempting pandas read_html() on all tables...")
            tables = pd.read_html(url, flavor='lxml')
            for table in tables:
                if len(table) > 10 and len(table) < 50:  # Reasonable size for team stats
                    # Check if it looks like team stats (has team names, stats)
                    if any(col in str(table.columns).upper() for col in ['TEAM', 'W', 'L', 'PF', 'PA']):
                        table['season'] = year
                        logger.info(f"Successfully fetched team stats for {year}")
                        return table
        except Exception as e:
            logger.warning(f"pandas read_html() failed for {year}: {str(e)[:100]}")
            # If it's a 403 or connection error, provide helpful guidance
            if '403' in str(e) or 'Forbidden' in str(e) or 'SSL' in str(e):
                logger.warning("="*60)
                logger.warning("PFR is blocking automated requests (403 Forbidden)")
                logger.warning("="*60)
                logger.warning("Options to proceed:")
                logger.warning("1. Use nflfastR for 1999+ data: --source nflfastr")
                logger.warning("2. Manually download PFR data and place in data/historical/pfr/")
                logger.warning("3. Use a VPN or different network connection")
                logger.warning("4. Increase delay between requests in config.py")
                logger.warning("="*60)
        
        # Fallback to requests + BeautifulSoup (likely to also fail with 403)
        response = _fetch_with_retry(url)
        if response:
            html_content = response.text
            table_ids = ['team_stats', 'team_offense', 'team_defense', 'AFC', 'NFC']
            
            for table_id in table_ids:
                df = _parse_html_table(html_content, table_id)
                if df is not None and len(df) > 0:
                    df['season'] = year
                    if df.columns.nlevels > 1:
                        df.columns = df.columns.droplevel(0)
                    return df
        
        logger.warning(f"No team stats table found for {year}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching team stats for {year}: {e}")
        return None


def fetch_pfr_season_standings(year: int) -> Optional[pd.DataFrame]:
    """
    Fetch season standings and records from PFR.
    
    Uses pandas read_html() directly for better compatibility.
    
    Parameters:
    -----------
    year : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Standings data, or None if fetch fails
    """
    try:
        url = _get_pfr_url(year, "standings")
        logger.info(f"Fetching standings for {year} from {url}")
        
        # Try pandas read_html first
        try:
            tables = pd.read_html(url, flavor='lxml')
            for table in tables:
                # Standings tables typically have team names and W-L records
                if 'W' in str(table.columns) and 'L' in str(table.columns):
                    table['season'] = year
                    logger.info(f"Successfully fetched standings for {year}")
                    return table
        except Exception as e:
            logger.debug(f"pandas read_html failed: {e}")
        
        # Fallback to requests
        response = _fetch_with_retry(url)
        if response:
            html_content = response.text
            table_ids = ['AFC', 'NFC', 'standings', 'div_standings']
            
            for table_id in table_ids:
                df = _parse_html_table(html_content, table_id)
                if df is not None and len(df) > 0:
                    df['season'] = year
                    return df
        
        logger.warning(f"No standings table found for {year}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching standings for {year}: {e}")
        return None


def fetch_pfr_season_player_stats(year: int, position: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch player season statistics from PFR.
    
    Parameters:
    -----------
    year : int
        Season year
    position : str, optional
        Position filter (QB, RB, WR, etc.)
        
    Returns:
    --------
    pd.DataFrame : Player statistics, or None if fetch fails
    """
    try:
        # PFR has position-specific pages
        base_url = data_config.pfr_base_url
        
        if position:
            # Position-specific URL
            pos_map = {'QB': 'passing', 'RB': 'rushing', 'WR': 'receiving', 
                      'TE': 'receiving', 'K': 'kicking', 'DEF': 'defense'}
            pos_url = pos_map.get(position.upper(), 'passing')
            url = f"{base_url}/years/{year}/{pos_url}.htm"
        else:
            # General passing leaders page (can be used as starting point)
            url = f"{base_url}/years/{year}/passing.htm"
        
        logger.info(f"Fetching player stats for {year} (position: {position}) from {url}")
        
        response = _fetch_with_retry(url)
        if not response:
            return None
        
        html_content = response.text
        
        # Try to find player stats table
        df = _parse_html_table(html_content)
        if df is not None and len(df) > 0:
            df['season'] = year
            if position:
                df['position'] = position.upper()
            return df
        
        logger.warning(f"No player stats table found for {year}, position: {position}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching player stats for {year}: {e}")
        return None


def fetch_pfr_game_results(year: int) -> Optional[pd.DataFrame]:
    """
    Fetch game-by-game results for a season from PFR.
    
    Uses pandas read_html() directly for better compatibility.
    
    Parameters:
    -----------
    year : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Game results, or None if fetch fails
    """
    try:
        # PFR has game results on schedule page
        url = f"{data_config.pfr_base_url}/years/{year}/games.htm"
        logger.info(f"Fetching game results for {year} from {url}")
        
        # Try pandas read_html first
        try:
            tables = pd.read_html(url, flavor='lxml')
            for table in tables:
                # Game tables typically have week, teams, scores
                if len(table) > 10 and any(col in str(table.columns).upper() for col in ['WEEK', 'TEAM', 'SCORE', 'VISITOR', 'HOME']):
                    table['season'] = year
                    logger.info(f"Successfully fetched game results for {year}")
                    return table
        except Exception as e:
            logger.debug(f"pandas read_html failed: {e}")
        
        # Fallback to requests
        response = _fetch_with_retry(url)
        if response:
            html_content = response.text
            df = _parse_html_table(html_content, 'games')
            if df is not None and len(df) > 0:
                df['season'] = year
                return df
        
        logger.warning(f"No game results table found for {year}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching game results for {year}: {e}")
        return None


def save_pfr_data(df: pd.DataFrame, data_type: str, year: int, 
                  position: Optional[str] = None) -> Path:
    """
    Save fetched PFR data to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    data_type : str
        Type of data (team_stats, player_stats, game_results, standings)
    year : int
        Season year
    position : str, optional
        Position (for player stats)
        
    Returns:
    --------
    Path : Path to saved file
    """
    base_dir = data_config.historical_data_dir / "pfr" / data_type
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if position:
        filename = base_dir / f"{data_type}_{year}_{position.lower()}.csv"
    else:
        filename = base_dir / f"{data_type}_{year}.csv"
    
    df.to_csv(filename, index=False)
    logger.info(f"Saved {data_type} for {year} to {filename}")
    return filename


def fetch_all_pfr_data(start_year: int = 1966, end_year: Optional[int] = None,
                       data_types: List[str] = None, resume: bool = True) -> Dict[str, List[Path]]:
    """
    Fetch all PFR data for a range of years.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1966)
    end_year : int, optional
        Ending year (default: current year)
    data_types : List[str], optional
        Types of data to fetch (team_stats, player_stats, game_results, standings)
        Default: all types
    resume : bool
        If True, skip years that already have data files
        
    Returns:
    --------
    Dict[str, List[Path]] : Dictionary mapping data types to lists of saved file paths
    """
    if end_year is None:
        end_year = datetime.now().year
    
    if data_types is None:
        data_types = ['team_stats', 'standings', 'game_results']
    
    results = {dt: [] for dt in data_types}
    
    logger.info(f"Fetching PFR data from {start_year} to {end_year}")
    logger.info(f"Data types: {data_types}")
    
    for year in range(start_year, end_year + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing year {year} ({year - start_year + 1}/{end_year - start_year + 1})")
        logger.info(f"{'='*60}")
        
        # Check if data already exists (if resuming)
        if resume:
            skip_count = 0
            for data_type in data_types:
                base_dir = data_config.historical_data_dir / "pfr" / data_type
                filename = base_dir / f"{data_type}_{year}.csv"
                if filename.exists():
                    logger.info(f"Skipping {data_type} for {year} (already exists)")
                    results[data_type].append(filename)
                    skip_count += 1
            
            if skip_count == len(data_types):
                logger.info(f"All data types for {year} already exist, skipping...")
                continue
        
        # Fetch each data type
        for data_type in data_types:
            try:
                if data_type == 'team_stats':
                    df = fetch_pfr_season_team_stats(year)
                elif data_type == 'standings':
                    df = fetch_pfr_season_standings(year)
                elif data_type == 'game_results':
                    df = fetch_pfr_game_results(year)
                elif data_type == 'player_stats':
                    # For player stats, we'd need to iterate positions
                    # For now, skip or implement position iteration
                    logger.info(f"Player stats fetching not fully implemented yet")
                    continue
                else:
                    logger.warning(f"Unknown data type: {data_type}")
                    continue
                
                if df is not None and len(df) > 0:
                    filepath = save_pfr_data(df, data_type, year)
                    results[data_type].append(filepath)
                else:
                    logger.warning(f"No data returned for {data_type} in {year}")
                    
            except Exception as e:
                logger.error(f"Error fetching {data_type} for {year}: {e}")
                continue
        
        # Small delay between years
        time.sleep(0.5)
    
    logger.info(f"\n{'='*60}")
    logger.info("PFR data fetching complete!")
    logger.info(f"{'='*60}")
    
    for data_type, files in results.items():
        logger.info(f"{data_type}: {len(files)} files")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL data from Pro-Football-Reference')
    parser.add_argument('--start-year', type=int, default=1966, help='Starting year')
    parser.add_argument('--end-year', type=int, default=None, help='Ending year (default: current year)')
    parser.add_argument('--data-types', nargs='+', 
                       choices=['team_stats', 'standings', 'game_results', 'player_stats'],
                       default=['team_stats', 'standings', 'game_results'],
                       help='Types of data to fetch')
    parser.add_argument('--no-resume', action='store_true', help='Re-fetch existing data')
    
    args = parser.parse_args()
    
    results = fetch_all_pfr_data(
        start_year=args.start_year,
        end_year=args.end_year,
        data_types=args.data_types,
        resume=not args.no_resume
    )
    
    print(f"\nFetched {sum(len(files) for files in results.values())} files total")

