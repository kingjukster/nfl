"""
Fetch NFL play-by-play data using nflfastR/nflverse.

This module provides functions to fetch play-by-play data, rosters, and schedules
from nflfastR (available from 1999-present).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_nflfastr_available() -> bool:
    """Check if nflfastR/nflreadr is available, or use nfl-data-py as fallback."""
    try:
        import nflreadr
        return True
    except ImportError:
        try:
            import nflfastr
            return True
        except ImportError:
            # Try nfl-data-py as fallback (already installed)
            try:
                import nfl_data_py as nfl
                logger.info("nflfastR/nflreadr not found, using nfl-data-py as alternative")
                return True
            except ImportError:
                logger.error("nflfastr/nflreadr not available. Options:")
                logger.error("  1. Install nfl-data-py: pip install nfl-data-py")
                logger.error("  2. Use R package nflfastR: install.packages('nflfastR') in R")
                logger.error("  3. Manual download from nflfastr.com")
                return False


def fetch_nflfastr_pbp(start_year: int = 1999, end_year: Optional[int] = None,
                       seasons: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """
    Fetch play-by-play data using nflfastR.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1999)
    end_year : int, optional
        Ending year (default: current year)
    seasons : List[int], optional
        Specific seasons to fetch (overrides start_year/end_year)
        
    Returns:
    --------
    pd.DataFrame : Play-by-play data, or None if fetch fails
    """
    if not _check_nflfastr_available():
        return None
    
    try:
        import nflreadr
        
        if seasons is None:
            if end_year is None:
                end_year = datetime.now().year
            seasons = list(range(start_year, end_year + 1))
        
        logger.info(f"Fetching play-by-play data for seasons: {seasons}")
        
        # nflreadr uses load_pbp() function
        try:
            pbp_data = nflreadr.load_pbp(seasons)
            logger.info(f"Successfully fetched {len(pbp_data)} play-by-play records")
            return pbp_data
        except AttributeError:
            # Try alternative function name
            try:
                pbp_data = nflreadr.load_pbp_data(seasons)
                logger.info(f"Successfully fetched {len(pbp_data)} play-by-play records")
                return pbp_data
            except Exception as e:
                logger.error(f"Error fetching pbp data: {e}")
                return None
                
    except ImportError:
        # Try nfl-data-py as fallback
        try:
            import nfl_data_py as nfl
            logger.info("Using nfl-data-py to fetch play-by-play data...")
            
            if seasons is None:
                if end_year is None:
                    end_year = datetime.now().year
                seasons = list(range(start_year, end_year + 1))
            
            # nfl-data-py has import_pbp_data method
            logger.info(f"Fetching play-by-play data for seasons {seasons} using nfl-data-py...")
            try:
                pbp_data = nfl.import_pbp_data(seasons)
                if pbp_data is not None and len(pbp_data) > 0:
                    logger.info(f"Successfully fetched {len(pbp_data)} play-by-play records using nfl-data-py")
                    return pbp_data
                else:
                    logger.warning("nfl-data-py returned empty play-by-play data")
                    return None
            except Exception as e:
                logger.error(f"Error fetching pbp data from nfl-data-py: {e}")
                logger.info("Note: nfl-data-py pbp data may have limited historical coverage")
                return None
                
        except ImportError:
            logger.error("Neither nflreadr, nflfastr, nor nfl-data-py is available")
            return None
    except Exception as e:
        logger.error(f"Error fetching play-by-play data: {e}")
        return None


def fetch_nflfastr_rosters(start_year: int = 1999, end_year: Optional[int] = None,
                           seasons: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """
    Fetch roster data using nflfastR.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1999)
    end_year : int, optional
        Ending year (default: current year)
    seasons : List[int], optional
        Specific seasons to fetch
        
    Returns:
    --------
    pd.DataFrame : Roster data, or None if fetch fails
    """
    if not _check_nflfastr_available():
        return None
    
    try:
        import nflreadr
        
        if seasons is None:
            if end_year is None:
                end_year = datetime.now().year
            seasons = list(range(start_year, end_year + 1))
        
        logger.info(f"Fetching roster data for seasons: {seasons}")
        
        try:
            rosters = nflreadr.load_rosters(seasons)
            logger.info(f"Successfully fetched {len(rosters)} roster records")
            return rosters
        except AttributeError:
            try:
                rosters = nflreadr.load_roster_data(seasons)
                logger.info(f"Successfully fetched {len(rosters)} roster records")
                return rosters
            except Exception as e:
                logger.error(f"Error fetching roster data: {e}")
                return None
                
    except ImportError:
        # Try nfl-data-py as fallback
        try:
            import nfl_data_py as nfl
            logger.info("Using nfl-data-py to fetch roster data...")
            all_rosters = []
            for season in seasons:
                try:
                    roster = nfl.import_seasonal_rosters([season])
                    if roster is not None and len(roster) > 0:
                        all_rosters.append(roster)
                except Exception as e:
                    logger.warning(f"Error fetching roster for {season}: {e}")
                    continue
            if all_rosters:
                combined = pd.concat(all_rosters, ignore_index=True)
                logger.info(f"Successfully fetched {len(combined)} roster records using nfl-data-py")
                return combined
            return None
        except ImportError:
            logger.error("Neither nflreadr nor nfl-data-py available for rosters")
            return None
    except Exception as e:
        logger.error(f"Error fetching roster data: {e}")
        return None


def fetch_nflfastr_schedules(start_year: int = 1999, end_year: Optional[int] = None,
                             seasons: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """
    Fetch schedule data using nflfastR.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1999)
    end_year : int, optional
        Ending year (default: current year)
    seasons : List[int], optional
        Specific seasons to fetch
        
    Returns:
    --------
    pd.DataFrame : Schedule data, or None if fetch fails
    """
    if not _check_nflfastr_available():
        return None
    
    try:
        import nflreadr
        
        if seasons is None:
            if end_year is None:
                end_year = datetime.now().year
            seasons = list(range(start_year, end_year + 1))
        
        logger.info(f"Fetching schedule data for seasons: {seasons}")
        
        try:
            schedules = nflreadr.load_schedules(seasons)
            logger.info(f"Successfully fetched {len(schedules)} schedule records")
            return schedules
        except AttributeError:
            try:
                schedules = nflreadr.load_schedule_data(seasons)
                logger.info(f"Successfully fetched {len(schedules)} schedule records")
                return schedules
            except Exception as e:
                logger.error(f"Error fetching schedule data: {e}")
                return None
                
    except ImportError:
        # Try nfl-data-py as fallback
        try:
            import nfl_data_py as nfl
            logger.info("Using nfl-data-py to fetch schedule data...")
            schedules = nfl.import_schedules(seasons)
            if schedules is not None and len(schedules) > 0:
                logger.info(f"Successfully fetched {len(schedules)} schedule records using nfl-data-py")
                return schedules
            return None
        except ImportError:
            logger.error("Neither nflreadr nor nfl-data-py available for schedules")
            return None
    except Exception as e:
        logger.error(f"Error fetching schedule data: {e}")
        return None


def save_nflfastr_data(df: pd.DataFrame, data_type: str, year: Optional[int] = None,
                       seasons: Optional[List[int]] = None) -> Path:
    """
    Save fetched nflfastR data to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    data_type : str
        Type of data (pbp, rosters, schedules)
    year : int, optional
        Single year (if saving single year)
    seasons : List[int], optional
        Seasons included (if saving multiple years)
        
    Returns:
    --------
    Path : Path to saved file
    """
    base_dir = data_config.historical_data_dir / "nflfastr" / data_type
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if year:
        filename = base_dir / f"{data_type}_{year}.csv"
    elif seasons:
        if len(seasons) == 1:
            filename = base_dir / f"{data_type}_{seasons[0]}.csv"
        else:
            filename = base_dir / f"{data_type}_{min(seasons)}_{max(seasons)}.csv"
    else:
        # Extract year from data if available
        if 'season' in df.columns:
            year = int(df['season'].min())
            filename = base_dir / f"{data_type}_{year}.csv"
        else:
            filename = base_dir / f"{data_type}_all.csv"
    
    df.to_csv(filename, index=False)
    logger.info(f"Saved {data_type} to {filename}")
    return filename


def fetch_all_nflfastr_data(start_year: int = 1999, end_year: Optional[int] = None,
                            data_types: List[str] = None, resume: bool = True) -> Dict[str, List[Path]]:
    """
    Fetch all nflfastR data for a range of years.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1999)
    end_year : int, optional
        Ending year (default: current year)
    data_types : List[str], optional
        Types of data to fetch (pbp, rosters, schedules)
        Default: all types
    resume : bool
        If True, skip years that already have data files
        
    Returns:
    --------
    Dict[str, List[Path]] : Dictionary mapping data types to lists of saved file paths
    """
    if not _check_nflfastr_available():
        return {}
    
    if end_year is None:
        end_year = datetime.now().year
    
    if data_types is None:
        data_types = ['pbp', 'rosters', 'schedules']
    
    results = {dt: [] for dt in data_types}
    
    logger.info(f"Fetching nflfastR data from {start_year} to {end_year}")
    logger.info(f"Data types: {data_types}")
    
    seasons = list(range(start_year, end_year + 1))
    
    # Check if data already exists (if resuming)
    if resume:
        skip_count = 0
        for data_type in data_types:
            base_dir = data_config.historical_data_dir / "nflfastr" / data_type
            # Check for combined file
            combined_file = base_dir / f"{data_type}_{start_year}_{end_year}.csv"
            if combined_file.exists():
                logger.info(f"Skipping {data_type} (combined file already exists)")
                results[data_type].append(combined_file)
                skip_count += 1
                continue
            
            # Check for individual year files
            all_exist = True
            for year in seasons:
                filename = base_dir / f"{data_type}_{year}.csv"
                if not filename.exists():
                    all_exist = False
                    break
            
            if all_exist and len(seasons) > 0:
                logger.info(f"All {data_type} files for range already exist")
                skip_count += 1
        
        if skip_count == len(data_types):
            logger.info("All data types already exist, skipping...")
            return results
    
    # Fetch each data type
    for data_type in data_types:
        try:
            logger.info(f"\nFetching {data_type} data...")
            
            if data_type == 'pbp':
                df = fetch_nflfastr_pbp(start_year=start_year, end_year=end_year, seasons=seasons)
            elif data_type == 'rosters':
                df = fetch_nflfastr_rosters(start_year=start_year, end_year=end_year, seasons=seasons)
            elif data_type == 'schedules':
                df = fetch_nflfastr_schedules(start_year=start_year, end_year=end_year, seasons=seasons)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                continue
            
            if df is not None and len(df) > 0:
                # Save combined file
                filepath = save_nflfastr_data(df, data_type, seasons=seasons)
                results[data_type].append(filepath)
                
                # Optionally save individual year files
                if 'season' in df.columns:
                    for year in seasons:
                        year_df = df[df['season'] == year]
                        if len(year_df) > 0:
                            year_filepath = save_nflfastr_data(year_df, data_type, year=year)
                            results[data_type].append(year_filepath)
            else:
                logger.warning(f"No data returned for {data_type}")
                
        except Exception as e:
            logger.error(f"Error fetching {data_type}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("nflfastR data fetching complete!")
    logger.info(f"{'='*60}")
    
    for data_type, files in results.items():
        logger.info(f"{data_type}: {len(files)} files")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL data from nflfastR')
    parser.add_argument('--start-year', type=int, default=1999, help='Starting year')
    parser.add_argument('--end-year', type=int, default=None, help='Ending year (default: current year)')
    parser.add_argument('--data-types', nargs='+',
                       choices=['pbp', 'rosters', 'schedules'],
                       default=['pbp', 'rosters', 'schedules'],
                       help='Types of data to fetch')
    parser.add_argument('--no-resume', action='store_true', help='Re-fetch existing data')
    
    args = parser.parse_args()
    
    results = fetch_all_nflfastr_data(
        start_year=args.start_year,
        end_year=args.end_year,
        data_types=args.data_types,
        resume=not args.no_resume
    )
    
    print(f"\nFetched {sum(len(files) for files in results.values())} files total")

