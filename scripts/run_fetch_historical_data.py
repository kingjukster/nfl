"""
Main CLI script for fetching historical NFL data.

This script orchestrates fetching data from:
- Pro-Football-Reference (1966-present)
- nflfastR (1999-present)
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.config import data_config
from src.data.fetching.fetch_pfr_data import fetch_all_pfr_data
from src.data.fetching.fetch_nflfastr_data import fetch_all_nflfastr_data
from src.data.fetching.aggregate_historical_data import create_unified_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/historical_data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch historical NFL data from Super Bowl era (1966) to present',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all data from 1966 to present
  python run_fetch_historical_data.py --start-year 1966 --data-types all

  # Fetch only team stats from PFR
  python run_fetch_historical_data.py --start-year 1966 --data-types team --source pfr

  # Fetch play-by-play data only (1999+)
  python run_fetch_historical_data.py --start-year 1999 --data-types pbp --source nflfastr

  # Fetch specific year range
  python run_fetch_historical_data.py --start-year 2020 --end-year 2023 --data-types all
        """
    )
    
    parser.add_argument('--start-year', type=int, default=1966,
                       help='Starting year (default: 1966, Super Bowl era start)')
    parser.add_argument('--end-year', type=int, default=None,
                       help='Ending year (default: current year)')
    parser.add_argument('--data-types', nargs='+',
                       choices=['team', 'player', 'games', 'pbp', 'all'],
                       default=['all'],
                       help='Types of data to fetch (default: all)')
    parser.add_argument('--source', choices=['pfr', 'nflfastr', 'both'],
                       default='both',
                       help='Data source to use (default: both)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Re-fetch existing data (default: skip existing)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of all data (same as --no-resume)')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate data into unified datasets after fetching')
    parser.add_argument('--pfr-only', action='store_true',
                       help='Fetch only PFR data (1966+)')
    parser.add_argument('--nflfastr-only', action='store_true',
                       help='Fetch only nflfastR data (1999+)')
    
    args = parser.parse_args()
    
    # Handle end year
    if args.end_year is None:
        args.end_year = datetime.now().year
    
    # Handle force refresh
    resume = not (args.no_resume or args.force_refresh)
    
    # Handle source selection
    if args.pfr_only:
        args.source = 'pfr'
    elif args.nflfastr_only:
        args.source = 'nflfastr'
    
    # Validate year ranges
    if args.start_year < 1966:
        logger.warning(f"Start year {args.start_year} is before Super Bowl era (1966). Adjusting to 1966.")
        args.start_year = 1966
    
    if args.source in ['nflfastr', 'both'] and args.start_year < 1999:
        logger.warning(f"nflfastR data only available from 1999. PFR will be used for {args.start_year}-1998.")
    
    logger.info("="*70)
    logger.info("NFL Historical Data Fetching")
    logger.info("="*70)
    logger.info(f"Start Year: {args.start_year}")
    logger.info(f"End Year: {args.end_year}")
    logger.info(f"Data Types: {args.data_types}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Resume: {resume}")
    logger.info("="*70)
    
    results = {}
    
    # Fetch PFR data
    if args.source in ['pfr', 'both']:
        logger.info("\n" + "="*70)
        logger.info("Fetching Pro-Football-Reference Data")
        logger.info("="*70)
        
        # Map data types to PFR data types
        pfr_data_types = []
        if 'all' in args.data_types or 'team' in args.data_types:
            pfr_data_types.append('team_stats')
            pfr_data_types.append('standings')
        if 'all' in args.data_types or 'games' in args.data_types:
            pfr_data_types.append('game_results')
        if 'all' in args.data_types or 'player' in args.data_types:
            pfr_data_types.append('player_stats')
        
        if pfr_data_types:
            try:
                pfr_results = fetch_all_pfr_data(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    data_types=pfr_data_types,
                    resume=resume
                )
                results['pfr'] = pfr_results
                logger.info(f"PFR fetch complete: {sum(len(files) for files in pfr_results.values())} files")
            except Exception as e:
                logger.error(f"Error fetching PFR data: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Fetch nflfastR data
    if args.source in ['nflfastr', 'both']:
        logger.info("\n" + "="*70)
        logger.info("Fetching nflfastR Data")
        logger.info("="*70)
        
        # nflfastR only available from 1999
        nflfastr_start = max(args.start_year, 1999)
        
        if nflfastr_start <= args.end_year:
            # Map data types to nflfastR data types
            nflfastr_data_types = []
            if 'all' in args.data_types or 'pbp' in args.data_types:
                nflfastr_data_types.append('pbp')
            if 'all' in args.data_types:
                nflfastr_data_types.append('rosters')
                nflfastr_data_types.append('schedules')
            
            if nflfastr_data_types:
                try:
                    nflfastr_results = fetch_all_nflfastr_data(
                        start_year=nflfastr_start,
                        end_year=args.end_year,
                        data_types=nflfastr_data_types,
                        resume=resume
                    )
                    results['nflfastr'] = nflfastr_results
                    logger.info(f"nflfastR fetch complete: {sum(len(files) for files in nflfastr_results.values())} files")
                except Exception as e:
                    logger.error(f"Error fetching nflfastR data: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        else:
            logger.info(f"Skipping nflfastR (start year {nflfastr_start} > end year {args.end_year})")
    
    # Aggregate data if requested
    if args.aggregate:
        logger.info("\n" + "="*70)
        logger.info("Aggregating Data into Unified Datasets")
        logger.info("="*70)
        
        try:
            datasets = create_unified_dataset()
            logger.info(f"Created {len(datasets)} unified datasets")
            for name, df in datasets.items():
                logger.info(f"  {name}: {len(df)} records")
            
            # Validation summary (warnings already logged during aggregation)
            logger.info("\n" + "="*70)
            logger.info("Aggregation Complete - Check warnings above for any data quality issues")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Fetching Complete!")
    logger.info("="*70)
    
    total_files = 0
    for source, source_results in results.items():
        source_total = sum(len(files) for files in source_results.values())
        total_files += source_total
        logger.info(f"{source.upper()}: {source_total} files")
    
    logger.info(f"Total files fetched: {total_files}")
    logger.info(f"\nData saved to: {data_config.historical_data_dir}")
    
    if args.aggregate:
        logger.info(f"Aggregated datasets: {data_config.historical_data_dir / 'aggregated'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nFetching interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

