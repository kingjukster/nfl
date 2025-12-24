"""
Configuration file for NFL Prediction Project.
Centralizes all configuration settings for easy modification.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Defensive model settings
    defensive_position: str = "CB"  # Options: "LB", "CB", "DT"
    ridge_alpha: float = 1.0
    use_grid_search: bool = True
    grid_search_alphas: list = None
    
    # Offensive model settings
    random_forest_n_estimators: int = 200
    random_forest_random_state: int = 42
    
    # General model settings
    test_size: float = 0.3
    random_state: int = 42
    use_chronological_split: bool = True
    
    def __post_init__(self):
        if self.grid_search_alphas is None:
            self.grid_search_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


@dataclass
class DataConfig:
    """Data configuration settings."""
    # Kaggle dataset
    kaggle_dataset: str = "philiphyde1/nfl-stats-1999-2022"
    
    # Historical data settings
    super_bowl_era_start: int = 1966
    pbp_era_start: int = 1999  # Play-by-play data availability
    
    # File paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    historical_data_dir: Path = Path("data/historical")
    logo_dir: Path = Path("data/logos")
    
    # Pro-Football-Reference settings
    pfr_base_url: str = "https://www.pro-football-reference.com"
    use_pfr_cache: bool = True
    pfr_request_delay: float = 1.0  # seconds between requests
    
    # Data files (relative to project root)
    defense_csv: str = "data/raw/yearly_player_stats_defense.csv"
    offense_csv: str = "data/raw/yearly_player_stats_offense.csv"
    team_stats_csv: str = "data/processed/team_stats_with_fantasy_clean.csv"
    merged_file_csv: str = "data/processed/merged_file.csv"
    tiers_csv: str = "data/processed/cornerback_tiers_2024.csv"
    
    # Data filtering
    max_games_missed: int = 7
    min_games_per_season: int = 10
    default_season_length: int = 17
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.historical_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create historical data subdirectories
        (self.historical_data_dir / "pfr" / "team_stats").mkdir(parents=True, exist_ok=True)
        (self.historical_data_dir / "pfr" / "player_stats").mkdir(parents=True, exist_ok=True)
        (self.historical_data_dir / "pfr" / "game_results").mkdir(parents=True, exist_ok=True)
        (self.historical_data_dir / "nflfastr" / "pbp").mkdir(parents=True, exist_ok=True)
        (self.historical_data_dir / "nflfastr" / "rosters").mkdir(parents=True, exist_ok=True)
        (self.historical_data_dir / "aggregated").mkdir(parents=True, exist_ok=True)
        
        # Create organized output subdirectories
        (self.output_dir / "playoffs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "validation").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparison").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: str = "logs/nfl_predictions.log"


# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
logging_config = LoggingConfig()

