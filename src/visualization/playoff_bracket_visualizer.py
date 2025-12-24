"""
Visualize NFL Playoff Bracket Predictions

Creates visual bracket diagrams showing:
- Playoff seeding
- Matchup win probabilities
- Predicted bracket outcomes

Architecture:
- Two-pass rendering: Layout (build_slot_map) → Render (draw_team_box, draw_connector)
- Deterministic grid layout with fixed columns and rows
- Broadcast-style boxes using FancyBboxPatch with rounded corners
- Orthogonal 3-segment connectors (horizontal → vertical → horizontal)
- Cached logo loading with Pillow (handles GIF first frames via ImageSequence)
- Normalized coordinate system (0-1) for resolution independence
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path
import json
import logging
try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageSequence = None

from src.config import data_config
from src.analysis.qb_playoff_stats import get_qb_playoff_stats_for_season
from src.visualization.qb_playoff_visualizer import create_qb_playoff_radar, create_qb_playoff_bars
from src.constants import TEAM_LOGO_ALIASES, get_team_logo_path
from src.visualization import load_team_logo

logger = logging.getLogger(__name__)


class PlayoffBracketVisualizer:
    """Visualize playoff bracket predictions"""
    
    def __init__(self, results_path: str = None, results_dict: dict = None):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to JSON results file
            results_dict: Results dictionary (alternative to file)
        """
        if results_path:
            with open(results_path, 'r') as f:
                self.results = json.load(f)
        elif results_dict:
            self.results = results_dict
        else:
            raise ValueError("Must provide either results_path or results_dict")
        
        # Initialize logo cache for efficient logo loading
        self._logo_cache = {}
    
    def create_bracket_diagram(self, conference: str, output_path: str = None):
        """
        Create a visual bracket diagram for a conference.
        
        Args:
            conference: 'AFC' or 'NFC'
            output_path: Path to save the image
        """
        if conference not in self.results.get('seeding', {}):
            logger.warning(f"No seeding data for {conference}")
            return
        
        seeding = self.results['seeding'][conference]
        conf_probs = self.results.get('conference_championship_probabilities', {}).get(conference, {})
        sb_probs = self.results.get('super_bowl_probabilities', {})
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(5, 11.5, f'{conference} Playoff Bracket', 
               ha='center', fontsize=16, fontweight='bold')
        
        # Wild Card Round (left side)
        wc_y_positions = [9, 7.5, 6, 4.5]
        wc_matchups = [(2, 7), (3, 6), (4, 5)]
        
        for i, (seed1, seed2) in enumerate(wc_matchups):
            team1 = next((s['team'] for s in seeding if s['seed'] == seed1), None)
            team2 = next((s['team'] for s in seeding if s['seed'] == seed2), None)
            
            if team1 and team2:
                y = wc_y_positions[i]
                # Team boxes
                self._draw_team_box_old(ax, 1, y, team1, seed1, sb_probs.get(team1, 0))
                self._draw_team_box_old(ax, 1, y-0.6, team2, seed2, sb_probs.get(team2, 0))
        
        # Divisional Round (middle)
        div_y_positions = [8.25, 5.25]
        seed1_team = next((s['team'] for s in seeding if s['seed'] == 1), None)
        seed2_team = next((s['team'] for s in seeding if s['seed'] == 2), None)
        
        if seed1_team:
            self._draw_team_box_old(ax, 4, 8.25, seed1_team, 1, sb_probs.get(seed1_team, 0))
        if seed2_team:
            self._draw_team_box_old(ax, 4, 5.25, seed2_team, 2, sb_probs.get(seed2_team, 0))
        
        # Conference Championship (right)
        if conf_probs:
            top_team = max(conf_probs.items(), key=lambda x: x[1])[0]
            self._draw_team_box_old(ax, 7, 6.75, top_team, None, sb_probs.get(top_team, 0),
                              is_champion=True)
            ax.text(7, 6.25, f"{conf_probs[top_team]*100:.1f}%", 
                   ha='center', fontsize=10, style='italic')
        
        # Arrows showing progression
        # Wild Card to Divisional
        for i in range(3):
            y_start = wc_y_positions[i] - 0.3
            if i == 0:  # Winner goes to seed 1 matchup
                y_end = 8.25
            elif i == 1:  # Winner goes to seed 2 matchup
                y_end = 5.25
            else:  # Winner goes to seed 1 matchup
                y_end = 8.25
            
            arrow = FancyArrowPatch((2.5, y_start), (3.5, y_end),
                                   arrowstyle='->', lw=2, color='gray', alpha=0.5)
            ax.add_patch(arrow)
        
        # Divisional to Conference
        arrow1 = FancyArrowPatch((5.5, 8.25), (6.5, 7.5),
                                arrowstyle='->', lw=2, color='blue', alpha=0.7)
        arrow2 = FancyArrowPatch((5.5, 5.25), (6.5, 6.0),
                                arrowstyle='->', lw=2, color='blue', alpha=0.7)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved bracket diagram to {output_path}")
        else:
            plt.show()
    
    def _draw_team_box_old(self, ax, x, y, team, seed, sb_prob, is_champion=False):
        """Draw a team box with information (legacy method for create_bracket_diagram)"""
        # Box color based on Super Bowl probability
        if sb_prob > 0.15:
            color = 'green'
        elif sb_prob > 0.05:
            color = 'yellow'
        else:
            color = 'lightgray'
        
        if is_champion:
            color = 'gold'
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', lw=2)
        else:
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', lw=1)
        
        ax.add_patch(box)
        
        # Team name
        ax.text(x, y, team, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Seed
        if seed:
            ax.text(x, y-0.15, f"#{seed}", ha='center', va='center', fontsize=7)
        
        # Super Bowl probability
        if sb_prob > 0:
            ax.text(x, y-0.25, f"{sb_prob*100:.1f}%", ha='center', va='center', 
                   fontsize=6, style='italic')
    
    def create_summary_chart(self, output_path: str = None):
        """Create a summary chart showing Super Bowl probabilities"""
        sb_probs = self.results.get('super_bowl_probabilities', {})
        
        if not sb_probs:
            logger.warning("No Super Bowl probabilities to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        teams = list(sb_probs.keys())
        probs = [sb_probs[t] * 100 for t in teams]
        
        # Sort by probability
        sorted_data = sorted(zip(teams, probs), key=lambda x: x[1], reverse=True)
        teams, probs = zip(*sorted_data)
        
        colors = ['green' if p > 15 else 'yellow' if p > 5 else 'lightblue' 
                 for p in probs]
        
        bars = ax.barh(teams, probs, color=colors, edgecolor='black')
        
        # Add value labels
        for i, (team, prob) in enumerate(zip(teams, probs)):
            ax.text(prob + 0.5, i, f'{prob:.1f}%', 
                   va='center', fontweight='bold')
        
        ax.set_xlabel('Super Bowl Win Probability (%)', fontsize=12)
        ax.set_title(f'Super Bowl Probabilities - Season {self.results.get("season", "N/A")}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(probs) * 1.2)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary chart to {output_path}")
        else:
            plt.show()
    
    def create_win_probability_heatmap(self, predictor=None, season=None, output_path: str = None):
        """
        Create a win probability heatmap for all playoff teams.
        
        Args:
            predictor: PlayoffPredictor instance (optional, will try to load if not provided)
            season: Season year (optional, will use from results if not provided)
            output_path: Path to save the image
        """
        try:
            # Get all playoff teams from both conferences
            playoff_teams = []
            for conf in ['AFC', 'NFC']:
                if conf in self.results.get('seeding', {}):
                    seeding = self.results['seeding'][conf]
                    playoff_teams.extend([s['team'] for s in seeding])
            
            if not playoff_teams:
                logger.warning("No playoff teams found in results")
                return
            
            # If predictor is not provided, try to create one
            if predictor is None:
                try:
                    from src.simulation.predictor import PlayoffPredictor
                    team_stats_path = "data/processed/team_stats_with_fantasy_clean.csv"
                    predictor = PlayoffPredictor(team_stats_path)
                except Exception as e:
                    logger.warning(f"Could not create predictor: {e}")
                    logger.warning("Creating heatmap with simplified probabilities")
                    predictor = None
            
            # Get season from results if not provided
            if season is None:
                season = self.results.get('season', 2024)
            
            # Build win probability matrix
            n = len(playoff_teams)
            prob_matrix = np.zeros((n, n))
            
            if predictor:
                # Load team stats for the season
                try:
                    team_stats_df = predictor.load_team_stats(season)
                    
                    # Check if we can add advanced features (need game results)
                    game_results_df = predictor.load_game_results(season)
                    can_use_advanced = not game_results_df.empty
                    
                    # Add advanced features if available (same as model training)
                    if can_use_advanced:
                        try:
                            from src.simulation.model_improvements import add_advanced_features
                            team_stats_df = add_advanced_features(team_stats_df, game_results_df, season=season)
                            logger.debug("Added advanced features to team stats for prediction")
                        except Exception as e:
                            logger.debug(f"Could not add advanced features: {e}")
                            can_use_advanced = False
                    
                    # Get win probability model - use same feature set as will be available for prediction
                    win_prob_model, features = predictor._load_win_prob_model(team_stats_df, use_advanced_features=can_use_advanced)
                    
                    if win_prob_model and features:
                        # Calculate probabilities using the model
                        for i, team1 in enumerate(playoff_teams):
                            for j, team2 in enumerate(playoff_teams):
                                if i == j:
                                    prob_matrix[i, j] = 0.5
                                else:
                                    prob = predictor.predict_matchup_win_prob(
                                        team1, team2, team_stats_df,
                                        win_prob_model=win_prob_model,
                                        features=features
                                    )
                                    prob_matrix[i, j] = prob
                    else:
                        # Fallback: use Super Bowl probabilities as proxy
                        sb_probs = self.results.get('super_bowl_probabilities', {})
                        for i, team1 in enumerate(playoff_teams):
                            for j, team2 in enumerate(playoff_teams):
                                if i == j:
                                    prob_matrix[i, j] = 0.5
                                else:
                                    prob1 = sb_probs.get(team1, 0.01)
                                    prob2 = sb_probs.get(team2, 0.01)
                                    # Simple probability based on relative strength
                                    total = prob1 + prob2
                                    prob_matrix[i, j] = prob1 / total if total > 0 else 0.5
                except Exception as e:
                    logger.warning(f"Error calculating probabilities with predictor: {e}")
                    # Fallback to Super Bowl probabilities
                    sb_probs = self.results.get('super_bowl_probabilities', {})
                    for i, team1 in enumerate(playoff_teams):
                        for j, team2 in enumerate(playoff_teams):
                            if i == j:
                                prob_matrix[i, j] = 0.5
                            else:
                                prob1 = sb_probs.get(team1, 0.01)
                                prob2 = sb_probs.get(team2, 0.01)
                                total = prob1 + prob2
                                prob_matrix[i, j] = prob1 / total if total > 0 else 0.5
            else:
                # Use Super Bowl probabilities as fallback
                sb_probs = self.results.get('super_bowl_probabilities', {})
                for i, team1 in enumerate(playoff_teams):
                    for j, team2 in enumerate(playoff_teams):
                        if i == j:
                            prob_matrix[i, j] = 0.5
                        else:
                            prob1 = sb_probs.get(team1, 0.01)
                            prob2 = sb_probs.get(team2, 0.01)
                            total = prob1 + prob2
                            prob_matrix[i, j] = prob1 / total if total > 0 else 0.5
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 12))
            im = ax.imshow(prob_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(playoff_teams, rotation=45, ha='right')
            ax.set_yticklabels(playoff_teams)
            
            # Add text annotations
            for i in range(n):
                for j in range(n):
                    text = ax.text(j, i, f'{prob_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f"Playoff Teams Win Probability Heatmap - Season {season}\n(Row beats Column)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel("Opponent Team", fontsize=12)
            ax.set_ylabel("Team", fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("P(Row Team Wins)", fontsize=11)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved win probability heatmap to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating win probability heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def create_qb_playoff_radar(self, season: int, output_path: str = None, max_qbs: int = 3):
        """
        Create QB playoff performance visualizations (bar chart + limited radar chart).
        
        Generates two complementary visualizations:
        1. Grouped horizontal bar chart (all QBs) - primary
        2. Limited radar chart (top 2-3 QBs) - secondary
        
        Args:
            season: Season year
            output_path: Path to save radar chart (optional, bar chart path auto-generated)
            max_qbs: Maximum number of QBs for radar chart (default: 3)
        """
        try:
            # Get all playoff teams from both conferences
            playoff_teams = []
            for conf in ['AFC', 'NFC']:
                if conf in self.results.get('seeding', {}):
                    seeding = self.results['seeding'][conf]
                    playoff_teams.extend([s['team'] for s in seeding])
            
            if not playoff_teams:
                logger.warning("No playoff teams found in results")
                return
            
            logger.info(f"Calculating QB playoff stats for {len(playoff_teams)} playoff teams")
            
            # Get QB playoff statistics
            qb_stats = get_qb_playoff_stats_for_season(
                season=season,
                playoff_teams=playoff_teams,
                min_playoff_games=3
            )
            
            if qb_stats.empty:
                logger.warning("No QB playoff stats found (may need more playoff games)")
                return
            
            # Get Super Bowl probabilities to select top QBs for radar
            sb_probs = self.results.get('super_bowl_probabilities', {})
            
            # Add Super Bowl probability to QB stats (for sorting)
            if sb_probs and 'team' in qb_stats.columns:
                qb_stats['sb_prob'] = qb_stats['team'].map(sb_probs).fillna(0.0)
            else:
                qb_stats['sb_prob'] = 0.0
            
            # Sort by Super Bowl probability, then by EPA as tiebreaker
            if 'epa_per_play' in qb_stats.columns:
                qb_stats = qb_stats.sort_values(['sb_prob', 'epa_per_play'], ascending=[False, False])
            else:
                qb_stats = qb_stats.sort_values('sb_prob', ascending=False)
            
            # Create bar chart (all QBs)
            bar_chart_path = str(data_config.output_dir / 'visualizations' / f'qb_playoff_bars_{season}.png')
            create_qb_playoff_bars(qb_stats, bar_chart_path, season)
            logger.info(f"Created QB playoff bar chart: {bar_chart_path}")
            
            # Create limited radar chart (top 2-3 QBs)
            if output_path is None:
                output_path = str(data_config.output_dir / 'visualizations' / f'qb_playoff_radar_top_{season}.png')
            
            # Select top QBs for radar
            top_qbs = qb_stats.head(max_qbs).copy()
            if not top_qbs.empty:
                create_qb_playoff_radar(top_qbs, output_path, season, max_qbs=max_qbs)
                logger.info(f"Created QB playoff radar chart (top {len(top_qbs)} QBs): {output_path}")
            else:
                logger.warning("No QBs available for radar chart")
            
        except Exception as e:
            logger.error(f"Error creating QB playoff visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_team_color(self, team):
        """Get team color for styling"""
        # NFL team colors (primary colors)
        team_colors = {
            'KC': '#E31837', 'BUF': '#00338D', 'BAL': '#241773', 'HOU': '#03202F',
            'LAC': '#0080C6', 'DEN': '#FB4F14', 'PIT': '#FFB612', 'CIN': '#FB4F14',
            'CLE': '#311D00', 'IND': '#002C5F', 'JAX': '#006778', 'TEN': '#0C2340',
            'MIA': '#008E97', 'NE': '#002244', 'NYJ': '#125740', 'LV': '#000000',
            'DET': '#0076B6', 'PHI': '#004C54', 'TB': '#D50A0A', 'SEA': '#002244',
            'MIN': '#4F2683', 'WAS': '#773141', 'GB': '#203731', 'CHI': '#0B162A',
            'CAR': '#0085CA', 'NO': '#D3BC8D', 'ATL': '#A71930', 'DAL': '#003594',
            'NYG': '#0B2265', 'ARI': '#97233F', 'LAR': '#003594', 'SF': '#AA0000'
        }
        return team_colors.get(team, '#808080')  # Default gray
    
    def _get_team_full_name(self, team):
        """Get full team name from abbreviation"""
        team_names = {
            'KC': 'Kansas City Chiefs', 'BUF': 'Buffalo Bills', 'BAL': 'Baltimore Ravens',
            'HOU': 'Houston Texans', 'LAC': 'Los Angeles Chargers', 'DEN': 'Denver Broncos',
            'PIT': 'Pittsburgh Steelers', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
            'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'TEN': 'Tennessee Titans',
            'MIA': 'Miami Dolphins', 'NE': 'New England Patriots', 'NYJ': 'New York Jets',
            'LV': 'Las Vegas Raiders', 'DET': 'Detroit Lions', 'PHI': 'Philadelphia Eagles',
            'TB': 'Tampa Bay Buccaneers', 'SEA': 'Seattle Seahawks', 'MIN': 'Minnesota Vikings',
            'WAS': 'Washington Commanders', 'GB': 'Green Bay Packers', 'CHI': 'Chicago Bears',
            'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints', 'ATL': 'Atlanta Falcons',
            'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants', 'ARI': 'Arizona Cardinals',
            'LAR': 'Los Angeles Rams', 'SF': 'San Francisco 49ers'
        }
        return team_names.get(team, team)
    
    def _format_probability(self, prob):
        """Format probability as percentage"""
        return f"{prob*100:.1f}%"
    
    def load_gif_logo_first_frame(self, path: Path, max_px: int = 64):
        """
        Load first frame of GIF logo and convert to numpy array.
        
        Args:
            path: Path to GIF file
            max_px: Maximum dimension for resizing (maintains aspect ratio)
            
        Returns:
            numpy array (RGBA) or None if error
        """
        if not PIL_AVAILABLE or ImageSequence is None:
            return None
        
        if not path.exists():
            return None
        
        try:
            # Use ImageSequence.Iterator to get first frame
            frame = next(ImageSequence.Iterator(Image.open(path))).convert("RGBA")
            
            # Resize maintaining aspect ratio
            width, height = frame.size
            if width > height:
                new_width = max_px
                new_height = int(height * (max_px / width))
            else:
                new_height = max_px
                new_width = int(width * (max_px / height))
            
            frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            return np.array(frame)
        except Exception as e:
            logger.debug(f"Could not load GIF logo from {path}: {e}")
            return None
    
    def get_logo_for_team(self, team_abbrev: str):
        """
        Get logo numpy array for a team, using alias mapping if needed.
        Cached for efficiency - logos are loaded once and reused.
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'KC', 'JAX')
            
        Returns:
            numpy array (RGBA) or None if not found
        """
        # Check cache first
        if team_abbrev in self._logo_cache:
            return self._logo_cache[team_abbrev]
        
        # Map team abbreviation to filename
        filename_base = TEAM_LOGO_ALIASES.get(team_abbrev, team_abbrev)
        logo_path = data_config.logo_dir / f"{filename_base}.gif"
        
        # Load first frame of GIF (or static image) using Pillow
        logo_array = self.load_gif_logo_first_frame(logo_path)
        
        # Cache result (even if None) to avoid repeated file system access
        self._logo_cache[team_abbrev] = logo_array
        
        return logo_array
    
    def draw_team_box(self, ax, x_left, y_center, team_name, seed, conf, logo=None):
        """
        Draw a team box with fixed dimensions.
        
        Args:
            ax: matplotlib axis
            x_left: left edge x position (normalized 0-1)
            y_center: center y position (normalized 0-1)
            team_name: team abbreviation
            seed: seed number (or None)
            conf: 'AFC' or 'NFC'
            logo: numpy array (RGBA) or None
            
        Returns:
            bbox: (x_left, y_center, BOX_W, BOX_H) tuple
        """
        BOX_W = 0.18
        BOX_H = 0.045
        BAR_W = 0.008
        
        # Conference colors
        AFC_COLOR = '#E31837'  # red
        NFC_COLOR = '#003594'  # blue
        
        box_bottom = y_center - BOX_H / 2
        box_top = y_center + BOX_H / 2
        
        # White box with rounded corners (broadcast-style card)
        box = FancyBboxPatch(
            (x_left, box_bottom),
            BOX_W, BOX_H,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='#333333',
            linewidth=1.0
        )
        ax.add_patch(box)
        
        # Accent bar
        bar_color = AFC_COLOR if conf == 'AFC' else NFC_COLOR
        if conf == 'AFC':
            # Bar on left side
            bar = Rectangle(
                (x_left, box_bottom),
                BAR_W, BOX_H,
                facecolor=bar_color,
                edgecolor='none'
            )
        else:
            # Bar on right side
            bar = Rectangle(
                (x_left + BOX_W - BAR_W, box_bottom),
                BAR_W, BOX_H,
                facecolor=bar_color,
                edgecolor='none'
            )
        ax.add_patch(bar)
        
        # Logo (if available)
        if logo is not None:
            try:
                logo_offset = OffsetImage(logo, zoom=0.15)
                if conf == 'AFC':
                    # Logo inside left side, just after red bar
                    logo_x = x_left + BAR_W + 0.01
                    logo_alignment = (0, 0.5)
                else:
                    # Logo inside right side, just before blue bar
                    logo_x = x_left + BOX_W - BAR_W - 0.01
                    logo_alignment = (1, 0.5)
                
                logo_ab = AnnotationBbox(logo_offset, (logo_x, y_center),
                                        frameon=False, box_alignment=logo_alignment)
                ax.add_artist(logo_ab)
            except Exception as e:
                logger.debug(f"Could not add logo for {team_name}: {e}")
        
        # Team name text (centered vertically, aligned slightly toward interior)
        team_full = self._get_team_full_name(team_name)
        team_name_upper = team_full.upper()
        seed_text = f" ({seed})" if seed else ""
        full_text = f"{team_name_upper}{seed_text}"
        
        # Text position: slightly toward interior
        if conf == 'AFC':
            text_x = x_left + BAR_W + 0.025
            text_ha = 'left'
        else:
            text_x = x_left + BOX_W - BAR_W - 0.025
            text_ha = 'right'
        
        ax.text(text_x, y_center, full_text, ha=text_ha, va='center',
               fontsize=8, fontweight='bold', color='#000000',
               family='sans-serif')
        
        # Return bbox tuple
        return (x_left, y_center, BOX_W, BOX_H)
    
    def draw_connector(self, ax, src_bbox, dst_bbox, direction):
        """
        Draw orthogonal 3-segment connector (horizontal → vertical → horizontal).
        
        Args:
            ax: matplotlib axis
            src_bbox: (x_left, y_center, BOX_W, BOX_H) tuple
            dst_bbox: (x_left, y_center, BOX_W, BOX_H) tuple
            direction: 'inward_left' (AFC) or 'inward_right' (NFC)
        """
        src_x_left, src_y, src_w, src_h = src_bbox
        dst_x_left, dst_y, dst_w, dst_h = dst_bbox
        
        BOX_W = 0.18
        OFFSET = 0.03  # mx offset outside columns
        
        if direction == 'inward_left':
            # AFC: source right edge → destination left edge
            sx = src_x_left + BOX_W
            sy = src_y
            dx = dst_x_left
            dy = dst_y
            
            # Routing: horizontal to mx, vertical to dy, horizontal to destination
            mx = sx + OFFSET
            
            x_points = [sx, mx, mx, dx]
            y_points = [sy, sy, dy, dy]
        else:  # inward_right
            # NFC: source left edge → destination right edge
            sx = src_x_left
            sy = src_y
            dx = dst_x_left + BOX_W
            dy = dst_y
            
            # Routing: horizontal to mx, vertical to dy, horizontal to destination
            mx = sx - OFFSET
            
            x_points = [sx, mx, mx, dx]
            y_points = [sy, sy, dy, dy]
        
        ax.plot(x_points, y_points, color='#666666', linewidth=1.5, zorder=1)
    
    def build_frozen_bracket(self, seeding, matchup_probs):
        """
        PHASE 1: Build frozen bracket structure (pure data, zero drawing).
        Returns exactly one matchup per slot per round. NO rendering, NO matplotlib, NO x/y.
        
        Args:
            seeding: dict with 'AFC' and 'NFC' keys containing seeding lists
            matchup_probs: dict of matchup probabilities with winners
            
        Returns:
            dict: Frozen bracket structure
            Format: {
                'AFC': {
                    'WC': [(team1, seed1, team2, seed2), ...],  # 3 matchups
                    'DIV': [(team1, seed1, team2, seed2), ...],  # 2 matchups
                    'CONF': [(team1, seed1, team2, seed2)],      # 1 matchup
                },
                'NFC': { same structure },
                'SB': (afc_team, afc_seed, nfc_team, nfc_seed)
            }
        """
        frozen = {
            'AFC': {'WC': [], 'DIV': [], 'CONF': []},
            'NFC': {'WC': [], 'DIV': [], 'CONF': []},
            'SB': None
        }
        
        # Helper to get team by seed
        def get_team_by_seed(conf_seeding, seed):
            for s in conf_seeding:
                if s['seed'] == seed:
                    return s['team']
            return None
        
        # Helper to get seed by team
        def get_seed_by_team(conf_seeding, team):
            for s in conf_seeding:
                if s['team'] == team:
                    return s['seed']
            return None
        
        for conf in ['AFC', 'NFC']:
            if conf not in seeding:
                continue
            
            conf_seeding = seeding[conf]
            
            # Wild Card: 3 matchups deterministically
            wc_matchups = [(2, 7), (3, 6), (4, 5)]
            for seed1, seed2 in wc_matchups:
                team1 = get_team_by_seed(conf_seeding, seed1)
                team2 = get_team_by_seed(conf_seeding, seed2)
                if team1 and team2:
                    frozen[conf]['WC'].append((team1, seed1, team2, seed2))
            
            # Divisional: 2 matchups from matchup_probs
            div_1vWC = matchup_probs.get(('Divisional', conf, '1vWC'))
            div_WCvWC = matchup_probs.get(('Divisional', conf, 'WCvWC'))
            
            if div_1vWC:
                team1 = div_1vWC['team1']
                team2 = div_1vWC['team2']
                seed1 = div_1vWC.get('seed1', 1)
                seed2 = div_1vWC.get('seed2')
                if not seed2:
                    seed2 = get_seed_by_team(conf_seeding, team2)
                frozen[conf]['DIV'].append((team1, seed1, team2, seed2))
            
            if div_WCvWC:
                team1 = div_WCvWC['team1']
                team2 = div_WCvWC['team2']
                seed1 = div_WCvWC.get('seed1')
                seed2 = div_WCvWC.get('seed2')
                if not seed1:
                    seed1 = get_seed_by_team(conf_seeding, team1)
                if not seed2:
                    seed2 = get_seed_by_team(conf_seeding, team2)
                frozen[conf]['DIV'].append((team1, seed1, team2, seed2))
            
            # Conference Championship: 1 matchup
            conf_matchup = matchup_probs.get(('Conference', conf, 'Championship'))
            if conf_matchup:
                team1 = conf_matchup['team1']
                team2 = conf_matchup['team2']
                seed1 = get_seed_by_team(conf_seeding, team1)
                seed2 = get_seed_by_team(conf_seeding, team2)
                frozen[conf]['CONF'].append((team1, seed1, team2, seed2))
        
        # Super Bowl
        sb_matchup = matchup_probs.get(('Super Bowl', 'NFL', 'Championship'))
        if sb_matchup:
            afc_champ = sb_matchup.get('team1')
            nfc_champ = sb_matchup.get('team2')
            if afc_champ and nfc_champ:
                afc_seed = get_seed_by_team(seeding.get('AFC', []), afc_champ)
                nfc_seed = get_seed_by_team(seeding.get('NFC', []), nfc_champ)
                frozen['SB'] = (afc_champ, afc_seed, nfc_champ, nfc_seed)
        
        return frozen
    
    def _convert_to_renderer_format(self, seeding, matchup_probs):
        """
        Convert current bracket data structure to format expected by bracket_renderer_fixed.
        
        Args:
            seeding: dict with 'AFC' and 'NFC' keys containing seeding lists
            matchup_probs: dict of matchup probabilities with winners
            
        Returns:
            dict: Bracket dict in format expected by render_from_dict()
            Format: {
              "AFC": {"WC": [..3..], "DIV": [..2..], "CONF": {...}},
              "NFC": {"WC": [..3..], "DIV": [..2..], "CONF": {...}}
            }
        """
        bracket_dict = {
            'AFC': {'WC': [], 'DIV': [], 'CONF': {}},
            'NFC': {'WC': [], 'DIV': [], 'CONF': {}}
        }
        
        # Helper to get seed by team
        def get_seed_by_team(conf_seeding, team):
            for s in conf_seeding:
                if s['team'] == team:
                    return s['seed']
            return None
        
        for conf in ['AFC', 'NFC']:
            if conf not in seeding:
                continue
            
            conf_seeding = seeding[conf]
            
            # Wild Card: 3 matchups deterministically (2v7, 3v6, 4v5)
            wc_matchups = [(2, 7), (3, 6), (4, 5)]
            for seed1, seed2 in wc_matchups:
                matchup_key = f"{seed1}v{seed2}"
                wc_matchup = matchup_probs.get(('Wild Card', conf, matchup_key))
                if wc_matchup:
                    team1 = wc_matchup['team1']
                    team2 = wc_matchup['team2']
                    winner = wc_matchup.get('winner')
                    winner_seed = seed1 if winner == team1 else seed2
                    
                    bracket_dict[conf]['WC'].append({
                        'a_name': team1,
                        'a_seed': seed1,
                        'b_name': team2,
                        'b_seed': seed2,
                        'winner_name': winner,
                        'winner_seed': winner_seed
                    })
            
            # Divisional: 2 matchups
            div_1vWC = matchup_probs.get(('Divisional', conf, '1vWC'))
            div_WCvWC = matchup_probs.get(('Divisional', conf, 'WCvWC'))
            
            if div_1vWC:
                team1 = div_1vWC['team1']
                team2 = div_1vWC['team2']
                seed1 = div_1vWC.get('seed1', 1)
                seed2 = div_1vWC.get('seed2')
                if not seed2:
                    seed2 = get_seed_by_team(conf_seeding, team2)
                winner = div_1vWC.get('winner')
                winner_seed = seed1 if winner == team1 else seed2
                
                bracket_dict[conf]['DIV'].append({
                    'a_name': team1,
                    'a_seed': seed1,
                    'b_name': team2,
                    'b_seed': seed2,
                    'winner_name': winner,
                    'winner_seed': winner_seed
                })
            
            if div_WCvWC:
                team1 = div_WCvWC['team1']
                team2 = div_WCvWC['team2']
                seed1 = div_WCvWC.get('seed1')
                seed2 = div_WCvWC.get('seed2')
                if not seed1:
                    seed1 = get_seed_by_team(conf_seeding, team1)
                if not seed2:
                    seed2 = get_seed_by_team(conf_seeding, team2)
                winner = div_WCvWC.get('winner')
                winner_seed = seed1 if winner == team1 else seed2
                
                bracket_dict[conf]['DIV'].append({
                    'a_name': team1,
                    'a_seed': seed1,
                    'b_name': team2,
                    'b_seed': seed2,
                    'winner_name': winner,
                    'winner_seed': winner_seed
                })
            
            # Conference Championship: 1 matchup
            conf_matchup = matchup_probs.get(('Conference', conf, 'Championship'))
            if conf_matchup:
                team1 = conf_matchup['team1']
                team2 = conf_matchup['team2']
                seed1 = get_seed_by_team(conf_seeding, team1)
                seed2 = get_seed_by_team(conf_seeding, team2)
                winner = conf_matchup.get('winner')
                winner_seed = seed1 if winner == team1 else seed2
                
                bracket_dict[conf]['CONF'] = {
                    'a_name': team1,
                    'a_seed': seed1,
                    'b_name': team2,
                    'b_seed': seed2,
                    'winner_name': winner,
                    'winner_seed': winner_seed
                }
        
        # Add Super Bowl winner if available
        sb_matchup = matchup_probs.get(('Super Bowl', 'NFL', 'Championship'))
        if sb_matchup:
            winner = sb_matchup.get('winner')
            if winner:
                bracket_dict['SUPERBOWL'] = {
                    'a_name': sb_matchup.get('team1'),
                    'b_name': sb_matchup.get('team2'),
                    'winner_name': winner
                }
        
        return bracket_dict
    
    def build_slot_map(self, seeding, matchup_probs):
        """
        Build complete slot map with fixed positions. Each team appears exactly once per round.
        This now uses build_frozen_bracket() first, then adds positions.
        
        Args:
            seeding: dict with 'AFC' and 'NFC' keys containing seeding lists
            matchup_probs: dict of matchup probabilities with winners
            
        Returns:
            dict: Complete slot structure with fixed x,y positions for all teams
        """
        # PHASE 1: Build frozen bracket (pure data)
        frozen = self.build_frozen_bracket(seeding, matchup_probs)
        # Fixed column x-positions (normalized 0-1)
        # Figure width is 14 inches, center is at 7 inches (0.5 normalized)
        # AFC: 2 inches left of center = 5 inches = 5/14 = 0.357 normalized
        # NFC: 2 inches right of center = 9 inches = 9/14 = 0.643 normalized
        # Box width is 0.18, so left edge = center - BOX_W/2
        BOX_W = 0.18
        FIGURE_WIDTH_INCHES = 14
        CENTER_INCHES = FIGURE_WIDTH_INCHES / 2  # 7 inches
        AFC_TARGET_INCHES = CENTER_INCHES - 2  # 5 inches
        NFC_TARGET_INCHES = CENTER_INCHES + 2  # 9 inches
        
        # Convert to normalized coordinates (0-1)
        AFC_TARGET = AFC_TARGET_INCHES / FIGURE_WIDTH_INCHES  # 5/14 = 0.357
        NFC_TARGET = NFC_TARGET_INCHES / FIGURE_WIDTH_INCHES  # 9/14 = 0.643
        
        # Position boxes so their centers are at target positions
        # Left edge = center - BOX_W/2
        AFC_BASE_X = AFC_TARGET - BOX_W / 2  # ~0.267
        NFC_BASE_X = NFC_TARGET - BOX_W / 2  # ~0.553
        
        # Spread rounds horizontally with spacing (AFC goes left to right, NFC goes right to left)
        ROUND_SPACING = 0.12  # Space between rounds
        
        A0 = AFC_BASE_X - ROUND_SPACING * 2  # AFC Wild Card (leftmost)
        A1 = AFC_BASE_X - ROUND_SPACING      # AFC Divisional
        A2 = AFC_BASE_X                      # AFC Conference (at target position)
        
        N0 = NFC_BASE_X + ROUND_SPACING * 2  # NFC Wild Card (rightmost)
        N1 = NFC_BASE_X + ROUND_SPACING      # NFC Divisional
        N2 = NFC_BASE_X                      # NFC Conference (at target position)
        
        SB_L = AFC_BASE_X  # AFC finalist (same as conference)
        SB_R = NFC_BASE_X  # NFC finalist (same as conference)
        
        # Fixed row y-positions (normalized 0-1)
        # Wild Card: 3 games, each with 2 teams = 6 distinct positions
        WC_Y_POSITIONS = [
            [0.90, 0.80],  # Slot 0: top team, bottom team
            [0.65, 0.55],  # Slot 1: top team, bottom team
            [0.40, 0.30],  # Slot 2: top team, bottom team
        ]
        
        # Divisional: 2 games, each with 2 teams = 4 distinct positions
        DIV_Y_POSITIONS = [
            [0.75, 0.65],  # Slot 0: top team, bottom team
            [0.50, 0.40],  # Slot 1: top team, bottom team
        ]
        
        # Conference: 1 game with 2 teams = 2 distinct positions
        CONF_Y_POSITIONS = [0.60, 0.50]  # Top team, bottom team
        
        # Super Bowl positions (at bottom of canvas)
        SB_Y_L = 0.10  # AFC finalist (bottom)
        SB_Y_R = 0.05  # NFC finalist (bottom, slightly lower)
        
        # PHASE 2: Add fixed positions to frozen bracket (mechanical, no inference)
        slot_map = {
            'AFC': {'WC': [], 'DIV': [], 'CONF': []},
            'NFC': {'WC': [], 'DIV': [], 'CONF': []},
            'SB': {}
        }
        
        # Build connector map from frozen bracket + winners (pre-computed, not inferred during render)
        connector_map = {}  # Maps (conf, round, slot, team) -> (next_round, next_slot, next_team)
        
        for conf in ['AFC', 'NFC']:
            x_wc = A0 if conf == 'AFC' else N0
            x_div = A1 if conf == 'AFC' else N1
            x_conf = A2 if conf == 'AFC' else N2
            
            # Wild Card: assign positions to frozen matchups (mechanical)
            for slot_idx, matchup in enumerate(frozen[conf]['WC']):
                team1, seed1, team2, seed2 = matchup
                slot_map[conf]['WC'].append({
                    'slot': slot_idx,
                    'x': x_wc,
                    'y_top': WC_Y_POSITIONS[slot_idx][0],
                    'y_bottom': WC_Y_POSITIONS[slot_idx][1],
                    'teams': [(team1, seed1), (team2, seed2)]
                })
                
                # Pre-compute connector: WC winner -> DIV slot
                matchup_key = f"{seed1}v{seed2}"
                wc_matchup = matchup_probs.get(('Wild Card', conf, matchup_key))
                if wc_matchup:
                    winner = wc_matchup.get('winner')
                    # Determine which DIV slot this winner goes to
                    div_1vWC = matchup_probs.get(('Divisional', conf, '1vWC'))
                    div_WCvWC = matchup_probs.get(('Divisional', conf, 'WCvWC'))
                    if div_1vWC and winner == div_1vWC.get('team2'):
                        connector_map[(conf, 'WC', slot_idx, winner)] = ('DIV', 0, winner)
                    elif div_WCvWC and winner in [div_WCvWC.get('team1'), div_WCvWC.get('team2')]:
                        connector_map[(conf, 'WC', slot_idx, winner)] = ('DIV', 1, winner)
            
            # Divisional: assign positions to frozen matchups (mechanical)
            for slot_idx, matchup in enumerate(frozen[conf]['DIV']):
                team1, seed1, team2, seed2 = matchup
                slot_map[conf]['DIV'].append({
                    'slot': slot_idx,
                    'x': x_div,
                    'y_top': DIV_Y_POSITIONS[slot_idx][0],
                    'y_bottom': DIV_Y_POSITIONS[slot_idx][1],
                    'teams': [(team1, seed1), (team2, seed2)]
                })
                
                # Pre-compute connector: DIV winner -> CONF slot
                if slot_idx == 0:
                    div_matchup = matchup_probs.get(('Divisional', conf, '1vWC'))
                else:
                    div_matchup = matchup_probs.get(('Divisional', conf, 'WCvWC'))
                if div_matchup:
                    winner = div_matchup.get('winner')
                    connector_map[(conf, 'DIV', slot_idx, winner)] = ('CONF', 0, winner)
            
            # Conference: assign positions to frozen matchups (mechanical)
            for slot_idx, matchup in enumerate(frozen[conf]['CONF']):
                team1, seed1, team2, seed2 = matchup
                slot_map[conf]['CONF'].append({
                    'slot': slot_idx,
                    'x': x_conf,
                    'y_top': CONF_Y_POSITIONS[0],
                    'y_bottom': CONF_Y_POSITIONS[1],
                    'teams': [(team1, seed1), (team2, seed2)]
                })
                
                # Pre-compute connector: CONF winner -> SB
                conf_matchup = matchup_probs.get(('Conference', conf, 'Championship'))
                if conf_matchup:
                    winner = conf_matchup.get('winner')
                    connector_map[(conf, 'CONF', slot_idx, winner)] = ('SB', None, winner)
        
        # Super Bowl finalists (mechanical)
        if frozen['SB']:
            afc_champ, afc_seed, nfc_champ, nfc_seed = frozen['SB']
            slot_map['SB']['AFC'] = {
                'x': SB_L,
                'y': SB_Y_L,
                'team': afc_champ,
                'seed': afc_seed
            }
            slot_map['SB']['NFC'] = {
                'x': SB_R,
                'y': SB_Y_R,
                'team': nfc_champ,
                'seed': nfc_seed
            }
        
        # Store connector map in slot_map for mechanical rendering
        slot_map['_connectors'] = connector_map
        
        return slot_map
    
    def _calculate_matchup_probabilities(self, seeding, predictor, team_stats_df, 
                                        win_prob_model, features, season):
        """
        Calculate head-to-head probabilities for all bracket matchups.
        
        Returns:
            dict: Mapping of matchups to probabilities
            Format: {('round', 'conference', matchup_key): {'team1': prob1, 'team2': prob2, 'winner': team}}
        """
        matchup_probs = {}
        
        # Helper to get team by seed
        def get_team_by_seed(conf_seeding, seed):
            for s in conf_seeding:
                if s['seed'] == seed:
                    return s['team']
            return None
        
        for conf in ['AFC', 'NFC']:
            if conf not in seeding:
                continue
            
            conf_seeding = seeding[conf]
            
            # Wild Card Round: 2v7, 3v6, 4v5
            wc_matchups = [(2, 7), (3, 6), (4, 5)]
            wc_winners = []
            
            for seed1, seed2 in wc_matchups:
                team1 = get_team_by_seed(conf_seeding, seed1)
                team2 = get_team_by_seed(conf_seeding, seed2)
                
                if team1 and team2:
                    # Higher seed is home team
                    home_team = team1
                    prob1 = predictor.predict_matchup_win_prob(
                        team1, team2, team_stats_df,
                        home_team=home_team,
                        win_prob_model=win_prob_model,
                        features=features
                    )
                    prob2 = 1.0 - prob1
                    winner = team1 if prob1 > 0.5 else team2
                    
                    matchup_key = f"{seed1}v{seed2}"
                    matchup_probs[('Wild Card', conf, matchup_key)] = {
                        'team1': team1, 'team2': team2,
                        'prob1': prob1, 'prob2': prob2,
                        'seed1': seed1, 'seed2': seed2,
                        'winner': winner
                    }
                    # Store winner with their original seed
                    winner_seed = seed1 if winner == team1 else seed2
                    wc_winners.append((winner, winner_seed))
            
            # Divisional Round
            # NFL structure: Seed 1 ALWAYS plays the HIGHEST seed number (worst seed) among WC winners
            # Example: If #2, #3, #4 all win, then #1 plays #4, and #2 and #3 play each other
            # The other two remaining WC winners play each other
            seed1_team = get_team_by_seed(conf_seeding, 1)
            
            div_winners = []
            
            # Find the team with the HIGHEST seed number (worst seed) among WC winners
            # Higher seed number = worse seed (seed 4 is worse than seed 2)
            if wc_winners and seed1_team:
                # Get the team with the largest seed number (worst seed)
                highest_wc = max(wc_winners, key=lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else 0)
                highest_wc_team = highest_wc[0] if isinstance(highest_wc, tuple) else highest_wc
                highest_wc_seed = highest_wc[1] if isinstance(highest_wc, tuple) else None
                
                if highest_wc_team:
                    prob1 = predictor.predict_matchup_win_prob(
                        seed1_team, highest_wc_team, team_stats_df,
                        home_team=seed1_team,
                        win_prob_model=win_prob_model,
                        features=features
                    )
                    prob2 = 1.0 - prob1
                    winner = seed1_team if prob1 > 0.5 else highest_wc_team
                    
                    matchup_probs[('Divisional', conf, '1vWC')] = {
                        'team1': seed1_team, 'team2': highest_wc_team,
                        'prob1': prob1, 'prob2': prob2,
                        'seed1': 1, 'seed2': highest_wc_seed,
                        'winner': winner
                    }
                    div_winners.append(winner)
            
            # The other two WC winners play each other
            if len(wc_winners) >= 2 and seed1_team:
                # Get all WC winners except the one playing seed 1 (the highest seed)
                highest_wc = max(wc_winners, key=lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else 0)
                remaining_wc = [w for w in wc_winners if w != highest_wc]
                
                if len(remaining_wc) >= 2:
                    # Sort by seed number (lower seed number = better)
                    remaining_wc.sort(key=lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else float('inf'))
                    team1, seed1_val = remaining_wc[0]
                    team2, seed2_val = remaining_wc[1]
                    
                    # Higher seed (lower number) is home team
                    home_team = team1 if seed1_val < seed2_val else team2
                    prob1 = predictor.predict_matchup_win_prob(
                        team1, team2, team_stats_df,
                        home_team=home_team,
                        win_prob_model=win_prob_model,
                        features=features
                    )
                    prob2 = 1.0 - prob1
                    winner = team1 if prob1 > 0.5 else team2
                    
                    matchup_probs[('Divisional', conf, 'WCvWC')] = {
                        'team1': team1, 'team2': team2,
                        'prob1': prob1, 'prob2': prob2,
                        'seed1': seed1_val, 'seed2': seed2_val,
                        'winner': winner
                    }
                    div_winners.append(winner)
            
            # Conference Championship
            if len(div_winners) == 2:
                # Determine home team (better seed or higher probability)
                team1, team2 = div_winners[0], div_winners[1]
                # Use neutral site for conference championship
                prob1 = predictor.predict_matchup_win_prob(
                    team1, team2, team_stats_df,
                    home_team=None,
                    win_prob_model=win_prob_model,
                    features=features
                )
                prob2 = 1.0 - prob1
                winner = team1 if prob1 > 0.5 else team2
                
                matchup_probs[('Conference', conf, 'Championship')] = {
                    'team1': team1, 'team2': team2,
                    'prob1': prob1, 'prob2': prob2,
                    'seed1': None, 'seed2': None,
                    'winner': winner
                }
        
        # Super Bowl
        afc_champ = matchup_probs.get(('Conference', 'AFC', 'Championship'), {}).get('winner')
        nfc_champ = matchup_probs.get(('Conference', 'NFC', 'Championship'), {}).get('winner')
        
        if afc_champ and nfc_champ:
            # Neutral site
            prob_afc = predictor.predict_matchup_win_prob(
                afc_champ, nfc_champ, team_stats_df,
                home_team=None,
                win_prob_model=win_prob_model,
                features=features
            )
            prob_nfc = 1.0 - prob_afc
            winner = afc_champ if prob_afc > 0.5 else nfc_champ
            
            matchup_probs[('Super Bowl', 'NFL', 'Championship')] = {
                'team1': afc_champ, 'team2': nfc_champ,
                'prob1': prob_afc, 'prob2': prob_nfc,
                'seed1': None, 'seed2': None,
                'winner': winner
            }
        
        return matchup_probs
    
    def _draw_conference_bracket(self, ax, conf, slot_map, matchup_probs, bboxes):
        """
        PHASE 2: Mechanical rendering - NO INFERENCE.
        Draws exactly what's in slot_map. Uses pre-computed connector_map.
        
        Args:
            ax: Matplotlib axes object
            conf: 'AFC' or 'NFC'
            slot_map: Complete slot structure with fixed positions and connector_map
            matchup_probs: Unused - kept for compatibility but renderer doesn't use it
            bboxes: Dict to store bounding boxes for connectors
            
        Returns:
            dict: Updated bboxes dictionary
        """
        direction = 'inward_left' if conf == 'AFC' else 'inward_right'
        connector_map = slot_map.get('_connectors', {})
        
        # MECHANICAL: Draw all teams from slot_map (no inference)
        for round_name in ['WC', 'DIV', 'CONF']:
            for slot in slot_map[conf][round_name]:
                teams = slot['teams']
                # Draw top team
                if len(teams) >= 1:
                    team1, seed1 = teams[0]
                    logo1 = self.get_logo_for_team(team1)
                    bbox1 = self.draw_team_box(ax, slot['x'], slot['y_top'], team1, seed1, conf, logo1)
                    bboxes[(conf, round_name, slot['slot'], team1, 'top')] = bbox1
                # Draw bottom team
                if len(teams) >= 2:
                    team2, seed2 = teams[1]
                    logo2 = self.get_logo_for_team(team2)
                    bbox2 = self.draw_team_box(ax, slot['x'], slot['y_bottom'], team2, seed2, conf, logo2)
                    bboxes[(conf, round_name, slot['slot'], team2, 'bottom')] = bbox2
        
        # MECHANICAL: Draw connectors from pre-computed connector_map (no inference)
        # WC → DIV connectors
        for wc_slot in slot_map[conf]['WC']:
            slot_idx = wc_slot['slot']
            teams = wc_slot['teams']
            if len(teams) >= 2:
                team1, seed1 = teams[0]
                team2, seed2 = teams[1]
                # Check connector map for winner
                for team in [team1, team2]:
                    conn_key = (conf, 'WC', slot_idx, team)
                    if conn_key in connector_map:
                        next_round, next_slot, next_team = connector_map[conn_key]
                        # Get source bbox
                        wc_pos = 'top' if team == team1 else 'bottom'
                        src_bbox_key = (conf, 'WC', slot_idx, team, wc_pos)
                        # Get destination bbox
                        if next_round == 'DIV':
                            div_slot = slot_map[conf]['DIV'][next_slot]
                            div_teams = div_slot['teams']
                            if len(div_teams) >= 2:
                                div_team1 = div_teams[0][0]
                                div_pos = 'top' if next_team == div_team1 else 'bottom'
                                dst_bbox_key = (conf, 'DIV', next_slot, next_team, div_pos)
                                if src_bbox_key in bboxes and dst_bbox_key in bboxes:
                                    self.draw_connector(ax, bboxes[src_bbox_key], bboxes[dst_bbox_key], direction)
        
        # DIV → CONF connectors
        for div_slot in slot_map[conf]['DIV']:
            slot_idx = div_slot['slot']
            teams = div_slot['teams']
            if len(teams) >= 2:
                team1, seed1 = teams[0]
                team2, seed2 = teams[1]
                # Check connector map for winner
                for team in [team1, team2]:
                    conn_key = (conf, 'DIV', slot_idx, team)
                    if conn_key in connector_map:
                        next_round, next_slot, next_team = connector_map[conn_key]
                        # Get source bbox
                        div_pos = 'top' if team == team1 else 'bottom'
                        src_bbox_key = (conf, 'DIV', slot_idx, team, div_pos)
                        # Get destination bbox
                        if next_round == 'CONF':
                            conf_slot = slot_map[conf]['CONF'][next_slot]
                            conf_teams = conf_slot['teams']
                            if len(conf_teams) >= 2:
                                conf_team1 = conf_teams[0][0]
                                conf_pos = 'top' if next_team == conf_team1 else 'bottom'
                                dst_bbox_key = (conf, 'CONF', next_slot, next_team, conf_pos)
                                if src_bbox_key in bboxes and dst_bbox_key in bboxes:
                                    self.draw_connector(ax, bboxes[src_bbox_key], bboxes[dst_bbox_key], direction)
        
        # CONF → SB connectors handled in main function (shared between conferences)
        
        return bboxes
    
    def create_combined_bracket_diagram(self, predictor, season, output_path=None):
        """
        Create NFL broadcast-style bracket with deterministic grid layout.
        Creates AFC and NFC brackets separately, then combines them.
        
        Args:
            predictor: PlayoffPredictor instance with loaded model
            season: Season year
            output_path: Path to save the image
        """
        try:
            # Get seeding from results
            seeding = self.results.get('seeding', {})
            if not seeding:
                logger.warning("No seeding data available")
                return
            
            # Load team stats and prepare model for matchup calculations
            team_stats_df = predictor.load_team_stats(season)
            team_stats_df = predictor.calculate_win_pct(team_stats_df)
            team_stats_df = predictor.calculate_weighted_win_pct(team_stats_df)
            team_stats_df = predictor._add_engineered_features(team_stats_df)
            
            # Add advanced features
            try:
                from src.model_improvements import add_advanced_features
                game_results_df = predictor.load_game_results(season)
                if not game_results_df.empty:
                    team_stats_df = add_advanced_features(team_stats_df, game_results_df, season=season)
            except Exception as e:
                logger.debug(f"Could not add advanced features: {e}")
            
            # Load win probability model
            win_prob_model, features = predictor._load_win_prob_model(team_stats_df, use_advanced_features=True)
            
            if not win_prob_model:
                logger.warning("Could not load win probability model, using fallback")
                win_prob_model = None
                features = None
            
            # Calculate all matchup probabilities to determine winners
            matchup_probs = self._calculate_matchup_probabilities(
                seeding, predictor, team_stats_df, win_prob_model, features, season
            )
            
            # Convert to format expected by fixed bracket renderer
            bracket_dict = self._convert_to_renderer_format(seeding, matchup_probs)
            
            # Determine output path
            if output_path:
                if 'combined_playoff_bracket' in str(output_path):
                    output_path = str(output_path).replace('combined_playoff_bracket', 'playoff_bracket')
                elif 'playoff_bracket' not in str(output_path):
                    from src.config import data_config
                    output_path = str(data_config.output_dir / 'visualizations' / f'playoff_bracket_{season}.png')
            else:
                from src.config import data_config
                output_path = str(data_config.output_dir / 'visualizations' / f'playoff_bracket_{season}.png')
            
            # Use fixed bracket renderer
            from src.visualization.bracket_renderer_fixed import render_from_dict
            from src.config import data_config
            
            render_from_dict(
                bracket_dict,
                season=season,
                out_path=Path(output_path),
                logo_dir=data_config.logo_dir
            )

            
            logger.info(f"Saved bracket diagram to {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating bracket: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_stage_visualizations(self, predictor, season, output_dir=None):
        """
        Create separate visualizations for each playoff stage using the fixed renderer only.
        No manual connector logic. No bboxes. No inference while rendering.
        """
        from pathlib import Path
        from src.config import data_config

        if output_dir is None:
            output_dir = data_config.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        seeding = self.results.get('seeding', {})
        if not seeding:
            logger.warning("No seeding data available")
            return

        # --- build matchup_probs exactly once (your existing code) ---
        team_stats_df = predictor.load_team_stats(season)
        team_stats_df = predictor.calculate_win_pct(team_stats_df)
        team_stats_df = predictor.calculate_weighted_win_pct(team_stats_df)
        team_stats_df = predictor._add_engineered_features(team_stats_df)

        try:
            from src.model_improvements import add_advanced_features
            game_results_df = predictor.load_game_results(season)
            if not game_results_df.empty:
                team_stats_df = add_advanced_features(team_stats_df, game_results_df, season=season)
        except Exception as e:
            logger.debug(f"Could not add advanced features: {e}")

        win_prob_model, features = predictor._load_win_prob_model(team_stats_df, use_advanced_features=True)
        matchup_probs = self._calculate_matchup_probabilities(
            seeding, predictor, team_stats_df, win_prob_model, features, season
        )

        # Convert once
        bracket_dict_full = self._convert_to_renderer_format(seeding, matchup_probs)

        # Import fixed renderer
        from src.bracket_renderer_fixed import render_from_dict

        # Helper: copy and “blank out” later rounds per stage
        def mask_rounds(bracket_dict, show_wc: bool, show_div: bool, show_conf: bool):
            bd = {
                "AFC": {"WC": [], "DIV": [], "CONF": {}},
                "NFC": {"WC": [], "DIV": [], "CONF": {}}
            }
            for conf in ["AFC", "NFC"]:
                if show_wc:
                    bd[conf]["WC"] = bracket_dict[conf]["WC"]
                if show_div:
                    bd[conf]["DIV"] = bracket_dict[conf]["DIV"]
                if show_conf:
                    bd[conf]["CONF"] = bracket_dict[conf]["CONF"]
            return bd

        stages = [
            ("wild_card",   True,  False, False),
            ("divisional",  True,  True,  False),
            ("conference",  True,  True,  True),
            ("super_bowl",  True,  True,  True),
        ]

        for stage_name, show_wc, show_div, show_conf in stages:
            bd = mask_rounds(bracket_dict_full, show_wc, show_div, show_conf)
            out_path = output_dir / f"playoff_bracket_{stage_name}_{season}.png"
            render_from_dict(
                bd,
                season=season,
                out_path=out_path,
                logo_dir=data_config.logo_dir
            )
            logger.info(f"Saved {stage_name} stage bracket to {out_path}")


def main():
    """Main entry point for visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize playoff bracket predictions')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to playoff predictions JSON file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for images')
    parser.add_argument('--conference', type=str, choices=['AFC', 'NFC', 'both'],
                       default='both', help='Conference to visualize')
    
    args = parser.parse_args()
    
    visualizer = PlayoffBracketVisualizer(results_path=args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary chart
    visualizer.create_summary_chart(
        output_path=str(output_dir / 'super_bowl_probabilities.png')
    )
    
    # Create bracket diagrams
    if args.conference in ['AFC', 'both']:
        visualizer.create_bracket_diagram(
            'AFC',
            output_path=str(output_dir / 'afc_playoff_bracket.png')
        )
    
    if args.conference in ['NFC', 'both']:
        visualizer.create_bracket_diagram(
            'NFC',
            output_path=str(output_dir / 'nfc_playoff_bracket.png')
        )
    
    print("Visualizations created successfully!")


if __name__ == '__main__':
    main()

