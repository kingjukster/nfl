"""
Visualize NFL Playoff Bracket Predictions

Creates visual bracket diagrams showing:
- Playoff seeding
- Matchup win probabilities
- Predicted bracket outcomes
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import json
import logging

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
                self._draw_team_box(ax, 1, y, team1, seed1, sb_probs.get(team1, 0))
                self._draw_team_box(ax, 1, y-0.6, team2, seed2, sb_probs.get(team2, 0))
        
        # Divisional Round (middle)
        div_y_positions = [8.25, 5.25]
        seed1_team = next((s['team'] for s in seeding if s['seed'] == 1), None)
        seed2_team = next((s['team'] for s in seeding if s['seed'] == 2), None)
        
        if seed1_team:
            self._draw_team_box(ax, 4, 8.25, seed1_team, 1, sb_probs.get(seed1_team, 0))
        if seed2_team:
            self._draw_team_box(ax, 4, 5.25, seed2_team, 2, sb_probs.get(seed2_team, 0))
        
        # Conference Championship (right)
        if conf_probs:
            top_team = max(conf_probs.items(), key=lambda x: x[1])[0]
            self._draw_team_box(ax, 7, 6.75, top_team, None, sb_probs.get(top_team, 0),
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
    
    def _draw_team_box(self, ax, x, y, team, seed, sb_prob, is_champion=False):
        """Draw a team box with information"""
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

