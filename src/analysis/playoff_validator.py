"""
Historical Validation System for Playoff Predictions

Tests predictions on past seasons to measure actual accuracy.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

# Handle imports - works both as module and standalone script
if __name__ == '__main__':
    # If running as standalone script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.predictor import PlayoffPredictor

logger = logging.getLogger(__name__)

# Historical playoff results (you'll need to populate this)
HISTORICAL_RESULTS = {
    2023: {
        'AFC': {
            'seeding': [
                {'team': 'BAL', 'seed': 1},
                {'team': 'BUF', 'seed': 2},
                {'team': 'KC', 'seed': 3},
                {'team': 'HOU', 'seed': 4},
                {'team': 'CLE', 'seed': 5},
                {'team': 'MIA', 'seed': 6},
                {'team': 'PIT', 'seed': 7}
            ],
            'champion': 'KC',
            'super_bowl_winner': 'KC'
        },
        'NFC': {
            'seeding': [
                {'team': 'SF', 'seed': 1},
                {'team': 'DAL', 'seed': 2},
                {'team': 'DET', 'seed': 3},
                {'team': 'TB', 'seed': 4},
                {'team': 'PHI', 'seed': 5},
                {'team': 'LAR', 'seed': 6},
                {'team': 'GB', 'seed': 7}
            ],
            'champion': 'SF',
            'super_bowl_winner': None  # Lost to KC
        }
    },
    # Add more seasons as needed
}


class PlayoffValidator:
    """Validate playoff predictions against historical results"""
    
    def __init__(self, predictor: PlayoffPredictor):
        self.predictor = predictor
        self.results = {}
    
    def validate_season(self, season: int, n_simulations: int = 10000) -> Dict:
        """
        Validate predictions for a single season.
        
        Returns accuracy metrics.
        """
        if season not in HISTORICAL_RESULTS:
            logger.warning(f"No historical data for season {season}")
            return None
        
        actual = HISTORICAL_RESULTS[season]
        
        # Make predictions
        logger.info(f"Validating season {season}...")
        predictions = self.predictor.simulate_full_playoffs(season, n_simulations)
        
        # Calculate accuracy metrics
        metrics = {
            'season': season,
            'seeding_accuracy': self._calculate_seeding_accuracy(
                predictions['seeding'], actual
            ),
            'super_bowl_accuracy': self._calculate_super_bowl_accuracy(
                predictions, actual
            ),
            'conference_champion_accuracy': self._calculate_conference_accuracy(
                predictions, actual
            ),
            'brier_score': self._calculate_brier_score(
                predictions, actual
            )
        }
        
        return metrics
    
    def _calculate_seeding_accuracy(self, predicted: Dict, actual: Dict) -> Dict:
        """Calculate seeding accuracy for both conferences"""
        results = {}
        
        for conf in ['AFC', 'NFC']:
            if conf not in predicted or conf not in actual:
                continue
            
            pred_seeds = {s['team']: s['seed'] for s in predicted[conf]}
            actual_seeds = {s['team']: s['seed'] for s in actual[conf]['seeding']}
            
            # Teams correctly seeded
            correct = sum(
                1 for team, seed in pred_seeds.items()
                if actual_seeds.get(team) == seed
            )
            
            total = len(actual_seeds)
            accuracy = correct / total if total > 0 else 0.0
            
            # Mean absolute error of seed positions
            mae = np.mean([
                abs(pred_seeds.get(team, 8) - actual_seeds.get(team, 8))
                for team in actual_seeds.keys()
            ])
            
            results[conf] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'mae': mae
            }
        
        return results
    
    def _calculate_super_bowl_accuracy(self, predictions: Dict, actual: Dict) -> Dict:
        """Calculate Super Bowl prediction accuracy"""
        sb_probs = predictions.get('super_bowl_probabilities', {})
        actual_winner = None
        
        # Find actual Super Bowl winner
        for conf in ['AFC', 'NFC']:
            if actual[conf].get('super_bowl_winner'):
                actual_winner = actual[conf]['super_bowl_winner']
                break
        
        if not actual_winner:
            return {'accuracy': None, 'predicted_prob': None}
        
        predicted_prob = sb_probs.get(actual_winner, 0.0)
        
        # Consider it "correct" if predicted probability > 25%
        is_correct = predicted_prob > 0.25
        
        return {
            'actual_winner': actual_winner,
            'predicted_prob': predicted_prob,
            'is_correct': is_correct,
            'rank': self._get_rank(actual_winner, sb_probs)
        }
    
    def _calculate_conference_accuracy(self, predictions: Dict, actual: Dict) -> Dict:
        """Calculate conference champion prediction accuracy"""
        results = {}
        
        for conf in ['AFC', 'NFC']:
            if conf not in actual:
                continue
            
            actual_champ = actual[conf].get('champion')
            if not actual_champ:
                continue
            
            conf_probs = predictions.get('conference_championship_probabilities', {}).get(conf, {})
            predicted_prob = conf_probs.get(actual_champ, 0.0)
            
            results[conf] = {
                'actual_champion': actual_champ,
                'predicted_prob': predicted_prob,
                'is_correct': predicted_prob > 0.25,
                'rank': self._get_rank(actual_champ, conf_probs)
            }
        
        return results
    
    def _calculate_brier_score(self, predictions: Dict, actual: Dict) -> float:
        """
        Calculate Brier score (lower is better).
        Measures calibration of probability predictions.
        """
        sb_probs = predictions.get('super_bowl_probabilities', {})
        actual_winner = None
        
        for conf in ['AFC', 'NFC']:
            if actual[conf].get('super_bowl_winner'):
                actual_winner = actual[conf]['super_bowl_winner']
                break
        
        if not actual_winner:
            return None
        
        # Brier score = mean((predicted_prob - actual_outcome)^2)
        # For Super Bowl winner: actual_outcome = 1 for winner, 0 for others
        brier_scores = []
        for team, prob in sb_probs.items():
            actual_outcome = 1.0 if team == actual_winner else 0.0
            brier_scores.append((prob - actual_outcome) ** 2)
        
        return np.mean(brier_scores) if brier_scores else None
    
    def _get_rank(self, team: str, probabilities: Dict) -> int:
        """Get rank of team in probability predictions"""
        sorted_teams = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for rank, (t, _) in enumerate(sorted_teams, 1):
            if t == team:
                return rank
        return len(sorted_teams) + 1
    
    def validate_multiple_seasons(self, seasons: List[int], 
                                  n_simulations: int = 10000) -> Dict:
        """Validate predictions for multiple seasons"""
        results = {
            'seasons': {},
            'overall': {}
        }
        
        all_seeding_acc = []
        all_sb_acc = []
        all_brier_scores = []
        
        for season in seasons:
            metrics = self.validate_season(season, n_simulations)
            if metrics:
                results['seasons'][season] = metrics
                
                # Aggregate metrics
                for conf in ['AFC', 'NFC']:
                    if conf in metrics['seeding_accuracy']:
                        all_seeding_acc.append(
                            metrics['seeding_accuracy'][conf]['accuracy']
                        )
                
                if metrics['super_bowl_accuracy'].get('is_correct') is not None:
                    all_sb_acc.append(
                        metrics['super_bowl_accuracy']['is_correct']
                    )
                
                if metrics['brier_score']:
                    all_brier_scores.append(metrics['brier_score'])
        
        # Calculate overall metrics
        results['overall'] = {
            'avg_seeding_accuracy': np.mean(all_seeding_acc) if all_seeding_acc else None,
            'avg_super_bowl_accuracy': np.mean(all_sb_acc) if all_sb_acc else None,
            'avg_brier_score': np.mean(all_brier_scores) if all_brier_scores else None,
            'n_seasons': len(seasons)
        }
        
        return results
    
    def save_validation_results(self, results: Dict, output_path: str):
        """Save validation results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved validation results to {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate playoff predictions')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2023],
                       help='Seasons to validate')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of simulations per season')
    parser.add_argument('--output', type=str,
                       default='output/validation/validation_results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize predictor and validator
    predictor = PlayoffPredictor("data/processed/team_stats_with_fantasy_clean.csv")
    validator = PlayoffValidator(predictor)
    
    # Run validation
    results = validator.validate_multiple_seasons(args.seasons, args.simulations)
    
    # Save results
    validator.save_validation_results(results, args.output)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    if results['overall']:
        overall = results['overall']
        print(f"\nOverall Metrics (across {overall['n_seasons']} seasons):")
        print(f"  Average Seeding Accuracy: {overall['avg_seeding_accuracy']*100:.1f}%")
        print(f"  Average Super Bowl Accuracy: {overall['avg_super_bowl_accuracy']*100:.1f}%")
        print(f"  Average Brier Score: {overall['avg_brier_score']:.4f} (lower is better)")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()

