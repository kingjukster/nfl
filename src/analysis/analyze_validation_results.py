"""
Analyze validation results and provide insights
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_validation_results(file_path: str) -> dict:
    """Load validation results from JSON"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_seeding_accuracy(results: dict) -> pd.DataFrame:
    """Analyze seeding accuracy by conference and season"""
    data = []
    
    for season, metrics in results.get('seasons', {}).items():
        seeding = metrics.get('seeding_accuracy', {})
        for conf in ['AFC', 'NFC']:
            if conf in seeding:
                data.append({
                    'season': int(season),
                    'conference': conf,
                    'accuracy': seeding[conf]['accuracy'],
                    'correct': seeding[conf]['correct'],
                    'total': seeding[conf]['total'],
                    'mae': seeding[conf]['mae']
                })
    
    return pd.DataFrame(data)

def analyze_super_bowl_accuracy(results: dict) -> pd.DataFrame:
    """Analyze Super Bowl prediction accuracy"""
    data = []
    
    for season, metrics in results.get('seasons', {}).items():
        sb = metrics.get('super_bowl_accuracy', {})
        if sb:
            data.append({
                'season': int(season),
                'actual_winner': sb.get('actual_winner'),
                'predicted_prob': sb.get('predicted_prob', 0),
                'is_correct': sb.get('is_correct', False),
                'rank': sb.get('rank', 999)
            })
    
    return pd.DataFrame(data)

def analyze_conference_champions(results: dict) -> pd.DataFrame:
    """Analyze conference champion predictions"""
    data = []
    
    for season, metrics in results.get('seasons', {}).items():
        conf_champs = metrics.get('conference_champion_accuracy', {})
        for conf in ['AFC', 'NFC']:
            if conf in conf_champs:
                champ = conf_champs[conf]
                data.append({
                    'season': int(season),
                    'conference': conf,
                    'actual_champion': champ.get('actual_champion'),
                    'predicted_prob': champ.get('predicted_prob', 0),
                    'is_correct': champ.get('is_correct', False),
                    'rank': champ.get('rank', 999)
                })
    
    return pd.DataFrame(data)

def create_analysis_report(results: dict, output_path: str = None):
    """Create comprehensive analysis report"""
    
    print("="*70)
    print("PLAYOFF PREDICTION VALIDATION ANALYSIS")
    print("="*70)
    
    # Overall metrics
    overall = results.get('overall', {})
    print(f"\nOVERALL METRICS (across {overall.get('n_seasons', 0)} seasons)")
    print("-"*70)
    print(f"Average Seeding Accuracy: {overall.get('avg_seeding_accuracy', 0)*100:.1f}%")
    print(f"Average Super Bowl Accuracy: {overall.get('avg_super_bowl_accuracy', 0)*100:.1f}%")
    print(f"Average Brier Score: {overall.get('avg_brier_score', 0):.4f} (lower is better)")
    
    # Seeding analysis
    seeding_df = analyze_seeding_accuracy(results)
    if not seeding_df.empty:
        print(f"\nSEEDING ACCURACY ANALYSIS")
        print("-"*70)
        print("\nBy Conference:")
        conf_summary = seeding_df.groupby('conference').agg({
            'accuracy': 'mean',
            'mae': 'mean',
            'correct': 'sum',
            'total': 'sum'
        })
        for conf, row in conf_summary.iterrows():
            print(f"  {conf}:")
            print(f"    Accuracy: {row['accuracy']*100:.1f}%")
            print(f"    Mean Absolute Error: {row['mae']:.2f} seed positions")
            print(f"    Correct: {int(row['correct'])}/{int(row['total'])}")
        
        print("\nBy Season:")
        for season in sorted(seeding_df['season'].unique()):
            season_data = seeding_df[seeding_df['season'] == season]
            print(f"  {season}:")
            for _, row in season_data.iterrows():
                print(f"    {row['conference']}: {row['accuracy']*100:.1f}% ({int(row['correct'])}/{int(row['total'])})")
    
    # Super Bowl analysis
    sb_df = analyze_super_bowl_accuracy(results)
    if not sb_df.empty:
        print(f"\nSUPER BOWL PREDICTION ANALYSIS")
        print("-"*70)
        print(f"Correct Predictions: {sb_df['is_correct'].sum()}/{len(sb_df)} ({sb_df['is_correct'].mean()*100:.1f}%)")
        print(f"Average Predicted Probability for Actual Winner: {sb_df['predicted_prob'].mean()*100:.1f}%")
        print(f"Average Rank of Actual Winner: {sb_df['rank'].mean():.1f}")
        
        print("\nBy Season:")
        for _, row in sb_df.iterrows():
            status = "CORRECT" if row['is_correct'] else "MISSED"
            print(f"  {row['season']}: {row['actual_winner']} won")
            print(f"    Predicted Probability: {row['predicted_prob']*100:.1f}%")
            print(f"    Rank: #{row['rank']}")
            print(f"    Status: {status}")
    
    # Conference champion analysis
    conf_df = analyze_conference_champions(results)
    if not conf_df.empty:
        print(f"\nCONFERENCE CHAMPION PREDICTION ANALYSIS")
        print("-"*70)
        print(f"Correct Predictions: {conf_df['is_correct'].sum()}/{len(conf_df)} ({conf_df['is_correct'].mean()*100:.1f}%)")
        
        print("\nBy Conference:")
        for conf in ['AFC', 'NFC']:
            conf_data = conf_df[conf_df['conference'] == conf]
            if not conf_data.empty:
                print(f"  {conf}:")
                print(f"    Accuracy: {conf_data['is_correct'].mean()*100:.1f}%")
                print(f"    Avg Predicted Prob: {conf_data['predicted_prob'].mean()*100:.1f}%")
                print(f"    Avg Rank: {conf_data['rank'].mean():.1f}")
    
    # Key insights
    print(f"\nKEY INSIGHTS")
    print("-"*70)
    
    if not seeding_df.empty:
        afc_acc = seeding_df[seeding_df['conference'] == 'AFC']['accuracy'].mean()
        nfc_acc = seeding_df[seeding_df['conference'] == 'NFC']['accuracy'].mean()
        if afc_acc < nfc_acc * 0.7:
            print("WARNING: AFC predictions significantly worse than NFC - investigate AFC-specific factors")
        elif nfc_acc < afc_acc * 0.7:
            print("WARNING: NFC predictions significantly worse than AFC - investigate NFC-specific factors")
        else:
            print("OK: Conference predictions are balanced")
    
    if not sb_df.empty:
        avg_prob = sb_df['predicted_prob'].mean()
        if avg_prob < 0.15:
            print("WARNING: Actual Super Bowl winners consistently have low predicted probabilities")
            print("   -> Model may be underrating playoff experience or recent form")
        elif avg_prob > 0.30:
            print("OK: Model is generally catching Super Bowl winners")
    
    brier = overall.get('avg_brier_score', 1.0)
    if brier > 0.25:
        print("WARNING: Brier score > 0.25 suggests probabilities are not well-calibrated")
        print("   -> Consider ensemble models or better feature engineering")
    elif brier < 0.20:
        print("OK: Brier score < 0.20 indicates well-calibrated probabilities")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS")
    print("-"*70)
    
    recommendations = []
    
    if not seeding_df.empty:
        if seeding_df['mae'].mean() > 1.5:
            recommendations.append("Implement full NFL tiebreaker rules (current MAE too high)")
        
        if seeding_df['accuracy'].mean() < 0.50:
            recommendations.append("Improve seeding accuracy - add strength of schedule adjustment")
    
    if not sb_df.empty:
        if sb_df['is_correct'].mean() < 0.50:
            recommendations.append("Add playoff experience features (teams with experience win more)")
        
        if sb_df['predicted_prob'].mean() < 0.20:
            recommendations.append("Improve Super Bowl predictions - consider recent form weighting")
    
    if brier > 0.22:
        recommendations.append("Use ensemble models to improve probability calibration")
    
    if not recommendations:
        recommendations.append("Model is performing well! Consider fine-tuning hyperparameters")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Save detailed analysis
    if output_path:
        analysis = {
            'overall_metrics': overall,
            'seeding_analysis': seeding_df.to_dict('records') if not seeding_df.empty else [],
            'super_bowl_analysis': sb_df.to_dict('records') if not sb_df.empty else [],
            'conference_analysis': conf_df.to_dict('records') if not conf_df.empty else [],
            'recommendations': recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nDetailed analysis saved to {output_path}")

def create_visualizations(results: dict, output_dir: str = 'output/visualizations'):
    """Create visualization charts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeding_df = analyze_seeding_accuracy(results)
    sb_df = analyze_super_bowl_accuracy(results)
    
    # Seeding accuracy by conference
    if not seeding_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        seeding_pivot = seeding_df.pivot(index='season', columns='conference', values='accuracy')
        seeding_pivot.plot(kind='bar', ax=ax, color=['#004C54', '#A71930'])
        ax.set_title('Seeding Accuracy by Conference and Season')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Season')
        ax.legend(title='Conference')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'seeding_accuracy_by_conference.png', dpi=300)
        print(f"Saved: {output_dir / 'seeding_accuracy_by_conference.png'}")
    
    # Super Bowl probability distribution
    if not sb_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sb_df['predicted_prob'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(sb_df['predicted_prob'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {sb_df["predicted_prob"].mean():.2%}')
        ax.set_title('Distribution of Predicted Probabilities for Actual Super Bowl Winners')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'super_bowl_probability_distribution.png', dpi=300)
        print(f"Saved: {output_dir / 'super_bowl_probability_distribution.png'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze validation results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to validation results JSON file')
    parser.add_argument('--output', type=str, default='output/analysis/validation_analysis.json',
                       help='Path to save analysis report')
    parser.add_argument('--viz', action='store_true',
                       help='Create visualization charts')
    
    args = parser.parse_args()
    
    # Load results
    results = load_validation_results(args.input)
    
    # Create analysis
    create_analysis_report(results, args.output)
    
    # Create visualizations
    if args.viz:
        create_visualizations(results)

if __name__ == '__main__':
    main()

