"""
Seeding validation functions.

Validates playoff seeding structures.
"""

from typing import Dict, List, Any
from src.models.seeding import ConferenceSeeding, PlayoffSeed


def validate_seeding(seeding: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Validate playoff seeding structure.
    
    Expected format:
    {
        'AFC': [
            {'team': str, 'seed': int, 'win_pct': float, ...},
            ...
        ],
        'NFC': [...]
    }
    
    Validation rules:
    - 7 unique seeds per conference (1-7)
    - 7 unique team names per conference
    - All seeds in range 1-7
    - No duplicate teams or seeds
    """
    if not isinstance(seeding, dict):
        raise ValueError("Seeding must be a dictionary")
    
    for conf in ['AFC', 'NFC']:
        if conf not in seeding:
            raise ValueError(f"Missing conference: {conf}")
        
        conf_seeding = seeding[conf]
        if not isinstance(conf_seeding, list):
            raise ValueError(f"{conf} seeding must be a list")
        
        if len(conf_seeding) != 7:
            raise ValueError(f"{conf}: Expected 7 seeds, got {len(conf_seeding)}")
        
        # Collect seeds and teams
        seeds = []
        teams = []
        
        for i, seed_data in enumerate(conf_seeding):
            if not isinstance(seed_data, dict):
                raise ValueError(f"{conf} seed[{i}] must be a dictionary")
            
            # Check required fields
            if 'team' not in seed_data:
                raise ValueError(f"{conf} seed[{i}]: Missing 'team' field")
            if 'seed' not in seed_data:
                raise ValueError(f"{conf} seed[{i}]: Missing 'seed' field")
            
            team = seed_data['team']
            seed = seed_data['seed']
            
            # Validate seed range
            if not isinstance(seed, int) or not (1 <= seed <= 7):
                raise ValueError(f"{conf} seed[{i}]: Invalid seed {seed} (must be 1-7)")
            
            seeds.append(seed)
            teams.append(team)
        
        # Check for unique seeds
        if len(set(seeds)) != 7:
            duplicates = [s for s in seeds if seeds.count(s) > 1]
            raise ValueError(f"{conf}: Duplicate seeds found: {duplicates}")
        
        # Check for unique teams
        if len(set(teams)) != 7:
            duplicates = [t for t in teams if teams.count(t) > 1]
            raise ValueError(f"{conf}: Duplicate teams found: {duplicates}")
        
        # Check all seeds 1-7 are present
        expected_seeds = set(range(1, 8))
        actual_seeds = set(seeds)
        if expected_seeds != actual_seeds:
            missing = expected_seeds - actual_seeds
            extra = actual_seeds - expected_seeds
            raise ValueError(f"{conf}: Seed mismatch. Missing: {missing}, Extra: {extra}")

