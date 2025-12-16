"""
Entry point script for training offensive player models.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.attacker import get_data

if __name__ == "__main__":
    print("Training offensive player models (QB, RB, WR)...")
    models = get_data()
    print("\nModel training complete!")

