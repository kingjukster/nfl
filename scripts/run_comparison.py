"""
Entry point script for comparing predictions with live NFL stats.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.comparison.run_comparison import main

if __name__ == "__main__":
    main()
