"""
Entry point script for training defensive player models.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import and run MLNFL
import importlib.util
mlnfl_path = project_root / "src" / "simulation" / "defensive.py"
spec = importlib.util.spec_from_file_location("MLNFL", mlnfl_path)
mlnfl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlnfl)

