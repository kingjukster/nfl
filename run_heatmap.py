"""
Entry point script for generating win probability heatmap.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import and run heatMap2
import importlib.util
heatmap_path = project_root / "src" / "heatMap2.py"
spec = importlib.util.spec_from_file_location("heatMap2", heatmap_path)
heatmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heatmap)

