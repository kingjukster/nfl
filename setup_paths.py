"""
Setup script to ensure all paths are correct after reorganization.
This script updates file paths in scripts to match the new directory structure.
"""
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print("Path setup complete. Project root:", project_root)
print("Python path includes:")
for p in sys.path[:3]:
    print(f"  - {p}")

