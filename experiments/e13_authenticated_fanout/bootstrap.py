"""Locate the existing implementation or the archived source snapshot."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/core/agent_id_v2.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
