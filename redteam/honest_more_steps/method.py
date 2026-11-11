"""honest_more_steps: actual candidate artifact producer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import produce
produce(Path(__file__).resolve().parent)
