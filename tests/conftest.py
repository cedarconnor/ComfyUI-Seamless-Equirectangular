"""Use the installed ComfyUI runtime for node integration tests."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
