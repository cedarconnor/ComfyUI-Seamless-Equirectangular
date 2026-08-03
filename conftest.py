import pathlib
import sys

# Ensure ComfyUI and repository roots are in sys.path for pytest
repo_dir = pathlib.Path(__file__).resolve().parent
comfy_dir = repo_dir.parents[1]

for path_str in (str(comfy_dir), str(repo_dir)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
