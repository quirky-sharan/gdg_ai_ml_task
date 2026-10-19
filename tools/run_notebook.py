"""Execute a notebook top-to-bottom with the repository root as working directory.

Usage:  python tools/run_notebook.py [input.ipynb] [output.ipynb]
        (defaults: wafer_guard_fixed.ipynb, executed in place)
"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "wafer_guard_fixed.ipynb"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src

nb = nbformat.read(src, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, dst)
print(f"executed {src.name} -> {dst} in {time.time() - t0:.0f}s")
