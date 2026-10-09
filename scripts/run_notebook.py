"""Execute a notebook in place, with this interpreter's packages and the notebook's own folder as the
working directory, so its saved outputs, tables and figures are regenerated.

    python scripts/run_notebook.py path/to/notebook.ipynb [--timeout SECONDS]

The notebook is only rewritten when every cell ran; a failed run leaves it untouched. Formatting of
the .ipynb (indent) is kept so that git diffs show the outputs and nothing else.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def detect_indent(path):
    """Indent of the notebook file as it is on disk (nbformat always writes 1)."""
    with open(path, encoding="utf-8") as f:
        f.readline()
        second = f.readline()
    return len(second) - len(second.lstrip(" ")) or 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--timeout", type=int, default=None, help="seconds per cell (default: no limit)")
    args = parser.parse_args()

    path = Path(args.notebook).resolve()
    indent = detect_indent(path)
    notebook = nbformat.read(path, as_version=4)

    # A kernel that runs this very interpreter (the default "python3" kernel may be another one).
    kernel_dir = Path(tempfile.mkdtemp()) / "kernels" / "polartox-run"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "polartox-run", "language": "python",
    }), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(kernel_dir.parent.parent)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    client = NotebookClient(notebook, kernel_name="polartox-run", timeout=args.timeout,
                            resources={"metadata": {"path": str(path.parent)}}, allow_errors=False)
    try:
        client.execute()
    except CellExecutionError as error:
        print(f"FAILED: {path.name}\n{error}", file=sys.stderr)
        return 1

    text = json.dumps(json.loads(nbformat.writes(notebook)), indent=indent, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"OK: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
