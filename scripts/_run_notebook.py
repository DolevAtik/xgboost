"""Execute a notebook in place, streaming each cell's output as it completes.

    python scripts/_run_notebook.py <notebook.ipynb> [output.ipynb]

`jupyter nbconvert --execute` buffers everything until the run ends, which is unusable
for a notebook whose slowest cell takes two hours. This prints a header per cell and
flushes that cell's captured output as soon as it finishes, so a tail of the log shows
how far along the run is.
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

nb_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset_backblaze_Analysis.ipynb"
out_path = sys.argv[2] if len(sys.argv) > 2 else nb_path
timeout = int(os.environ.get("NB_TIMEOUT", "86400"))

nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                        allow_errors=False,
                        resources={"metadata": {"path": os.getcwd()}})

t0 = time.time()
n = len([c for c in nb.cells if c.cell_type == "code"])
print(f"executing {nb_path}: {n} code cells, timeout {timeout}s", flush=True)

with client.setup_kernel():
    idx = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        idx += 1
        t1 = time.time()
        head = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
        print(f"\n{'=' * 78}\n[cell {idx}/{n}] {head}\n{'=' * 78}", flush=True)
        try:
            client.execute_cell(cell, i)
        except Exception as exc:
            print(f"\n!! cell {idx} FAILED after {time.time() - t1:.1f}s: {exc}",
                  flush=True)
            nbformat.write(nb, out_path)
            raise
        for o in cell.get("outputs", []):
            if o.output_type == "stream":
                sys.stdout.write(o.text)
            elif o.output_type in ("execute_result", "display_data"):
                txt = o.get("data", {}).get("text/plain")
                if txt:
                    sys.stdout.write(str(txt)[:2000] + "\n")
        print(f"[cell {idx} done in {time.time() - t1:.1f}s]", flush=True)

nbformat.write(nb, out_path)
print(f"\nwrote {out_path} -- total {time.time() - t0:.1f}s", flush=True)
