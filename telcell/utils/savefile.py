import lir.plotting
import numpy as np
import pathlib


def make_output_plots(lrs, y_true, output_dir):
    # If no `output_dir` was specified, use the current working directory.
    output_dir = pathlib.Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert the predicted LRs and ground-truth labels to numpy arrays
    # so that they are accepted by `lir.metrics` functions.
    lrs_array = np.array(lrs)
    y_true = np.array(y_true, dtype=int)

    # Write metrics to disk.
    cllr = lir.metrics.cllr(lrs_array, y_true)
    with open(output_dir / "cllr.txt", "w") as f:
        f.write(str(cllr))  # TODO: formatting

    # Save visualizations to disk.
    pav_file = str(output_dir / "pav.png")
    with lir.plotting.savefig(pav_file) as ax:
        ax.pav(lrs_array, y_true)
