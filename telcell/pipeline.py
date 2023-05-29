from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

import lir.plotting
import numpy as np

from telcell.data.models import Track, is_colocated
from telcell.models import Model


def run_pipeline(
        data: Iterable[Tuple[Track, Track, Mapping[str, Any]]],
        models: Iterable[Model],
        output_dir: Union[str, Path] = None,
) -> Sequence[Sequence[float]]:
    # If no `output_dir` was specified, use the current working directory.
    output_dir = Path(output_dir or ".")

    # Keep track of all likelihood ratios computed by each model.
    all_lrs = []

    # TODO: discussion about the following:
    # Do we want to iterate over `data` in the outer loop instead of iterating
    # over `models`? Iterating over `data` first would have 2 main advantages:
    #   1a. Any lazy preprocessing logic only has to be executed once, or
    #   1b. data won't have to be kept in memory;
    #   2. Allows for `data` to be an `Iterator` (can only be traversed once).
    # Some downsides would be:
    #   1. Requires all models to be kept in memory simultaneously, which
    #      could be problematic in case of multiple GPU models;
    #   2. No intermediate evaluation results for any particular model, you
    #      have to wait until all models have been applied to all data. This
    #      also means any errors that occur during evaluation are delayed.
    for model in models:
        lrs = []
        y_true = []
        for track_a, track_b, kwargs in data:
            lrs.append(model.predict_lr(track_a, track_b, **kwargs))
            y_true.append(is_colocated(track_a, track_b))

        # Convert the predicted LRs and ground-truth labels to numpy arrays
        # so that they are accepted by `lir.metrics` functions.
        lrs_array = np.array(lrs)
        y_true = np.array(y_true, dtype=int)

        # Write metrics to disk.
        cllr = lir.metrics.cllr(lrs_array, y_true)
        with open(output_dir / "lrs.txt", "w") as f:
            f.write(str(cllr))  # TODO: formatting

        # Save visualizations to disk.
        pav_file = str(output_dir / "pav.png")
        with lir.plotting.savefig(pav_file) as ax:
            ax.pav(lrs_array, y_true)

        all_lrs.append(lrs)
    return all_lrs
