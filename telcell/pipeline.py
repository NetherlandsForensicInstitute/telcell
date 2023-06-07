from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

import lir.plotting
import numpy as np

from telcell.data.models import Track, is_colocated
from telcell.models import Model


def run_pipeline(
        data: Iterable[Tuple[Track, Track, Mapping[str, Any]]],
        model: Model,
        output_dir: Union[str, Path] = None,
) -> Sequence[float]:
    # If no `output_dir` was specified, use the current working directory.
    output_dir = Path(output_dir or ".")

    lrs = []
    y_true = []
    for track_a, track_b, kwargs in data:
        lrs.append(model.predict_lr(track_a, track_b, **kwargs))
        y_true.append(is_colocated(track_a, track_b))

    # Convert the predicted LRs and ground-truth labels to numpy arrays
    # so that they are accepted by `lir.metrics` functions.
    lrs_array = np.array(lrs)
    y_true = np.array(y_true, dtype=int)

    # Use the model name in the output filenames.
    model_name = model.__class__.__name__

    # Write metrics to disk.
    cllr = lir.metrics.cllr(lrs_array, y_true)
    with open(output_dir / f"{model_name}_lrs.txt", "w") as f:
        f.write(str(cllr))  # TODO: formatting

    # Save visualizations to disk.
    pav_file = str(output_dir / f"{model_name}_pav.png")
    with lir.plotting.savefig(pav_file) as ax:
        ax.pav(lrs_array, y_true)

    return lrs
