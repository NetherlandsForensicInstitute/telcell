from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

import lir.plotting
import numpy as np

from telcell.data.models import Track
from telcell.models import Model


def is_colocated(track_a: Track, track_b: Track) -> bool:
    """
    Dummy implementation until actual implementation is merged.

    TODO: replace when https://github.com/NetherlandsForensicInstitute/telcell/pull/8 is merged.
    """
    import random
    return random.choice([True, False])


def run_pipeline(
        data: Iterable[Tuple[Track, Track, Mapping[str, Any]]],
        models: Iterable[Model],
        output_dir: Union[str, Path] = None,
) -> Sequence[Sequence[float]]:
    if not output_dir:
        output_dir = Path()

    # Keep track of all likelihood ratios computed by each model.
    all_lrs = []

    # TODO: discussion about the following:
    # Do we want to iterate over `data` in the outer loop? This has two main
    # advantages:
    #   1a. Any lazy preprocessing logic only has to be executed once, or
    #   1b. data won't have to be kept in memory;
    #   2. Allows for `data` to be an `Iterator` (can only be traversed once);
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

            # TODO: check if `track_a.owner` or `track_b.owner` is `None`.
            # TODO: import actual `is_colocated()` function once available.
            y_true.append(is_colocated(track_a, track_b))

        lrs_array = np.array(lrs)
        y_true = np.array(y_true, dtype=int)
        print(lir.metrics.cllr(lrs_array, y_true))  # TODO: write to disk
        with lir.plotting.show() as ax:
            ax.pav(lrs_array, y_true)  # TODO: savefig

        all_lrs.append(lrs)
    return all_lrs
