from typing import Any, Iterable, Mapping, Tuple

from telcell.data.models import Track, is_colocated
from telcell.models import Model


def run_pipeline(
        data: Iterable[Tuple[Track, Track, Mapping[str, Any]]],
        model: Model,
        **kwargs
) -> tuple[list[float], list[bool]]:

    lrs = []
    y_true = []
    for track_a, track_b, kwargs in data:
        lrs.append(model.predict_lr(track_a, track_b, **kwargs))
        y_true.append(is_colocated(track_a, track_b))

    return lrs, y_true
