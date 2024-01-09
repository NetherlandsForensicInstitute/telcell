from typing import Any, Optional, Tuple

Bin = Tuple[int, int]
DISTANCE_STEP = 10


def round_to_nearest(x: Any, base: Optional[int] = DISTANCE_STEP):
    """
    Round a number or array to the nearest multiple of 'base'. E.g. for base=10, x=14 becomes x=10.
    """
    return base * (x // base)
