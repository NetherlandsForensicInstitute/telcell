import warnings
from typing import Optional, Sequence

from telcell.cell_identity import CellIdentity
from telcell.celldb.models import CellInfo


def get_duplicate_policy(name: str):
    assert name is not None
    name = name.replace("-", "_")  # normalize name
    assert name in globals(), f"unknown duplicate policy: {name}"
    return globals()[name]


def exception(ci: CellIdentity, _results: Sequence[CellInfo]) -> Optional[CellInfo]:
    raise ValueError(f"duplicate cell id {ci} (not allowed by current policy)")


def warn(ci: CellIdentity, results: Sequence[CellInfo]) -> Optional[CellInfo]:
    warnings.warn(f"duplicate cell id {ci}")
    return results[0]


def take_first(
    _ci: CellIdentity, results: Sequence[CellInfo]
) -> Optional[CellInfo]:
    return results[0]


def drop(_ci: CellIdentity, _results: Sequence[CellInfo]) -> Optional[CellInfo]:
    return None
