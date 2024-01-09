import pytest
from more_itertools import pairwise

from telcell.cell_identity import CellIdentity, Radio

CELL_IDENTITIES = [
    ("204-1", CellIdentity.create(mcc=204, mnc=1)),
    ("GSM/204-1", CellIdentity.create(radio=Radio.GSM, mcc=204, mnc=1)),
    ("204-2", CellIdentity.create(mcc=204, mnc=2)),
    ("204-1-2000", CellIdentity.create(mcc=204, mnc=1, eci=2000)),
    ("204-1-2-2000", CellIdentity.create(mcc=204, mnc=1, lac=2, ci=2000)),
    ("GSM/204-1-2-2000", CellIdentity.create(radio=Radio.GSM, mcc=204, mnc=1, lac=2, ci=2000)),
    ("UMTS/204-1-2-2000", CellIdentity.create(radio=Radio.UMTS, mcc=204, mnc=1, lac=2, ci=2000)),
    ("LTE/204-1-2000", CellIdentity.create(radio=Radio.LTE, mcc=204, mnc=1, eci=2000)),
    ("NR/204-1-2000", CellIdentity.create(radio=Radio.NR, mcc=204, mnc=1, eci=2000)),
    ("GSM/204-1-?-?", CellIdentity.create(radio=Radio.GSM, mcc=204, mnc=1)),
]


def test_operators():
    for spec, ci in CELL_IDENTITIES:
        assert CellIdentity.parse(spec) == ci
        assert CellIdentity.parse(str(ci)) == ci
        assert hash(CellIdentity.parse(spec)) == hash(ci)

    for (spec1, ci1), (spec2, ci2) in pairwise(CELL_IDENTITIES):
        assert ci1 != ci2
        assert hash(ci1) != hash(ci2)


def test_ci():
    with pytest.raises(ValueError):
        CellIdentity.parse("aardbei")
    with pytest.raises(ValueError):
        CellIdentity.parse("1-2-3-4-5")
    with pytest.raises(ValueError):
        CellIdentity.parse("NONE/1-2-3-4")
