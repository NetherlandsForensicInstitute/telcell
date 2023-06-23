from itertools import pairwise

from telcell.cell_identity import CellIdentity, RADIO_GSM, RADIO_UMTS, RADIO_LTE, RADIO_NR

CELL_IDENTITIES = [
    ("204-1", CellIdentity.create(mcc=204, mnc=1)),
    ("GSM/204-1", CellIdentity.create(radio=RADIO_GSM, mcc=204, mnc=1)),
    ("204-2", CellIdentity.create(mcc=204, mnc=2)),
    ("204-1-2000", CellIdentity.create(mcc=204, mnc=1, eci=2000)),
    ("204-1-2-2000", CellIdentity.create(mcc=204, mnc=1, lac=2, ci=2000)),
    ("GSM/204-1-2-2000", CellIdentity.create(radio=RADIO_GSM, mcc=204, mnc=1, lac=2, ci=2000)),
    ("UMTS/204-1-2-2000", CellIdentity.create(radio=RADIO_UMTS, mcc=204, mnc=1, lac=2, ci=2000)),
    ("LTE/204-1-2000", CellIdentity.create(radio=RADIO_LTE, mcc=204, mnc=1, eci=2000)),
    ("NR/204-1-2000", CellIdentity.create(radio=RADIO_NR, mcc=204, mnc=1, eci=2000)),
]


def test_operators(testdata_path):
    for spec, ci in CELL_IDENTITIES:
        assert CellIdentity.parse(spec) == ci
        assert hash(CellIdentity.parse(spec)) == hash(ci)

    for (spec1, ci1), (spec2, ci2) in pairwise(CELL_IDENTITIES):
        assert ci1 != ci2
        assert hash(ci1) != hash(ci2)
