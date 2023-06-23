import re
from functools import cached_property
from typing import Optional, Any

RADIO_GSM = "GSM"
RADIO_UMTS = "UMTS"
RADIO_LTE = "LTE"
RADIO_NR = "NR"

CELL_IDENTITY_PATTERN = re.compile(
    r'^((?P<radio>[a-zA-Z]+)/)?(?P<mcc>[0-9]+)-(?P<mnc>[0-9]+)'
    + '(-((?P<lac>[0-9]+)-(?P<ci>[0-9]+)|(?P<eci>[0-9]+)))?$')


class CellIdentity:
    """
    An antenna is identified by a Cell Global Identity (CGI) or an E-UTRAN Cell
    Global Identity (eCGI), both 15 digit codes.

    Resouces:
        * https://www.cellmapper.net/
    """

    @staticmethod
    def create(
        radio: Optional[str] = None,
        mcc: Optional[int] = None,
        mnc: Optional[int] = None,
        lac: Optional[int] = None,
        ci: Optional[int] = None,
        eci: Optional[int] = None,
    ) -> "CellIdentity":
        radio = radio.upper() if radio is not None else None

        if radio == RADIO_GSM:
            return GSMCell(mcc, mnc, lac, ci)
        elif radio == RADIO_UMTS:
            return UMTSCell(mcc, mnc, lac, ci)
        elif radio == RADIO_LTE:
            return LTECell(mcc, mnc, eci)
        elif radio == RADIO_NR:
            return NRCell(mcc, mnc, eci)
        elif radio is not None:
            raise ValueError(f"unsupported radio technology: {radio}")
        elif eci is not None and (ci is None or lac is None):
            # we have `eci` but not `lac`, `ci` --> guess LTE/NR
            return EutranCellGlobalIdentity(mcc, mnc, eci)
        elif ci is not None and lac is not None and eci is None:
            # we have `lac` and `ci` but not `eci` --> guess GSM/UMTS
            return CellGlobalIdentity(mcc, mnc, lac, ci)
        elif eci is None and lac is None and ci is None:
            return CellIdentity(mcc, mnc)  # guess it's a cell
        else:
            raise ValueError(f"either `ci` or `eci` should be provided, but not both")

    @staticmethod
    def parse(spec: str) -> "CellIdentity":
        m = CELL_IDENTITY_PATTERN.match(spec)

        def convert_value(key: str, value: str) -> Any:
            if value is None or key == "radio":
                return value
            else:
                return int(value)

        groups = {key: convert_value(key, value) for key, value in m.groupdict().items()}
        return CellIdentity.create(**groups)

    def __init__(self, mcc: Optional[int], mnc: Optional[int]):
        """
        General info on mobile network cell identifiers:

            * [https://arimas.com/cgi-ecgi/]
            * [http://www.techtrained.com/the-ultimate-cheat-sheet-for-lte-identifiers/]

        Parameters:
            mcc: Mobile Country Code (MCC, 3 digits)
            mnc: Mobile Network Code (MNC, 2 digits)
        """
        assert mcc is None or isinstance(mcc, int), "mcc must be of type `int`"
        assert mnc is None or isinstance(mnc, int), "mnc must be of type `int`"
        self.mcc = mcc
        self.mnc = mnc

    @property
    def radio(self) -> Optional[str]:
        return None

    @property
    def plmn(self) -> str:
        """
        A mobile operator is uniquely identified by a Public Land Mobile
        Network (PLMN) code, a five digit code. The PLMN code is part of the
        subscribe identifier (IMSI) and the cell identifier (CGI or eCGI). The
        PLMN code consists of a Mobile Country Code (MCC, 3 digits), and a
        Mobile Network Code (MNC, 2 digits).
        """
        return f"{self.mcc or '?'}-{self.mnc or '?'}"

    @property
    def unique_identifier(self) -> str:
        return self.plmn

    def is_complete(self) -> bool:
        return self.mcc is not None and self.mnc is not None

    def is_valid(self) -> bool:
        return 0 < self.mcc < 1000 and 0 < self.mnc < 100

    def __eq__(self, other):
        return (
            isinstance(other, CellIdentity)
            and self.radio == other.radio
            and self.mcc == other.mcc
            and self.mnc == other.mnc
        )

    def __repr__(self) -> str:
        return f"CellIdentity({self.unique_identifier})"

    def __hash__(self):
        return hash(self.unique_identifier)


class CellGlobalIdentity(CellIdentity):
    """
    The LAC is 16 bits (or 5 digits). The CI in GSM and CDMA/UMTS networks is
    16 bits (or 5 digits).
    """

    __hash__ = CellIdentity.__hash__

    def __init__(self, mcc: int, mnc: int, lac: int, ci: int):
        super().__init__(mcc, mnc)
        assert lac is None or isinstance(lac, int), "lac must be of type `int`"
        assert ci is None or isinstance(ci, int), "ci must be of type `int`"
        self.lac = lac
        self.ci = ci

    @cached_property
    def unique_identifier(self) -> str:
        if self.radio is not None:
            return f"{self.radio}/{self.cgi}"
        else:
            return self.cgi

    @property
    def cgi(self) -> str:
        """
        The CGI is used in GSM or
        CDMA/UMTS networks and consists of:
            * a PLMN code,
            * a Location Area Code (LAC), and
            * a Cell Identifier (CI).
        """
        return f"{self.plmn}-{self.lac or '?'}-{self.ci or '?'}"

    def is_complete(self) -> bool:
        return super().is_complete() and self.lac is not None and self.ci is not None

    def is_valid(self) -> bool:
        if not super().is_valid():
            return False
        if self.lac < 0 or self.lac > 0xFFFF:
            return False
        if not 0 <= self.ci <= 0xFFFF:
            return False
        return True

    def __eq__(self, other):
        return (
            isinstance(other, CellGlobalIdentity)
            and super().__eq__(other)
            and self.lac == other.lac
            and self.ci == other.ci
        )

    def __repr__(self) -> str:
        return f"CellGlobalIdentity({self.cgi})"


class GSMCell(CellGlobalIdentity):
    __hash__ = CellGlobalIdentity.__hash__

    @property
    def radio(self) -> str:
        return RADIO_GSM

    def _asdict(self) -> dict:
        return {
            "radio": RADIO_GSM,
            "cgi": (self.mcc, self.mnc, self.lac, self.ci),
        }

    def __repr__(self):
        return f"GSMCell({self.cgi})"


class UMTSCell(CellGlobalIdentity):
    __hash__ = CellGlobalIdentity.__hash__

    def __init__(self, mcc: int, mnc: int, tac: int, ci: int):
        """
        For UMTS antennas, not the CI may be provided (e.g. in Android OS), but
        the "full CI". This is a concatenation of the Radio Network Controller
        (RNC) and CI.
        """
        super().__init__(mcc, mnc, tac, ci)
        self.rnc = ci >> 16 if ci is not None else None

    @property
    def radio(self) -> str:
        return RADIO_UMTS

    def _asdict(self) -> dict:
        return {
            "radio": RADIO_UMTS,
            "cgi": (self.mcc, self.mnc, self.lac, self.ci),
        }

    def __repr__(self) -> str:
        return f"UMTSCell({self.cgi})"


class EutranCellGlobalIdentity(CellIdentity):
    __hash__ = CellIdentity.__hash__

    def __init__(self, mcc: int, mnc: int, eci: int):
        """
        E-UTRAN Cell Identifier (ECI). Used to identify a cell uniquely within
        a Public Land Mobile Network (PLMN). The ECI has a length of 28 bits
        and contains the eNodeB-IDentifier (eNB-ID). The ECI can address either
        1 or up to 256 cells per eNodeB, depending on the length of the eNB-ID.

        In short, it is used to identify a cell within a PLMN.

        In LTE and NR networks, an antenna is identified by an eCGI, which consists
        of: * a PLMN code, * an evolved node b (eNB), and * a Cell Identifier (CI).

        The LAC counterpart of LTE antennas is the Tracking Area Code (TAC),
        which is a 16 bit code, but this is not required to identify a cell.

        Source: http://www.techtrained.com/the-ultimate-cheat-sheet-for-lte-identifiers/
        """
        super().__init__(mcc, mnc)
        if eci is not None and not isinstance(eci, int):
            raise ValueError("eci must be of type `int`")
        self.eci = eci

    @cached_property
    def unique_identifier(self) -> str:
        if self.radio is not None:
            return f"{self.radio}/{self.ecgi}"
        else:
            return self.ecgi

    @property
    def ecgi(self) -> str:
        return f"{self.plmn}-{self.eci or '?'}"

    def is_complete(self) -> bool:
        return super().is_complete() and self.eci is not None

    def is_valid(self) -> bool:
        if not super().is_valid():
            return False
        if self.eci < 0x100:
            return False  # enodeb is missing
        if self.eci > 0xFFFFFFF:
            return False  # outside range
        return True

    def __eq__(self, other):
        return (
            isinstance(other, EutranCellGlobalIdentity)
            and super().__eq__(other)
            and self.eci == other.eci
        )


class LTECell(EutranCellGlobalIdentity):
    __hash__ = EutranCellGlobalIdentity.__hash__

    @property
    def radio(self) -> str:
        return RADIO_LTE

    def _asdict(self) -> dict:
        return {
            "radio": RADIO_LTE,
            "ecgi": (self.mcc, self.mnc, self.eci),
        }

    def __repr__(self):
        return f"LTECell({self.ecgi})"


class NRCell(EutranCellGlobalIdentity):
    __hash__ = EutranCellGlobalIdentity.__hash__

    @property
    def radio(self) -> str:
        return RADIO_NR

    def _asdict(self) -> dict:
        return {
            "radio": RADIO_NR,
            "ecgi": (self.mcc, self.mnc, self.eci),
        }

    def __repr__(self):
        return f"NRCell({self.ecgi})"
