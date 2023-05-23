from typing import Optional

RADIO_GSM = "GSM"
RADIO_UMTS = "UMTS"
RADIO_LTE = "LTE"
RADIO_NR = "NR"

RADIO_TR = {"2G": RADIO_GSM, "3G": RADIO_UMTS, "4G": RADIO_LTE, "5G": RADIO_NR}


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
        radio = RADIO_TR.get(radio, radio)

        if radio == RADIO_GSM:
            return GSMCell(mcc, mnc, lac, ci)
        elif radio == RADIO_UMTS:
            return UMTSCell(mcc, mnc, lac, ci)
        elif radio == RADIO_LTE:
            return LTECell(mcc, mnc, eci)
        elif radio is not None:
            raise ValueError(f"unsupported radio technology: {radio}")
        elif eci is not None and (ci is None or lac is None):
            return LTECell(
                mcc, mnc, eci
            )  # insufficient info to identify GSM/UMTS --> guess LTE
        elif ci is not None and lac is not None and eci is None:
            return CellGlobalIdentity(
                mcc, mnc, lac, ci
            )  # insufficient info for LTE/NR --> guess GSM/UMTS
        elif eci is not None and ci is lac is not None or ci is not None:
            if ci > 0xFFFF:
                return LTECell(mcc, mnc, eci)  # invalid ci for GSM/UMTS --> guess LTE
            else:
                return CellGlobalIdentity(mcc, mnc, lac, ci)  # guess GSM/UMTS
        else:
            return CellIdentity(mcc, mnc)  # guess it's a cell

    @staticmethod
    def parse(global_identity, radio: str = None):
        elems = [int(e) for e in global_identity.split("-")]
        if len(elems) == 4:
            mcc, mnc, lac, ci = elems
            return CellIdentity.create(radio=radio, mcc=mcc, mnc=mnc, lac=lac, ci=ci)
        elif len(elems) == 3:
            mcc, mnc, eci = elems
            return CellIdentity.create(radio=radio, mcc=mcc, mnc=mnc, eci=eci)
        elif len(elems) == 2:
            return CellIdentity.create(radio=radio, mcc=elems[0], mnc=elems[1])
        else:
            raise ValueError(f"unrecognized cell: {global_identity}")

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
    def radio(self) -> str:
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
            and self.mcc == other.mcc
            and self.mnc == other.mnc
        )

    def __ne__(self, other):
        return not (self == other)

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

    @property
    def unique_identifier(self) -> str:
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

    def __init__(self, mcc: int, mnc: int, lac: int, ci: int):
        """
        For UMTS antennas, not the CI may be provided (e.g. in Android OS), but
        the "full CI". This is a concatenation of the Radio Network Controller
        (RNC, 12 bits) and CI (16 bits).
        """
        super().__init__(mcc, mnc, lac, ci & 0xFFFF)

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


class ECellGlobalIdentity(CellIdentity):
    __hash__ = CellIdentity.__hash__

    def __init__(self, mcc: int, mnc: int, eci: int):
        """
        E-UTRAN Cell Identifier (ECI). Used to identify a cell uniquely within
        a Public Land Mobile Network (PLMN). The ECI has a length of 28 bits
        and contains the eNodeB-IDentifier (eNB-ID). The ECI can address either
        1 or up to 256 cells per eNodeB, depending on the length of the eNB-ID.

        In short, it is used to identify a cell within a PLMN.

        In LTE networks, an antenna is identified by an eCGI, which consists of: * a PLMN code, * an evolved node b (eNB), and * a Cell Identifier (CI).

        The LAC counterpart of LTE antennas is the Tracking Area Code (TAC),
        which is a 16 bit code. I am unsure whether a CGI could be constructed
        for LTE antennas by using the PLMN, TAC and CI (TODO: please clarify!).

        Source: http://www.techtrained.com/the-ultimate-cheat-sheet-for-lte-identifiers/
        """
        super().__init__(mcc, mnc)
        if eci is not None and not isinstance(eci, int):
            raise ValueError("eci must be of type `int`")
        self.eci = eci

    @property
    def unique_identifier(self) -> str:
        return self.ecgi

    @property
    def ecgi(self) -> str:
        return f"{self.plmn}-{self.eci or '?'}"

    def is_complete(self) -> bool:
        return super().is_complete() and self.eci is not None

    def is_valid(self) -> bool:
        if not super().is_valid():
            return False
        if self.eci < 0x100 or self.eci > 0xFFFFFFF:
            return False
        return True

    def __eq__(self, other):
        return (
            isinstance(other, ECellGlobalIdentity)
            and super().__eq__(other)
            and self.radio == other.radio
            and self.eci == other.eci
        )

    def __repr__(self) -> str:
        return f"ECellGlobalIdentity({self.ecgi})"


class LTECell(ECellGlobalIdentity):
    __hash__ = ECellGlobalIdentity.__hash__

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


class NRCell(ECellGlobalIdentity):
    __hash__ = ECellGlobalIdentity.__hash__

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
