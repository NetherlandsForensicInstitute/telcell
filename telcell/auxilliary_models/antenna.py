import logging
from datetime import datetime
from typing import Optional, Tuple

from telcell.auxilliary_models.geography import GridPoint
from telcell.cell_identity import CellIdentity

LOG = logging.getLogger(__name__)


class Antenna:
    """
    Antenna class, represents an antenna with attributes:
    - address: CellIdentity, the cell id
    - coords: GridPoint, the location of the antenna in rijksdriehoek coordinates
    - azimuth: the azimuth in degrees 0..360

    Optionally, the Antenna has some other attributes, which are directly passed to (and defined in) self.__init__().
    """

    def __init__(self, rdx: int, rdy: int, azimuth: float, ci: CellIdentity,
                 zipcode: str = None, city: str = None,
                 valid_from_to: Optional[Tuple[datetime.date, Optional[datetime.date]]] = None,
                 date: Optional[datetime.date] = None):
        """
        :param rdx: Rijksdriehoeks-x-coordinate
        :param rdy: Rijksdriehoeks-y-coordinate
        :param azimuth: the azimuth (angle) of the antenna
        :param ci: the cell identity
        :param zipcode: the antenna's zipcode
        :param city: the city where the antenna resides
        :param date: the date for which the antenna is active, formatted as 'YYYY-MM-DD' (Optional)
        :param valid_from_to: Period for which the antenna was active should be a tuple with dates formatted as
                            'YYYY-MM-DD'
        """
        self.address = ci
        self.coords = GridPoint(rdx, rdy)
        self.azimuth = azimuth
        self.zipcode = zipcode
        self.city = city
        self.date = date
        self.valid_from_to = valid_from_to
        # hard coded country code as we are only dealing with dutch cell records atm
        self._mcc = 204

    def __hash__(self):
        return hash((self.address, self.valid_from_to))

    def __eq__(self, other):
        return isinstance(other, Antenna) and self.address == other.address and self.date == other.date

    def __repr__(self) -> str:
        return f"{self.address}; coords={self.coords}; azi={self.azimuth}; zipcode={self.zipcode}; city={self.city};"
