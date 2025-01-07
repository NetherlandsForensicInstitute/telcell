from typing import Optional

import geopy
from pydantic import BaseModel, ConfigDict

from telcell.cell_identity import CellIdentity
from telcell.geography import Angle


class CellInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    cell: Optional[CellIdentity] = None
    wgs84: Optional[geopy.Point] = None
    azimuth: Optional[Angle] = None
    accuracy_m: Optional[float] = None
