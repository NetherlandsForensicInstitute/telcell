from typing import Optional, Mapping, Tuple

from telcell.data.models import Track
from telcell.models.model import Model


class DummyModel(Model):
    """
    Dummy model that always predicts a likelihood ratio of 1.0 for any two
    tracks, regardless of their measurements or background information.
    """

    def predict_lr(
            self,
            track_a: Track,
            track_b: Track,
            **kwargs,
    ) -> Tuple[float, Optional[Mapping]]:
        return 1.0, None
