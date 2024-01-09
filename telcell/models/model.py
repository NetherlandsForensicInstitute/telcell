from abc import ABC, abstractmethod
from typing import Tuple, Optional, Mapping

from telcell.data.models import Track


class Model(ABC):
    """
    Abstract base class for models that compute likelihood ratios for pairs of
    tracks (`track_a` and `track_b`) under the hypotheses that the tracks are
    colocated or dislocated.
    """

    @abstractmethod
    def predict_lr(
            self,
            track_a: Track,
            track_b: Track,
            **kwargs,
    ) -> Tuple[Optional[float], Optional[Mapping]]:
        """
        Computes a likelihood ratio for `track_a` and  `track_b`.

        Concrete implementations may specify additional `kwargs` that they
        require for computing the likelihood ratios, for example:

            - Additional measurements to context to `track_a` or `track_b`
            - Heatmap of cell tower usage/availability in a certain region
            - ...

        If such keyword arguments are included in the definition of this method
        in a subclass, a default value must be specified in order to satisfy
        the Liskov Substitution Principle (i.e. references to this class should
        be replaceable with references to a subclass without breaking code).

        :param track_a: The left track of the pair
        :param track_b: The right track of the pair
        :return: A likelihood ratio and an optional mapping with additional information
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
