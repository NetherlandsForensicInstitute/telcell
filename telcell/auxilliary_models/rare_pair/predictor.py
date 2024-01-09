from typing import Mapping, Tuple, Sequence, Optional

from more_itertools import first

from telcell.auxilliary_models.rare_pair.utils import Bin
from telcell.data.models import Measurement
from telcell.auxilliary_models.rare_pair.coverage_model import CoverageModel
from telcell.auxilliary_models.geography import Grid, Area


class Predictor:
    """
    Calculates the probabilities for antennas given a specific coverage model.

    It is responsible for choosing the correct coverage model, given the provider (mnc) and timediff of the antennas,
    and similarly, it calls the correct calibrator to calibrate the scores returned from the coverage model.

    See README for more details.
    """

    def __init__(self,
                 models: Mapping[Tuple[int, Bin], CoverageModel]):
        self.models = models
        self._bins = list(sorted(set(bin for _, bin in models.keys())))

    def predict_probability_locations_given_antenna(self,
                                                    measurements: Sequence[Measurement],
                                                    reference_area: Area,
                                                    delta_t: float) -> Sequence[Optional[Grid]]:
        """
        For each of the (reference) antennas, calculate the (normalized) probabilities for this antenna at the
        locations inside `area`, and average them.

        :param measurements: the antenna registrations of the reference phone_id
        :param reference_area: the area for which we want to know the probability that the reference phone_id was there
        :param delta_t: the time difference between the registrations
        :returns: a probability grid for `grid`, or `None` if the area is outside the range of all `antennas`
        """
        bin = self.get_bin(delta_t)

        normalized_probabilities_for_reference_area = []
        for measurement in measurements:
            model = self.models[(measurement.extra['mnc'], bin)]
            measurement_area = model.measurement_area(measurement)
            if measurement_area.intersect(reference_area):
                normalized_probabilities_for_reference_area.append(
                    model.normalized_probabilities(measurement).move(reference_area.southwest))
            else:
                normalized_probabilities_for_reference_area.append(None)

        return normalized_probabilities_for_reference_area

    def predict_probability_antenna_given_locations(self,
                                                    measurement: Measurement,
                                                    delta_t: float) -> Grid:
        """
        Gets the probabilities for the query antenna from the coverage model. Returns a grid with probabilities.

        :param measurement: the  antenna registration
        :param delta_t: the time difference between the two antenna registrations
        :returns: the probability
        """
        model = self.models[(measurement.extra['mnc'], self.get_bin(delta_t))]
        return model.probabilities(measurement)

    def get_probability_e_h(self,
                            c_a: Measurement,
                            *c_b: Measurement,
                            delta_t: int = 0
                            ) -> float:
        """
        This function calculates the probabilities:
            * P(E|H1), for which `antennas` consists of a single antenna, and
            * P(E|H2), for which `antennas` consists of multiple antennas (the historic registrations).

        :param c_a: antenna registration for the known phone
        :param c_b: one or more antenna registrations for reference
        :param delta_t: time difference between registrations
        :returns: the probability
        """
        # The time difference is only applied to the reference antennas (under H1), for the query antenna it is always 0
        probabilities_c_a_given_locations = \
            self.predict_probability_antenna_given_locations(c_a, delta_t=0)

        # For the other `antennas` (the reference antennas), we only need the probabilities at the locations
        # that are inside the first grid
        probabilities_locations_given_c_b = \
            self.predict_probability_locations_given_antenna(c_b,
                                                             probabilities_c_a_given_locations.get_empty_grid(),
                                                             delta_t)
        return sum(
            (grid * probabilities_c_a_given_locations).sum() / len(c_b) if grid else 0
            for grid in probabilities_locations_given_c_b
        )

    def get_bin(self, delta_t: float) -> Bin:
        """
        Get corresponding bin for given delta_t
        :param delta_t: time difference in seconds
        :return: bin which covers the given time difference
        """
        return first(bin for bin in self._bins if bin[0] <= int(delta_t) <= bin[1])

    def cache_clear(self):
        """
        For each model used by this class, clear the lru caches.
        """
        for model in self.models.values():
            model.cache_clear()
