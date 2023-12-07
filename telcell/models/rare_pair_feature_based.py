from itertools import groupby
from typing import Sequence, Mapping, Optional, List, Any, Tuple, Callable

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from telcell.auxilliary_models.rare_pair.corrected_priors import CorrectedPriorsModel
from telcell.auxilliary_models.rare_pair.coverage_model import CoverageData, \
    ExtendedAngleDistanceClassificationCoverageModel
from telcell.auxilliary_models.rare_pair.predictor import Predictor
from telcell.auxilliary_models.rare_pair.transformers import BaseTransformer
from telcell.auxilliary_models.rare_pair.utils import Bin
from telcell.data.models import Track
from telcell.models import Model
from telcell.utils.transform import get_switches, get_pair_with_rarest_measurement_b


class RarePairModel(Model):
    """
    Model that computes a likelihood ratio on a single switch (=pair of consecutive measurements from different
    devices) that is selected based on the rarity of the location of one of the measurements. The model assesses
    probabilities of connecting to a cell from a location from a separate coverage dataset, which consists of many
    cells that were and were not connected to from given locations. A transformer should be supplied to convert
    these connections to features, such as angle and distance. Coverage models are trained per provider and time
    difference bin. The likelihood ratio is computed as the ratio of the likelihood of connecting to cell A given
    you just connect to cell B from the same location and the likelihood of being in a location that connects to
    cell A (given your history).
    """

    def __init__(self, coverage_training_data: Sequence[CoverageData], transformer: BaseTransformer, bins: List[Bin],
                 categorize_measurement_for_rarity: Callable,
                 priors: Mapping, fit_models: bool = True):
        self.bins = bins
        self.categorize_measurement_for_rarity = categorize_measurement_for_rarity
        self.transformer = transformer
        self.coverage_training_data = self.filter_timediff(coverage_training_data)
        self.fit_models = fit_models
        self.max_delay = self.bins[-1][1]
        self.predictor = Predictor(self.fit_coverage_models(transformer, priors))

    @staticmethod
    def filter_track(track: Track, filter: Mapping) -> Track:
        measurements = track.measurements
        if 'mnc' in filter:
            measurements = [m for m in measurements if m.extra['mnc'] in filter['mnc']]
        return Track(measurements=measurements, device=track.device, owner=track.owner)

    def predict_lr(self, track_a: Track, track_b: Track, **kwargs) \
            -> Tuple[Optional[float], Optional[Mapping[str, Any]]]:
        track_a = self.filter_track(track_a, filter=kwargs['filter'])
        track_b = self.filter_track(track_b, filter=kwargs['filter'])
        if not track_a or not track_b:
            return None, None
        switches = get_switches(track_a, track_b)
        _, rarest_pair = get_pair_with_rarest_measurement_b(
            switches=switches, history_track_b=kwargs['background_b'],
            categorize_measurement_for_rarity=self.categorize_measurement_for_rarity,
            max_delay=self.max_delay)

        if rarest_pair is None:
            return None, None

        h_1 = self.predictor.get_probability_e_h(rarest_pair.measurement_a, rarest_pair.measurement_b,
                                                 delta_t=rarest_pair.time_difference.total_seconds())
        r_a = kwargs['background_a'].measurements
        h_2 = self.predictor.get_probability_e_h(rarest_pair.measurement_a, *(r_a + [rarest_pair.measurement_a]),
                                                 delta_t=rarest_pair.time_difference.total_seconds())

        lr = h_1 / h_2

        # clip LR to maximum of dataset size
        lr = min(lr, len(r_a))
        lr = max(lr, 1 / len(r_a))
        return lr, {'measurement_a': rarest_pair.measurement_a,
                    'measurement_b': rarest_pair.measurement_b}

    def fit_coverage_models(self, transformer, priors):
        models = {}

        for (mnc, bin), group in groupby(sorted(self.coverage_training_data,
                                                key=lambda x: (x.positive_antenna.extra['mnc'], x.get_bin(self.bins))),
                                         lambda x: (x.positive_antenna.extra['mnc'], x.get_bin(self.bins))):
            _model = CorrectedPriorsModel(model=LogisticRegression, penalty=None)
            _calibrator = CorrectedPriorsModel(IsotonicRegression, y_min=0, y_max=1, out_of_bounds='clip')
            model = ExtendedAngleDistanceClassificationCoverageModel(outer_diameter=28000, outer_resolution=500,
                                                                     inner_diameter=4000, inner_resolution=50,
                                                                     model=_model,
                                                                     calibrator=_calibrator)

            if self.fit_models:
                x, y = transformer.get_features(group)
                model.fit(x, y, x_cal=x, y_cal=y, **priors.get(mnc))

            models[(mnc, bin)] = model

        if self.fit_models:
            self.coverage_training_data = None

        return models

    def filter_timediff(self, coverage_training_data: Sequence[CoverageData]) -> List[CoverageData]:
        return [c for c in coverage_training_data if  # add extra bin to make sure the delta t falls in a bin
                c.get_bin(self.bins) is not None]
