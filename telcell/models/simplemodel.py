from typing import List, Tuple, Optional, Mapping, Callable

import lir
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from telcell.data.models import Track
from telcell.models import Model
from telcell.utils.transform import get_switches, get_colocation_switches, generate_all_pairs, \
    get_pair_with_rarest_measurement_b


class MeasurementPairClassifier(Model):
    """
    Model that computes a likelihood ratio based on the distance between two
    antennas of a measurement pair. This pair is chosen based on the rarest
    location and for a certain time interval. The distances are scaled using a
    standard scaler. A logistic regression model is trained on colocated
    and dislocated pairs and a KDE and ELUB bounder is used to calibrate
    scores that are provided by the logistic regression.
    """

    def __init__(self, colocated_training_data: List[Track], categorize_measurement_for_rarity: Callable):
        self.training_data = colocated_training_data
        self.colocated_training_pairs = get_colocation_switches(self.training_data)
        self.categorize_measurement_for_rarity = categorize_measurement_for_rarity

    def predict_lr(self, track_a: Track, track_b: Track, **kwargs) -> Tuple[float, Optional[Mapping]]:
        pairs = get_switches(track_a, track_b)
        _, pair = get_pair_with_rarest_measurement_b(
            switches=pairs, history_track_b=kwargs['background_b'],
            categorize_measurement_for_rarity=self.categorize_measurement_for_rarity)

        # resulting pairs need not be really dislocated, but simulated
        # dislocation by temporally shifting track a's history towards the
        # timestamp of the singular measurement of track b
        dislocated_training_pairs = generate_all_pairs(pair.measurement_b, kwargs['background_b'])
        training_pairs = self.colocated_training_pairs + dislocated_training_pairs
        training_labels = [1] * len(self.colocated_training_pairs) + [0] * len(dislocated_training_pairs)

        # calculate for each pair the distance between the two antennas
        training_features = np.array(list(map(lambda x: x.distance, training_pairs))).reshape(-1, 1)
        comparison_features = np.array([pair.distance]).reshape(-1, 1)

        # scale the features
        scaler = StandardScaler()
        scaler.fit(training_features)
        training_features = scaler.transform(training_features)
        comparison_features = scaler.transform(comparison_features)

        estimator = LogisticRegression()
        calibrator = lir.ELUBbounder(lir.KDECalibrator(bandwidth=1.0))
        calibrated_scorer = lir.CalibratedScorer(estimator, calibrator)
        calibrated_scorer.fit(training_features, np.array(training_labels))

        return float(calibrated_scorer.predict_lr(comparison_features)), None
