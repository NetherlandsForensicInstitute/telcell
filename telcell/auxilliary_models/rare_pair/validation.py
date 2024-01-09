import itertools
import os
from collections import defaultdict
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from lir import to_odds
from lir.metrics import cllr
from lir.plotting import savefig
from lir.util import to_probability
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

from telcell.auxilliary_models.rare_pair.corrected_priors import CorrectedPriorsModel
from telcell.auxilliary_models.rare_pair.coverage_model import CoverageData, \
    ExtendedAngleDistanceClassificationCoverageModel, AngleDistanceClassificationCoverageModel
from telcell.auxilliary_models.rare_pair.transformers import BaseTransformer
from telcell.auxilliary_models.rare_pair.utils import Bin
from telcell.models.rare_pair_feature_based import RarePairModel


class GetReferenceDataCV:
    """
    This class internally splits the data and returns the train partition to the model when get_reference_data is
    provided as a callable for the coverage model. For validation the test partition belonging to the train partition
    that was used for training the model for a (timediff_bin, mnc) combination can be returned by calling the
    get_test_data method.
    """

    def __init__(self, coverage_data: Sequence[CoverageData], splits: int = 10, bins=Sequence[Bin],
                 transformer=BaseTransformer):
        """
        Class is initialized with a reference database and a number of splits to split the data for each
        (timediff_bin, mnc) combination.
        """
        self._kf = KFold(n_splits=splits)
        self.coverage_data = coverage_data
        self.transformer = transformer
        self._training_data = {}
        self._test_data = {}
        self.bins = bins
        self.max_distance = 28000

    def get_reference_data(self, split_nr: int, bin: Bin, mnc: int):
        filtered_coverage_data = np.array([c for c in self.coverage_data if
                                           c.positive_antenna.extra['mnc'] == mnc and c.get_bin(self.bins) == bin])
        key = (split_nr, bin, mnc)
        if key not in self._training_data:
            print(f'Getting data for key {key}')
            for i, (train_indices, test_indices) in enumerate(self._kf.split(filtered_coverage_data)):
                self._training_data[(i, bin, mnc)] = self._get_data_from_indices(filtered_coverage_data, train_indices)
                self._test_data[(i, bin, mnc)] = self._get_data_from_indices(filtered_coverage_data, test_indices)
        return self._training_data[key]

    def get_test_data(self, split_nr: int, bin: Bin, mnc: int):
        key = (split_nr, bin, mnc)
        return self._test_data[key]

    def _get_data_from_indices(self, filtered_coverage_data, indices):
        split_data = filtered_coverage_data[indices]
        features = self.transformer.get_features(measurements=split_data)
        mask = np.array(features[0])[:, 0] <= self.max_distance
        return np.array(features[0])[mask], np.array(features[1])[mask]


def get_scores_and_metrics_coverage_model(coverage_model: AngleDistanceClassificationCoverageModel, X_test, y_test,
                                          true_prior_odds):
    distances, angles = zip(*X_test)
    scores = coverage_model.predict(distances=tf.expand_dims(distances, 1), angles=tf.expand_dims(angles, 1))
    scores = coverage_model.calibrator.predict_proba(scores)[:, 1]

    # transform scores to incorporate prior of test data
    test_prior = np.sum(y_test) / np.sum(y_test == 0)
    corrected_scores = to_odds(scores) / true_prior_odds * test_prior

    lrs = corrected_scores / test_prior
    brier = brier_score_loss(y_true=y_test, y_prob=scores,
                             sample_weight=compute_sample_weight(class_weight='balanced', y=y_test))
    cllrs = cllr(lrs, y_test)

    return to_probability(corrected_scores), y_test, lrs, brier, cllrs


def validate_classification_coverage_model(model: RarePairModel, priors, out_folder):
    """
    Runs the validation script for the classification coverage models.

    Output:
        - Calibration curve (https://scikit-learn.org/stable/modules/calibration.html#calibration) are made and saved
            for each bin
        - Predicted probability distributions for each bin
        - Odds distributions for each bin
        - PAV figures for the odds for each bin
        - Boxplot with all Cllr values calculated over the odds
        - Boxplot with all Brier scores calculated over the probability
            This is measure is comparable to the Cllr applied to probabilities. Like the Cllr it can be decomposed
            in a calibration loss and a classification loss. (https://en.wikipedia.org/wiki/Brier_score)
    """
    NUM_SPLITS = 5

    # defaultdict to store results for each bin
    predicted_probs = defaultdict(list)
    lrs = defaultdict(list)
    true_labels = defaultdict(list)
    brier_scores = defaultdict(list)
    cllrs = defaultdict(list)
    n_positives = defaultdict(list)
    n_registrations = defaultdict(list)

    reference_data = GetReferenceDataCV(model.coverage_training_data, splits=NUM_SPLITS, bins=model.bins,
                                        transformer=model.transformer)

    # go over the models for the different bins and mnc
    combinations = list(itertools.product(range(NUM_SPLITS), model.predictor.models))
    for split_nr, (mnc, bin) in tqdm(combinations, desc='validating classification models'):
        # fit and calibrate coverage model for this split and model
        x_train, y_train = reference_data.get_reference_data(split_nr, bin, mnc)
        _model = CorrectedPriorsModel(model=LogisticRegression, penalty=None)
        _calibrator = CorrectedPriorsModel(IsotonicRegression, y_min=0, y_max=1, out_of_bounds='clip')
        coverage_model = \
            ExtendedAngleDistanceClassificationCoverageModel(outer_diameter=28000,
                                                             outer_resolution=500,
                                                             inner_diameter=4000,
                                                             inner_resolution=50,
                                                             model=_model,
                                                             calibrator=_calibrator)
        coverage_model.fit(x_train, y_train, x_cal=x_train, y_cal=y_train, **priors.get(mnc))
        # get the test data for the model that was trained for the (bin, mnc) combination
        X_test, y_test = reference_data.get_test_data(split_nr, bin, mnc)

        scores, y, lr, brier, cllr = get_scores_and_metrics_coverage_model(coverage_model, X_test, y_test,
                                                                           priors.get(mnc)["true_prior_odds"])

        # save relevant information and metrics for the figures
        predicted_probs[bin].append(scores)
        true_labels[bin].append(y)
        brier_scores[bin].append(brier)
        lrs[bin].append(lr)
        cllrs[bin].append(cllr)
        n_positives[bin].append(float(np.sum(y)))
        n_registrations[bin].append(len(y))

    for bin in model.bins:
        # write calibration curves
        prob_true, prob_pred = calibration_curve(np.concatenate(true_labels[bin]), np.concatenate(predicted_probs[bin]),
                                                 n_bins=25, strategy='quantile')

        plt.plot([min(prob_pred) * .90, max(prob_pred) * 1.02], [min(prob_pred) * .90, max(prob_pred) * 1.02],
                 linestyle='--')
        plt.plot(prob_pred, prob_true, marker='.')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(os.path.join(out_folder, f'calibration_curves_timediff_bin_{bin}.png'))
        plt.close()

        # create pav figure
        with savefig(os.path.join(out_folder, f'pav_figures_timediff_bin_{bin}.png')) as ax:
            ax.pav(np.concatenate(lrs[bin]), np.concatenate(true_labels[bin]))

        # create lr distribution
        with savefig(os.path.join(out_folder, f'lr_distributions_timediff_bin_{bin}.png')) as ax:
            ax.lr_histogram(np.concatenate(lrs[bin]), np.concatenate(true_labels[bin]))

        # create probability distribution
        with savefig(os.path.join(out_folder, f'probability_distributions_timediff_bin_{bin}.png')) as ax:
            _x, _y = np.concatenate(predicted_probs[bin]), np.concatenate(true_labels[bin])
            _bins, data, weights = extract_histogram_data(_x, _y)
            ax.hist(data, _bins, weights=weights, alpha=.25, label=['0', '1'])

        with open(os.path.join(out_folder, f'descriptives_timediff_bin_{bin}.yaml'), 'w') as yaml_file:
            descriptives = {'Positives': {'min': int(np.min(n_positives[bin])), 'max': int(np.max(n_positives[bin])),
                                          'mean': int(np.mean(n_positives[bin]))},
                            'Total': {'min': int(np.min(n_registrations[bin])),
                                      'max': int(np.max(n_registrations[bin])),
                                      'mean': int(np.mean(n_registrations[bin]))}}
            yaml.dump(descriptives, yaml_file, default_flow_style=False)

    def write_boxplot_scores(scores: list, y_label: str, figure_name: str):
        """
        Helper function to write the boxplots without to much code repetition.
        """
        _, ax = plt.subplots()
        sns.boxplot(data=scores, ax=ax)
        ax.set_xticklabels([str(bin) for bin in model.bins])
        plt.xlabel('timediff bins')
        plt.ylabel(y_label)
        plt.savefig(os.path.join(out_folder, figure_name))
        plt.close()

    # write the boxplots for the Cllrs and Brier scores
    write_boxplot_scores(list(cllrs.values()), 'cllr', 'boxplot_cllr')
    write_boxplot_scores(list(brier_scores.values()), 'brier score', 'boxplot_brier_score')


def extract_histogram_data(_x, _y):
    x1, x0 = _x[_y == 1], _x[_y == 0]
    x1_w = np.empty(x1.shape)
    x1_w.fill(1 / x1.shape[0])
    x0_w = np.empty(x0.shape)
    x0_w.fill(1 / x0.shape[0])
    bins = np.linspace(0, _x.max(), 30)
    return bins, [x0, x1], [x0_w, x1_w]
