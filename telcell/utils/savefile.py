import csv
import os
import pathlib
from typing import Tuple, Mapping, Any, Callable, List, Union, Sequence, Optional

import lir.plotting
import numpy as np

from telcell.data.models import Track


def make_output_plots(lrs, y_true, output_dir: Union[pathlib.Path, str] = '.', ignore_missing_lrs: bool = False):
    """
    writes standard evaluation output to outputdir (cllr value, plots of lr distribution and pav). By default
    sets lrs not provided to 1, optionally drops these. The former makes sense when comparing systems that
    can provide LRs in different settings or when we expect all LRs to be provided.

    :param lrs: predicted lrs
    :param y_true: true values, of same length as lrs
    :param output_dir: dir to write true
    :param ignore_missing_lrs: if true, drop all LR=None, otherwise treats them as 1
    :return:
    """
    assert len(lrs) == len(y_true), 'there should be equal number of predicted lrs and ground truth'
    output_dir = pathlib.Path(output_dir or ".")
    # If no `output_dir` was specified, use the current working directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert the predicted LRs and ground-truth labels to numpy arrays
    # so that they are accepted by `lir.metrics` functions.
    lrs = np.array(lrs, dtype=float)
    y_true = np.array(y_true, dtype=int)

    if nr_nans := np.isnan(lrs).sum():
        # log any 'None' LRs (=method could not provide an LR)
        print(f'{nr_nans} LR values were not provided (=None)')
        if ignore_missing_lrs:
            print(f'Dropping LRs that are None')
            mask = ~np.isnan(lrs)

            # Use the mask to filter both arrays a and b
            lrs = lrs[mask]
            y_true = y_true[mask]
        else:
            print(f'Treating LRs that are None as 1')
            lrs = np.nan_to_num(lrs, nan=1)

    # Write metrics to disk.
    cllr = lir.metrics.cllr(lrs, y_true)
    with open(output_dir / "cllr.txt", "w") as f:
        f.write(f'cllr: {cllr:.3f}')

    # Save visualizations to disk.
    with lir.plotting.savefig(str(output_dir / "pav.png")) as ax:
        ax.pav(lrs, y_true)

    with lir.plotting.savefig(str(output_dir / 'lr_distribution.png')) as ax:
        ax.lr_histogram(lrs, y_true)

    with lir.plotting.savefig(str(output_dir / 'tippett.png')) as ax:
        ax.tippett(lrs, y_true)


def write_lrs(lrs: Sequence[Optional[float]], output_dir: pathlib.Path,
              track_pairs: List[Tuple[Track, Track, Mapping[str, Any]]],
              write_extra: Callable[[Mapping[str, Any]], Mapping[str, Any]]):
    """
    writes out all the lrs, and associated tracks and time slice

    :param lrs: list of predicted lrs
    :param output_dir: output dir
    :param track_pairs: list of pairs of tracks, with possible extra information per pair
    :param write_extra: function that will be called on the 'extra' mapping in the data to write specific information.

    it should return a list of column names and a list of strings of the same length per track pair.
    """
    # If no `output_dir` was specified, use the current working directory.
    output_dir = pathlib.Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    assert len(track_pairs) == len(
        lrs), f'Assuming we have information on all track pairs, but there are {len(lrs)} lrs and {len(track_pairs)} pairs'
    with (open(os.path.join(output_dir, 'lrs_per_track_pair.csv'), 'w', newline='') as file):
        writer = csv.writer(file)
        writer.writerow(['lr', 'track_a_owner', 'track_a_device', 'track_b_owner',
                         'track_b_device', 'date', 'track_a_timestamp', 'track_a_antenna_id',
                         'track_a_zipcode', 'track_a_city', 'track_b_timestamp',
                         'track_b_antenna_id', 'track_b_zipcode', 'track_b_city'])
        for lr, track_pair in zip(lrs, track_pairs):
            writer.writerow([lr, track_pair[0].owner, track_pair[0].device,
                             track_pair[1].owner, track_pair[1].device,
                             *write_extra(track_pair[2]).values()])
