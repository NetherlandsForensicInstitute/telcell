"""
A minimal implementation of

    Bosma, W., Dalm, S., van Eijk, E., El Harchaoui, R., Rijgersberg, E., Tops, H. T., ... & Ypma, R. (2020).
    Establishing phone-pair co-usage by comparing mobility patterns. Science & Justice, 60(2), 180-190.
    https://arxiv.org/pdf/2104.11683.pdf

Briefly summarized:

- A fully supervized model learns to distinguish pairs of tracks of same-user devices from pairs of different-user
  devices from training on sample datasets of both
- Tracks are split into days; LRs are calculated per day TODO is dit waar?
- The model operates on _switches_ in the track pair: "measurement pairs originating from both tracks which are adjacent
  in time".
- The machine learning is a three step process:
    1. A switch-based logistic regression model is trained on switches from same-user and different-user measurement
    pairs, resulting in a score between 0 and 1 for each switch.
    2. The (variable number of) individual switches scores of a track pair are aggregated into normalized frequencies
    of bins with a range of 0.1 points each. A track-pair-based logistic regression model is then trained on the binned
    scores, again on same-user and different-user track pairs.
    3. The scores of the track-pair-based model are calibrated into Likelihood Ratios by using a KDE with a Gaussian
    kernel.
- Of course, when using a machine learning model that can handle a varying number of samples, such as an RNN or a
  Transformer, steps 1 and 2 could be combined into a single step. For this paper the two-step logistic regression was
  used to offer interpretable (sub)-models for use in court.

"""


