import random

from telcell.data.models import Track


class BasicModel:
    """
    Basic model to predict the LR of two tracks belonging to the same instance.
    Other models can use this as a template for more sophisticated methods.
    """

    @staticmethod
    def predict(track_a: Track, track_b: Track, background_info) -> float:
        """
        Gives random LR that two tracks belong to the same instance.
        Returns 1 LR for two tracks.
        """
        return random.random()
