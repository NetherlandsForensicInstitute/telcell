from telcell.data.models import Track


class BasicModel:
    """
    Basic model to predict the LR of two tracks belonging to the same instance.
    Other models can use this as a template for more sophisticated methods.
    """

    def predict(self, track_a: Track, track_b: Track, background_info=None) -> float:
        """
        Gives LR that two tracks belong to the same instance.
        Returns 1 LR for two tracks that is alwoys 1.
        """
        return 1.0
