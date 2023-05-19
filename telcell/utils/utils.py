from telcell.data.models import Track


def is_colocated(track_a: Track, track_b: Track) -> bool:
    """Checks if two tracks are colocated to each other."""
    return track_a.owner == track_b.owner
