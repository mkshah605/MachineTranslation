from beamsearch import beam_search
from unittest.mock import Mock

FakeModel = Mock()

def test_beamsearch():
    """
    - model: beamsearch implementation
    - max_depth: we will consider at_most this many generations of a new token. If max_depth is reached, take top n beams
    - n_beams: the number of highest probabilities that we consider at each step

    """
    samples = beam_search(FakeModel, max_depth=10, n_beams=3)
    assert samples.shape[-1] == 3