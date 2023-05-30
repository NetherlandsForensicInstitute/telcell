"""
Import model classes from submodules into the `telcell.models` namespace for
convenient and consistent access. In the future we may want to build an
auto-loader to do this for us.
"""
from telcell.models.dummy import DummyModel
from telcell.models.model import Model
