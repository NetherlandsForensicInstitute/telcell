import logging


# silence logging for namespace "telcell" (and any submodules)
logging.getLogger(__name__).addHandler(logging.NullHandler())
