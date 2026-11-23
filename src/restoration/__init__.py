"""Historical photograph restoration with fine-tuned generative models (AAI-521, Group 3).

The package is organised by pipeline stage:

* :mod:`restoration.data` -- dataset acquisition, curation, splits and synthetic degradations.
* :mod:`restoration.models` -- pretrained model registry and wrappers for the four tasks.
* :mod:`restoration.utils` -- logging, seeding and HTTP helpers shared across stages.
"""

__version__ = "0.1.0"
