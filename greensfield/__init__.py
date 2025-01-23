"""
greensfield is a package for calculating magnetic field extrapolations from solar data using
Green's function methods.
"""
from greensfield.models import ObliqueSchmidtExtrapolator

from .version import version as __version__

__all__ = [
    'ObliqueSchmidtExtrapolator'
]
