"""
greensfield is a package for calculating magnetic field extrapolations from solar data using
Green's function methods.
"""
from .version import version as __version__

from greensfield.models import ObliqueSchmidtExtrapolator

__all__ = [
    'ObliqueSchmidtExtrapolator'
]
