"""
Models for field extrapolation
"""
import astropy.units as u
import numpy as np
import sunpy.map

from astropy.coordinates import SkyCoord
from sunpy.coordinates import (
    Heliocentric,
    HeliographicStonyhurst,
    Helioprojective,
    PlanarScreen,
)

__all__ = ['ExtrapolatorBase', 'ObliqueSchmidtExtrapolator']


class ExtrapolatorBase:

    def __init__(self, magnetogram, tangent_coord, corners, resample_factor=1):
        self.magnetogram = magnetogram
        self.tangent_coord = tangent_coord
        self.boundary_magnetogram = self.make_boundary_magnetogram(corners, resample_factor)

    @property
    def _planar_screen(self):
        return PlanarScreen(self.tangent_coord, distance_from_center=self.tangent_coord.radius)

    def make_boundary_magnetogram(self, corners, resample_factor):
        """
        Reproject, crop, and resample magnetogram to bottom boundary of a Cartesian box.

        Parameters
        ----------
        corners
        resample_factor
        """
        observer = HeliographicStonyhurst(lon=self.tangent_coord.lon,
                                          lat=self.tangent_coord.lat,
                                          radius=self.magnetogram.observer_coordinate.radius,
                                          obstime=self.magnetogram.reference_date)
        hpc_frame = Helioprojective(observer=observer,
                                    rsun=self.magnetogram.coordinate_frame.rsun)
        header = sunpy.map.make_fitswcs_header(
            self.magnetogram.data.shape,
            SkyCoord(Tx=0, Ty=0, unit='arcsec', frame=hpc_frame),
            scale=u.Quantity(self.magnetogram.scale),
            observatory=self.magnetogram.observatory,
            instrument=self.magnetogram.instrument,
            wavelength=self.magnetogram.wavelength,
            unit=self.magnetogram.unit,
        )
        with self._planar_screen:
            boundary_magnetogram = self.magnetogram.reproject_to(header)
        boundary_magnetogram = sunpy.map.Map(boundary_magnetogram.data, header)
        boundary_magnetogram = boundary_magnetogram.submap(corners[0], top_right=corners[1])
        boundary_magnetogram = boundary_magnetogram.resample(
            np.floor(resample_factor*u.Quantity(boundary_magnetogram.dimensions))
        )
        return boundary_magnetogram

    @property
    def hcc_frame(self):
        return Heliocentric(observer=self.boundary_magnetogram.observer_coordinate)

    def extrapolate(self):
        raise NotImplementedError

    def trace(self, seeds):
        """
        """
        ...


class ObliqueSchmidtExtrapolator(ExtrapolatorBase):
    """
    Extrapolate magnetic field using the oblique Schmidt method of Sakurai (1981).

    Parameters
    ----------
    """

    def extrapolate(self):
        ...
