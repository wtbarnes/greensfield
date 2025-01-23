"""
Utility functions for field extrapolations
"""

__all__ = ['make_boundary_magnetogram']

import astropy.units as u
import numpy as np
import sunpy.map

from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, PlanarScreen


def make_boundary_magnetogram(magnetogram, tangent_coord, corners, resample_factor):
        """
        Reproject, crop, and resample magnetogram to bottom boundary of a Cartesian box.

        Parameters
        ----------
        corners
        resample_factor
        """
        observer = HeliographicStonyhurst(lon=tangent_coord.lon,
                                          lat=tangent_coord.lat,
                                          radius=magnetogram.observer_coordinate.radius,
                                          obstime=magnetogram.reference_date)
        hpc_frame = Helioprojective(observer=observer, rsun=magnetogram.coordinate_frame.rsun)
        header = sunpy.map.make_fitswcs_header(
            magnetogram.data.shape,
            SkyCoord(Tx=0, Ty=0, unit='arcsec', frame=hpc_frame),
            scale=u.Quantity(magnetogram.scale),
            observatory=magnetogram.observatory,
            instrument=magnetogram.instrument,
            wavelength=magnetogram.wavelength,
            unit=magnetogram.unit,
        )
        with PlanarScreen(tangent_coord, distance_from_center=tangent_coord.radius, only_off_disk=False):
            boundary_magnetogram = magnetogram.reproject_to(header)
        boundary_magnetogram = sunpy.map.Map(boundary_magnetogram.data, header)
        boundary_magnetogram = boundary_magnetogram.submap(corners[0], top_right=corners[1])
        boundary_magnetogram = boundary_magnetogram.resample(
            np.floor(resample_factor*u.Quantity(boundary_magnetogram.dimensions))
        )
        return boundary_magnetogram
