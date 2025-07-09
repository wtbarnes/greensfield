"""
Utility functions for field extrapolations
"""
import astropy.units as u
import numpy as np
import shapely
import sunpy.map

from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, PlanarScreen

__all__ = [
    'make_boundary_magnetogram',
    'get_coordinates_above_threshold',
    'find_fieldlines_intersecting_loop',
    'find_fieldlines_within_loop_footprint',
]


def make_boundary_magnetogram(magnetogram, tangent_coord, corners, resample_factor, **reproject_kwargs):
    """
    Reproject and crop magnetogram to bottom boundary of a Cartesian box.

    Parameters
    ----------
    magnetogram: `~sunpy.map.GenericMap`
    tangent_coord: `~astropy.coordinates.SkyCoord`
    corners: array-like of `~astropy.coordinates.SkyCoord`
    resample_factor: `float`
    reproject_kwargs: `dict`
    """
    observer = HeliographicStonyhurst(lon=tangent_coord.lon,
                                      lat=tangent_coord.lat,
                                      radius=magnetogram.observer_coordinate.radius,
                                      obstime=magnetogram.reference_date)
    hpc_frame = Helioprojective(observer=observer, rsun=magnetogram.coordinate_frame.rsun)
    header = sunpy.map.make_fitswcs_header(
        magnetogram.data.shape,
        SkyCoord(Tx=0, Ty=0, unit='arcsec', frame=hpc_frame),
        scale=u.Quantity(magnetogram.scale)/resample_factor,
        observatory=magnetogram.observatory,
        instrument=magnetogram.instrument,
        wavelength=magnetogram.wavelength,
        unit=magnetogram.unit,
    )
    with PlanarScreen(tangent_coord, distance_from_center=tangent_coord.radius, only_off_disk=False):
        boundary_magnetogram = magnetogram.reproject_to(header, **reproject_kwargs)
    boundary_magnetogram = sunpy.map.Map(boundary_magnetogram.data, header)
    boundary_magnetogram = boundary_magnetogram.submap(corners[0], top_right=corners[1])
    return boundary_magnetogram


def get_coordinates_above_threshold(smap, threshold):
    """
    Return world coordinates of pixels above a threshold

    Parameters
    ----------
    smap: `~sunpy.map.GenericMap`
    threshold: `float`
    """
    threshold = threshold*np.nanmax(smap.data)
    iy, ix = np.where(smap.data<threshold)
    seeds = smap.wcs.array_index_to_world(iy, ix)
    return seeds


def _coordinate_to_linestring(coord, wcs):
    return shapely.LineString(
        np.array(wcs.world_to_pixel(coord)).T
    )


def find_fieldlines_intersecting_loop(fieldlines, coordinate, wcs, width=None, intersection_tolerance=0.7):
    """
    Find fieldlines that intersect a visually defined loop.

    Parameters
    ----------
    fieldlines: `list` of `greensfield.models.Fieldline`
        Fieldlines traced through field extrapolation to iterate over
    coordinate: `~astropy.coordinates.SkyCoord`
        The coordinate corresponding to the visually defined loop. This can either be a 1D path defining the
        "spine" of the loop or a closed path defining the shape of the loop. If the former, ``width`` must
        be specified to define the loop width.
    wcs: `~astropy.wcs.WCS`
        The WCS of the image from which ``coordinate`` was derived.
    width: `float`, optional
        The distance around which to buffer a 1D path to convert it to a polygon. Effectively the width,
        in pixel space, of the visually-defined loop. If None, it assumed that ``coordinate`` defines the
        full polygon of the loop feature.
    intersection_tolerance: `float`, optional
        Any fieldline that has an intersection with the loop polygon above this value will be treated as
        being comprised of the loop.

    Returns
    -------
    intersecting_fieldlines: `list` of `greensfield.models.Fieldline`
    """
    if width is None:
        loop_polygon = coordinate
    else:
        loop_polygon = _coordinate_to_linestring(
            coordinate,
            wcs,
        ).buffer(
            width,
            cap_style='flat'
        )
    intersecting_fieldlines = []
    for f in fieldlines:
        f_shape = _coordinate_to_linestring(f.coordinate, wcs)
        intersection = loop_polygon.intersection(f_shape)
        if intersection.length / f_shape.length >= intersection_tolerance:
            intersecting_fieldlines.append(f)
    return intersecting_fieldlines


def find_fieldlines_within_loop_footprint(fieldlines, coordinate, wcs, radius=None, index=0):
    """
    Find fieldlines whose footpoints fall within the footprint of a visually-defined loop.

    Parameters
    ----------
    fieldlines: `list` of `greensfield.models.Fieldline`
        Fieldlines traced through field extrapolation to iterate over
    coordinate: `~astropy.coordinates.SkyCoord`
        The coordinate defining the base of the visually-defined loop. If a single point,
        ``radius`` is used to define the circular footprint. If radius is None, it is assumed
        that this defines closed path of the loop footprint.
    wcs: `~astropy.wcs.WCS`
        The WCS of the image from which ``coordinate`` was derived.
    radius: `float`, optional
        The radius of the circular loop footprint.
    index: `int`, optional
        The index of the fieldline coordinate to use that defines the fieldline footpoint.

    Returns
    -------
    bundle: `list` of `greensfield.models.Fieldline`
    """
    if radius is None:
        footprint = coordinate
    else:
        footprint = shapely.geometry.Point(wcs.world_to_pixel(coordinate)).buffer(radius)
    bundle = []
    for f in fieldlines:
        footpoint = shapely.Point(wcs.world_to_pixel(f.coordinate[index]))
        if footprint.contains(footpoint):
            bundle.append(f)
    return bundle
