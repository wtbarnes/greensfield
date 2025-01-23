"""
Models for field extrapolation
"""
import astropy.units as u
import numpy as np
import xarray

from astropy.coordinates import CartesianRepresentation, SkyCoord
from streamtracer import StreamTracer, VectorGrid
from sunpy.coordinates import Heliocentric, PlanarScreen
from sunpy.coordinates.utils import solar_angle_equivalency

from greensfield.algorithms import oblique_schmidt, magnetic_field_current_free
from greensfield.util import make_boundary_magnetogram

__all__ = ['ExtrapolatorBase', 'ObliqueSchmidtExtrapolator']


class ExtrapolatorBase:
    """
    Extrapolate 3D magnetic field in a Cartesian box.

    Parameters
    ----------
    magnetogram
    tangent_coord
    corners
    resample_factor
    scale_z
    extent_z
    """

    def __init__(self,
                 magnetogram,
                 tangent_coord,
                 corners,
                 resample_factor=1,
                 scale_z=None,
                 extent_z=None):
        self.magnetogram = magnetogram
        self.tangent_coord = tangent_coord
        self.boundary_magnetogram = make_boundary_magnetogram(self.magnetogram,
                                                              self.tangent_coord,
                                                              corners,
                                                              resample_factor)
        self.scale = self._get_scale(scale_z)
        self.shape = self._get_shape(extent_z)

    @property
    def _planar_screen(self):
        return PlanarScreen(self.tangent_coord,
                            distance_from_center=self.tangent_coord.radius,
                            only_off_disk=False)

    @property
    def hcc_frame(self):
        return Heliocentric(observer=self.boundary_magnetogram.observer_coordinate)
    
    @u.quantity_input(sale_z='Mm')
    def _get_scale(self, scale_z=None) -> u.Mm:
        scale = (u.Quantity(self.boundary_magnetogram.scale) * 1*u.pix).to(
            'Mm',
            equivalencies=solar_angle_equivalency(self.magnetogram.observer_coordinate)
        )
        if scale_z is None:
            scale_z = scale.min()
        scale = np.append(scale, scale_z)
        return scale
    
    @u.quantity_input(extent_z='Mm')
    def _get_shape(self, extent_z=None):
        shape = np.round(
            u.Quantity(self.boundary_magnetogram.dimensions)
            .to_value('pix')
        ).astype(int)
        if extent_z is None:
            shape_z = shape.max()
        else:
            shape_z = (extent_z / self.scale[-1]).decompose().to_value(u.dimensionless_unscaled)
            shape_z = np.round(shape_z).astype(int)
        shape = np.append(shape, shape_z)
        return shape

    @property
    def grid(self):
        """
        List of arrays representing the physical coordinates along each axis.
        """
        grid = [d*np.arange(s) for d,s in zip(self.scale, self.shape, strict=True)]
        llc = self.lower_left_corner
        grid = [g+o for g,o in zip(grid, llc.cartesian.xyz, strict=True)]
        return grid
    
    @property
    def _xarray_coords(self):
        names = ['x', 'y', 'z']
        return {
            name: xarray.Variable([name], grid.to_value('Mm'), attrs={'unit':'Mm'})
            for name, grid in zip(names, self.grid, strict=True)
        }

    @property
    def lower_left_corner(self):
        """
        Coordinate of of the lower left corner of the boundary magnetogram
        """
        with self._planar_screen:
            llc = (self.boundary_magnetogram
                   .wcs
                   .pixel_to_world(0,0)
                   .transform_to(self.hcc_frame))
        return llc

    def extrapolate(self):
        raise NotImplementedError

    def trace(self, ds, seeds=None, seed_threshold=-1e-3, tracer=None):
        """
        Trace fieldlines through extrapolated volume.

        Parameters
        ----------
        ds: `xarray.Dataset`
            Dataset containing extrapolated magnetic field
        seeds: `~astropy.coordinates.SkyCoord`, optional
            Coordinates of seed points. By default, these will be computed
            by thresholding the boundary magnetogram
        seed_threshold: `float`, optional
            Fraction of maximum value of boundary magnetogram below which seed
            points will be placed. If ``seed_points`` are specified explicitly,
            this will be ignored.
        """
        # create seed points
        if seeds is None:
            threshold = seed_threshold*np.nanmax(self.boundary_magnetogram.data)
            iy, ix = np.where(self.boundary_magnetogram.data<threshold)
            seeds = self.boundary_magnetogram.wcs.array_index_to_world(iy, ix)
        with self._planar_screen:
            seeds = seeds.transform_to(self.hcc_frame)
        # build tracing grid
        ds_B = ds['magnetic_field']
        vg = VectorGrid(ds_B.data,
                        grid_coords=[ds_B.x.data, ds_B.y.data, ds_B.z.data])
        # trace
        if tracer is None:
            tracer = StreamTracer(100000, 0.01)
        tracer.trace(seeds.cartesian.xyz.T.to_value(ds_B.x.unit),
                     vg,
                     direction=0)
        fieldlines = [SkyCoord(*sl.T, unit=ds_B.x.unit, frame=self.hcc_frame) for sl in tracer.xs]
        # TODO: drop fieldlines back down to surface in z
        return fieldlines


class ObliqueSchmidtExtrapolator(ExtrapolatorBase):
    """
    Extrapolate magnetic field using the oblique Schmidt method of Sakurai (1981).

    Parameters
    ----------
    """ 

    @property
    def l_hat(self):
        l_hat = self.magnetogram.observer_coordinate.transform_to(self.hcc_frame).cartesian
        l_hat -= CartesianRepresentation(0,0,1) * self.boundary_magnetogram.rsun_meters
        l_hat /= l_hat.norm()
        return l_hat

    def extrapolate(self):
        """
        Extrapolate magnetic field from lower boundary

        Returns
        -------
        : `xarray.Dataset`
            Dataset containing potential and magnetic field on 3D Cartesian grid
        """
        z_depth = -self.scale[-1] / np.sqrt(2*np.pi)
        potential = np.zeros(self.shape)
        potential = oblique_schmidt(potential,
                                    self.boundary_magnetogram.quantity.to_value('G').T,
                                    self.scale.to_value('Mm'),
                                    self.shape,
                                    z_depth.to_value('Mm'),
                                    self.l_hat.xyz.value)
        potential = u.Quantity(potential, 'G Mm')
        field = magnetic_field_current_free(potential, self.scale)
        coords = self._xarray_coords
        da_potential = xarray.DataArray(potential.to_value('G Mm'),
                                        dims=['x','y','z'],
                                        coords=coords,
                                        attrs={'unit':'G Mm'})
        da_field =  xarray.DataArray(field.to_value('G'),
                                     dims=list(da_potential.dims)+['component'],
                                     coords={**coords, **{'component': ['Bx','By','Bz']}},
                                     attrs={'unit':'G'})
        return xarray.Dataset({'potential': da_potential, 'magnetic_field': da_field})
