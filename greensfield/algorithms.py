"""
Algorithms for field extrapolation
"""
import astropy.units as u
import numba
import numpy as np

__all__ = ['oblique_schmidt', 'magnetic_field_current_free']


@numba.jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def oblique_schmidt(phi, boundary, delta, shape, z_depth, l_hat):
    """
    Calculate scalar magnetic potential using the oblique Schmidt method
    :cite:p:`schmidt_observable_1964`.

    .. note:: This function is accelerated with ``numba``.

    Parameters
    ----------
    phi: array-like
        Scalar magnetic potential as a function of :math:`x`, :math:`y`,
        and :math:`z` with shape ``(n_x,n_y,n_z)``. This should be initially
        an empty array.
    boundary: array-like
        Magnetic field in the :math:`z`-direction at :math:`z=0` as
        a function of :math:`x` and :math:`y` with shape ``(n_x,n_y)``.
    delta: array-like
        Size of each grid cell in :math:`x`, :math:`y`, and :math:`z`
        with shape ``(3,)``. Must have same units as ``z_depth``.
    shape: array-like
        Number of grid cells in :math:`x`, :math:`y`, and :math:`z`
        with shape ``(3,)``.
    z_depth: `float`
        The depth below the surface at which the magnetic monopole
        is submerged.
    l_hat: array-like
        Unit vector with shape ``(3,)`` indicating the surface normal
        of the lower boundary of the computational domain.

    Returns
    -------
    phi: array-like
        Scalar magnetic potential as a function of :math:`x`, :math:`y`,
        and :math:`z` with shape ``(n_x,n_y,n_z)``.
    """
    factor = 1. / (2. * np.pi) * delta[0] * delta[1]
    for i in numba.prange(shape[0]):
        for j in numba.prange(shape[1]):
            for k in numba.prange(shape[2]):
                Rz = k * delta[2] - z_depth
                lzRz = l_hat[2] * Rz
                for i_prime in range(shape[0]):
                    for j_prime in range(shape[1]):
                        Rx = delta[0] * (i - i_prime)
                        Ry = delta[1] * (j - j_prime)
                        R_mag = np.sqrt(Rx**2 + Ry**2 + Rz**2)
                        num = l_hat[2] + Rz / R_mag
                        denom = R_mag + lzRz + Rx*l_hat[0] + Ry*l_hat[1]
                        green = num / denom
                        phi[i, j, k] += boundary[i_prime, j_prime] * green * factor
    return phi


@u.quantity_input
def magnetic_field_current_free(phi: u.G * u.cm, delta: u.cm):
    r"""
    Compute vector magnetic field.

    Calculate the vector magnetic field using the current-free approximation,

    .. math::
        \\vec{B} = -\\nabla\phi

    The gradient is computed numerically using a five-point stencil,

    .. math::
        \\frac{\partial B}{\partial x_i} \\approx -\left(\\frac{-B_{x_i}(x_i + 2\Delta x_i) + 8B_{x_i}(x_i + \Delta x_i) - 8B_{x_i}(x_i - \Delta x_i) + B_{x_i}(x_i - 2\Delta x_i)}{12\Delta x_i}\\right)

    Parameters
    ----------
    phi : `~astropy.units.Quantity`
        Scalar magnetic potential as a function of :math:`x`, :math:`y`,
        and :math:`z` with shape ``(n_x,n_y,n_z)``.
    delta : `~astropy.units.Quantity`
        Size of each grid cell in :math:`x`, :math:`y`, and :math:`z`
        with shape ``(3,)``.

    Returns
    -------
    B_field : `~astropy.units.Quantity`
        :math:`x`, :math:`y`, and :math:`z` components of the vector magnetic field with
        shape ``(n_x,n_y,n_z,3)``.
    """
    Bfield = u.Quantity(np.zeros(phi.shape+(3,)), 'G')
    # Take gradient using a five-point stencil
    Bfield[2:-2, 2:-2, 2:-2, 0] = -(phi[:-4, 2:-2, 2:-2]
                                    - 8.*phi[1:-3, 2:-2, 2:-2]
                                    + 8.*phi[3:-1, 2:-2, 2:-2]
                                    - phi[4:, 2:-2, 2:-2])/12./delta[0]
    Bfield[2:-2, 2:-2, 2:-2, 1] = -(phi[2:-2, :-4, 2:-2]
                                    - 8.*phi[2:-2, 1:-3, 2:-2]
                                    + 8.*phi[2:-2, 3:-1, 2:-2]
                                    - phi[2:-2, 4:, 2:-2])/12./delta[1]
    Bfield[2:-2, 2:-2, 2:-2, 2] = -(phi[2:-2, 2:-2, :-4]
                                    - 8.*phi[2:-2, 2:-2, 1:-3]
                                    + 8.*phi[2:-2, 2:-2, 3:-1]
                                    - phi[2:-2, 2:-2, 4:])/12./delta[2]
    # Set boundary conditions such that the last two cells in either direction in each dimension
    # are the same as the preceding cell.
    for i in range(Bfield.shape[-1]):
        for j in [0, 1]:
            Bfield[j, :, :, i] = Bfield[2, :, :, i]
            Bfield[:, j, :, i] = Bfield[:, 2, :, i]
            Bfield[:, :, j, i] = Bfield[:, :, 2, i]
        for j in [-2, -1]:
            Bfield[j, :, :, i] = Bfield[-3, :, :, i]
            Bfield[:, j, :, i] = Bfield[:, -3, :, i]
            Bfield[:, :, j, i] = Bfield[:, :, -3, i]

    return Bfield
