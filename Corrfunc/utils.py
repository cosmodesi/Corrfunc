#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of utility routines
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import warnings
from os.path import exists as file_exists

import wurlitzer
from contextlib import contextmanager

import numpy as np


__all__ = ['translate_isa_string_to_enum', 'translate_bin_type_string_to_enum', 'translate_los_type_string_to_enum',
           'get_edges', 'compute_nbins', 'gridlink_sphere']
if sys.version_info[0] < 3:
    __all__ = [n.encode('ascii') for n in __all__]

try:
    xrange
except NameError:
    xrange = range


def get_edges(binfile):
    """
    Helper function to return edges corresponding to ``binfile``.

    Parameters
    -----------
    binfile : string or array-like
       Expected to be a path to a bin file (two columns, lower and upper) or an array containing the bins.

    Returns
    -------
    edges : array
    """
    if isinstance(binfile, str):
        if file_exists(binfile):
            # The equivalent of read_binfile() in io.c
            with open(binfile, 'r') as file:
                binfile = []
                for iline, line in enumerate(file):
                    lowhi = line.split()
                    if len(lowhi) == 1:
                        binfile.append(lowhi[0])
                    elif len(lowhi) == 2:
                        low, hi = lowhi
                        if iline == 0:
                            binfile.append(low)
                        binfile.append(hi)
                    else:
                        break
        else:
            msg = "Could not find file = `{0}` containing the bins".format(binfile)
            raise IOError(msg)

    # For a valid bin specifier, there must be at least 1 bin.
    if len(binfile) >= 1:
        binfile = np.array(binfile, order='C', dtype='f8')
        binfile.sort()
        return binfile

    msg = "Input `binfile` was not a valid array (>= 1 element)."\
          "Num elements = {0}".format(len(binfile))
    raise TypeError(msg)


def translate_isa_string_to_enum(isa):
    """
    Helper function to convert an user-supplied string to the
    underlying enum in the C-API. The extensions only have specific
    implementations for AVX512F, AVX, SSE42 and FALLBACK. Any other
    value will raise a ValueError.

    Parameters
    ------------
    isa: string
       A string containing the desired instruction set. Valid values are
       ['AVX512F', 'AVX', 'SSE42', 'FALLBACK', 'FASTEST']

    Returns
    --------
    instruction_set: integer
       An integer corresponding to the desired instruction set, as used in the
       underlying C API. The enum used here should be defined *exactly* the
       same way as the enum in ``utils/defs.h``.

    """

    msg = "Input to translate_isa_string_to_enum must be "\
          "of string type. Found type = {0}".format(type(isa))
    try:
        if not isinstance(isa, basestring):
            raise TypeError(msg)
    except NameError:
        if not isinstance(isa, str):
            raise TypeError(msg)
    valid_isa = ['FALLBACK', 'AVX512F', 'AVX2', 'AVX', 'SSE42', 'FASTEST']
    isa_upper = isa.upper()
    if isa_upper not in valid_isa:
        msg = "Desired instruction set = {0} is not in the list of valid "\
              "instruction sets = {1}".format(isa, valid_isa)
        raise ValueError(msg)

    enums = {'FASTEST': -1,
             'FALLBACK': 0,
             'SSE': 1,
             'SSE2': 2,
             'SSE3': 3,
             'SSSE3': 4,
             'SSE4': 5,
             'SSE42': 6,
             'AVX': 7,
             'AVX2': 8,
             'AVX512F': 9
             }
    try:
        return enums[isa_upper]
    except KeyError:
        raise ValueError("Do not know instruction type = {}, valid instructions are {}".format(isa, enums.keys()))


def translate_bin_type_string_to_enum(bin_type):
    """
    Helper function to convert an user-supplied string to the
    underlying enum for binning type in the C-API.

    Parameters
    ------------
    bin_type: string
       A string containing the desired bin type. Valid values are
       ['AUTO', 'CUSTOM', 'LIN'].

    Returns
    --------
    bin_type: integer
       An integer corresponding to the desired bin type, as used in the
       underlying C API. The enum used here should be defined *exactly* the
       same way as the enum in ``utils/defs.h``.

    """

    msg = "Input to translate_bin_type_string_to_enum must be "\
          "of string type. Found type = {0}".format(type(bin_type))
    try:
        if not isinstance(bin_type, basestring):
            raise TypeError(msg)
    except NameError:
        if not isinstance(bin_type, str):
            raise TypeError(msg)
    valid_bin_type = ['AUTO', 'CUSTOM', 'LIN']
    bin_type_upper = bin_type.upper()
    if bin_type_upper not in valid_bin_type:
        msg = "Desired bin type = {0} is not in the list of valid "\
              "bin types = {1}".format(bin_type, valid_bin_type)
        raise ValueError(msg)

    enums = {'AUTO': 0,
             'LIN': 1,
             'CUSTOM': 2,
             }
    try:
        return enums[bin_type_upper]
    except KeyError:
        raise ValueError("Do not know bin type = {}, valid bin types are {}".format(bin_type, enums.keys()))


def translate_los_type_string_to_enum(los_type):
    """
    Helper function to convert an user-supplied string to the
    underlying enum for line-of-sight type in the C-API.

    Parameters
    ------------
    los_type: string
       A string containing the desired line-of-sight. Valid values are
       ['MIDPOINT', 'FIRSTPOINT']

    Returns
    --------
    bin_type: integer
       An integer corresponding to the desired line-of-sight type, as used in the
       underlying C API. The enum used here should be defined *exactly* the
       same way as the enum in ``utils/defs.h``.

    """

    msg = "Input to translate_los_type_string_to_enum must be "\
          "of string type. Found type = {0}".format(type(los_type))
    try:
        if not isinstance(los_type, basestring):
            raise TypeError(msg)
    except NameError:
        if not isinstance(los_type, str):
            raise TypeError(msg)
    valid_los_type = ['MIDPOINT', 'FIRSTPOINT']
    los_type_upper = los_type.upper()
    if los_type_upper not in valid_los_type:
        msg = "Desired los type = {0} is not in the list of valid "\
              "los types = {1}".format(bin_type, valid_bin_type)
        raise ValueError(msg)

    enums = {'MIDPOINT': 0,
             'FIRSTPOINT': 1,
             }
    try:
        return enums[los_type_upper]
    except KeyError:
        raise ValueError("Do not know los type = {}, valid los types are {}".format(los_type, enums.keys()))


def compute_nbins(max_diff, binsize,
                 refine_factor=1,
                 max_nbins=None):
    """
    Helper utility to find the number of bins for
    that satisfies the constraints of (binsize, refine_factor, and max_nbins).

    Parameters
    ------------

    max_diff : double
       Max. difference (spatial or angular) to be spanned,
       (i.e., range of allowed domain values)

    binsize : double
       Min. allowed binsize (spatial or angular)

    refine_factor : integer, default 1
       How many times to refine the bins. The refinements occurs
       after ``nbins`` has already been determined (with ``refine_factor-1``).
       Thus, the number of bins will be **exactly** higher by
       ``refine_factor`` compared to the base case of ``refine_factor=1``

    max_nbins : integer, default None
       Max number of allowed cells

    Returns
    ---------

    nbins: integer, >= 1
       Number of bins that satisfies the constraints of
       bin size >= ``binsize``, the refinement factor
       and nbins <= ``max_nbins``.

    Example
    ---------

    >>> from Corrfunc.utils import compute_nbins
    >>> max_diff = 180
    >>> binsize = 10
    >>> compute_nbins(max_diff, binsize)
    18
    >>> refine_factor=2
    >>> max_nbins = 20
    >>> compute_nbins(max_diff, binsize, refine_factor=refine_factor,
    ...              max_nbins=max_nbins)
    20

    """

    if max_diff <= 0 or binsize <= 0:
        msg = 'Error: Invalid value for max_diff = {0} or binsize = {1}. '\
              'Both must be positive'.format(max_diff, binsize)
        raise ValueError(msg)
    if max_nbins is not None and max_nbins < 1:
        msg = 'Error: Invalid for the max. number of bins allowed = {0}.'\
              'Max. nbins must be >= 1'.format(max_nbins)
        raise ValueError(msg)

    if refine_factor < 1:
        msg = 'Error: Refine factor must be >=1. Found refine_factor = '\
              '{0}'.format(refine_factor)
        raise ValueError(msg)

    # At least 1 bin
    ngrid = max(int(1), int(max_diff/binsize))

    # Then refine
    ngrid *= refine_factor

    # But don't exceed max number of bins
    # (if passed as a parameter)
    if max_nbins:
        ngrid = min(int(max_nbins), ngrid)

    return ngrid


def gridlink_sphere(thetamax,
                    ra_limits=None,
                    dec_limits=None,
                    link_in_ra=True,
                    ra_refine_factor=1, dec_refine_factor=1,
                    max_ra_cells=100, max_dec_cells=200,
                    return_num_ra_cells=False,
                    input_in_degrees=True):
    """
    A method to optimally partition spherical regions such that pairs of
    points within a certain angular separation, ``thetamax``, can be quickly
    computed.

    Generates the  binning scheme used in :py:mod:`Corrfunc.mocks.DDtheta_mocks`
    for a spherical region in Right Ascension (RA), Declination (DEC)
    and a maximum angular separation.

    For a given ``thetamax``, regions on the sphere are divided into bands
    in DEC bands, with the width in DEC equal to ``thetamax``. If
    ``link_in_ra`` is set, then these DEC bands are further sub-divided
    into RA cells.

    Parameters
    ----------

    thetamax : double
       Max. angular separation of pairs. Expected to be in degrees
       unless ``input_in_degrees`` is set to ``False``.

    ra_limits : array of 2 doubles. Default [0.0, 2*pi]
       Range of Righ Ascension (longitude) for the spherical region

    dec_limits : array of 2 doubles. Default [-pi/2, pi/2]
       Range of Declination (latitude) values for the spherical region

    link_in_ra : Boolean. Default True
       Whether linking in RA is done (in addition to linking in DEC)

    ra_refine_factor : integer, >= 1. Default 1
       Controls the sub-division of the RA cells. For a large number of
       particles, higher `ra_refine_factor` typically results in a faster
       runtime

    dec_refine_factor : integer, >= 1. Default 1
       Controls the sub-division of the DEC cells. For a large number of
       particles, higher `dec_refine_factor` typically results in a faster
       runtime

    max_ra_cells : integer, >= 1. Default 100
       The max. number of RA cells **per DEC band**.

    max_dec_cells : integer >= 1. Default 200
       The max. number of total DEC bands

    return_num_ra_cells: bool, default False
       Flag to return the number of RA cells per DEC band

    input_in_degrees : Boolean. Default True
       Flag to show if the input quantities are in degrees. If set to
       False, all angle inputs will be taken to be in radians.

    Returns
    ---------

    sphere_grid : A numpy compound array, shape (ncells, 2)
       A numpy compound array with fields ``dec_limit`` and ``ra_limit`` of
       size 2 each. These arrays contain the beginning and end of DEC
       and RA regions for the cell.

    num_ra_cells: numpy array, returned if ``return_num_ra_cells`` is set
       A numpy array containing the number of RA cells per declination band


    .. note:: If ``link_in_ra=False``, then there is effectively one RA bin
       per DEC band. The  'ra_limit' field will show the range of allowed
       RA values.


    .. seealso:: :py:mod:`Corrfunc.mocks.DDtheta_mocks`

    Example
    --------

    >>> from Corrfunc.utils import gridlink_sphere
    >>> import numpy as np
    >>> try:  # Backwards compatibility with old Numpy print formatting
    ...     np.set_printoptions(legacy='1.13')
    ... except TypeError:
    ...     pass
    >>> thetamax=30
    >>> grid = gridlink_sphere(thetamax)
    >>> print(grid)  # doctest: +NORMALIZE_WHITESPACE
    [([-1.57079633, -1.04719755], [ 0.        ,  3.14159265])
     ([-1.57079633, -1.04719755], [ 3.14159265,  6.28318531])
     ([-1.04719755, -0.52359878], [ 0.        ,  3.14159265])
     ([-1.04719755, -0.52359878], [ 3.14159265,  6.28318531])
     ([-0.52359878,  0.        ], [ 0.        ,  1.25663706])
     ([-0.52359878,  0.        ], [ 1.25663706,  2.51327412])
     ([-0.52359878,  0.        ], [ 2.51327412,  3.76991118])
     ([-0.52359878,  0.        ], [ 3.76991118,  5.02654825])
     ([-0.52359878,  0.        ], [ 5.02654825,  6.28318531])
     ([ 0.        ,  0.52359878], [ 0.        ,  1.25663706])
     ([ 0.        ,  0.52359878], [ 1.25663706,  2.51327412])
     ([ 0.        ,  0.52359878], [ 2.51327412,  3.76991118])
     ([ 0.        ,  0.52359878], [ 3.76991118,  5.02654825])
     ([ 0.        ,  0.52359878], [ 5.02654825,  6.28318531])
     ([ 0.52359878,  1.04719755], [ 0.        ,  3.14159265])
     ([ 0.52359878,  1.04719755], [ 3.14159265,  6.28318531])
     ([ 1.04719755,  1.57079633], [ 0.        ,  3.14159265])
     ([ 1.04719755,  1.57079633], [ 3.14159265,  6.28318531])]
    >>> grid = gridlink_sphere(60, dec_refine_factor=3, ra_refine_factor=2)
    >>> print(grid)  # doctest: +NORMALIZE_WHITESPACE
    [([-1.57079633, -1.22173048], [ 0.        ,  1.57079633])
     ([-1.57079633, -1.22173048], [ 1.57079633,  3.14159265])
     ([-1.57079633, -1.22173048], [ 3.14159265,  4.71238898])
     ([-1.57079633, -1.22173048], [ 4.71238898,  6.28318531])
     ([-1.22173048, -0.87266463], [ 0.        ,  1.57079633])
     ([-1.22173048, -0.87266463], [ 1.57079633,  3.14159265])
     ([-1.22173048, -0.87266463], [ 3.14159265,  4.71238898])
     ([-1.22173048, -0.87266463], [ 4.71238898,  6.28318531])
     ([-0.87266463, -0.52359878], [ 0.        ,  1.57079633])
     ([-0.87266463, -0.52359878], [ 1.57079633,  3.14159265])
     ([-0.87266463, -0.52359878], [ 3.14159265,  4.71238898])
     ([-0.87266463, -0.52359878], [ 4.71238898,  6.28318531])
     ([-0.52359878, -0.17453293], [ 0.        ,  1.57079633])
     ([-0.52359878, -0.17453293], [ 1.57079633,  3.14159265])
     ([-0.52359878, -0.17453293], [ 3.14159265,  4.71238898])
     ([-0.52359878, -0.17453293], [ 4.71238898,  6.28318531])
     ([-0.17453293,  0.17453293], [ 0.        ,  1.57079633])
     ([-0.17453293,  0.17453293], [ 1.57079633,  3.14159265])
     ([-0.17453293,  0.17453293], [ 3.14159265,  4.71238898])
     ([-0.17453293,  0.17453293], [ 4.71238898,  6.28318531])
     ([ 0.17453293,  0.52359878], [ 0.        ,  1.57079633])
     ([ 0.17453293,  0.52359878], [ 1.57079633,  3.14159265])
     ([ 0.17453293,  0.52359878], [ 3.14159265,  4.71238898])
     ([ 0.17453293,  0.52359878], [ 4.71238898,  6.28318531])
     ([ 0.52359878,  0.87266463], [ 0.        ,  1.57079633])
     ([ 0.52359878,  0.87266463], [ 1.57079633,  3.14159265])
     ([ 0.52359878,  0.87266463], [ 3.14159265,  4.71238898])
     ([ 0.52359878,  0.87266463], [ 4.71238898,  6.28318531])
     ([ 0.87266463,  1.22173048], [ 0.        ,  1.57079633])
     ([ 0.87266463,  1.22173048], [ 1.57079633,  3.14159265])
     ([ 0.87266463,  1.22173048], [ 3.14159265,  4.71238898])
     ([ 0.87266463,  1.22173048], [ 4.71238898,  6.28318531])
     ([ 1.22173048,  1.57079633], [ 0.        ,  1.57079633])
     ([ 1.22173048,  1.57079633], [ 1.57079633,  3.14159265])
     ([ 1.22173048,  1.57079633], [ 3.14159265,  4.71238898])
     ([ 1.22173048,  1.57079633], [ 4.71238898,  6.28318531])]

    """

    from math import radians, pi
    import numpy as np


    if input_in_degrees:
        thetamax = radians(thetamax)
        if ra_limits:
            ra_limits = [radians(x) for x in ra_limits]
        if dec_limits:
            dec_limits = [radians(x) for x in dec_limits]

    if not ra_limits:
        ra_limits = [0.0, 2.0*pi]

    if not dec_limits:
        dec_limits = [-0.5*pi, 0.5*pi]

    if dec_limits[0] >= dec_limits[1]:
        msg = 'Declination limits should be sorted in increasing '\
              'order. However, dec_limits = [{0}, {1}] is not'.\
              format(dec_limits[0], dec_limits[1])
        raise ValueError(msg)

    if ra_limits[0] >= ra_limits[1]:
        msg = 'Declination limits should be sorted in increasing '\
              'order. However, ra_limits = [{0}, {1}] is not'.\
              format(ra_limits[0], ra_limits[1])
        raise ValueError(msg)

    if dec_limits[0] < -0.5*pi or dec_limits[1] > 0.5*pi:
        msg = 'Valid range of values for declination are [-pi/2, +pi/2] deg. '\
              'However, dec_limits = [{0}, {1}] does not fall within that '\
              'range'.format(dec_limits[0], dec_limits[1])
        raise ValueError(msg)

    if ra_limits[0] < 0.0 or ra_limits[1] > 2.0*pi:
        msg = 'Valid range of values for declination are [0.0, 2*pi] deg. '\
              'However, ra_limits = [{0}, {1}] does not fall within that '\
              'range'.format(ra_limits[0], ra_limits[1])
        raise ValueError(msg)

    dec_diff = abs(dec_limits[1] - dec_limits[0])
    ngrid_dec = compute_nbins(dec_diff, thetamax,
                             refine_factor=dec_refine_factor,
                             max_nbins=max_dec_cells)

    dec_binsize = dec_diff/ngrid_dec

    # Upper and lower limits of the declination bands
    grid_dtype = np.dtype({'names': ['dec_limit', 'ra_limit'],
                          'formats': [(np.float64, (2, )), (np.float64, (2, ))]
    })
    if not link_in_ra:
        sphere_grid = np.zeros(ngrid_dec, dtype=grid_dtype)
        for i, r in enumerate(sphere_grid['dec_limit']):
            r[0] = dec_limits[0] + i*dec_binsize
            r[1] = dec_limits[0] + (i+1)*dec_binsize

        for r in sphere_grid['ra_limit']:
            r[0] = ra_limits[0]
            r[1] = ra_limits[1]

        return sphere_grid

    # RA linking is requested
    ra_diff = ra_limits[1] - ra_limits[0]
    sin_half_thetamax = np.sin(thetamax)

    totncells = 0
    num_ra_cells = np.zeros(ngrid_dec, dtype=np.int64)
    num_ra_cells[:] = ra_refine_factor
    # xrange is replaced by range for python3
    # by using a try/except at the top
    for idec in xrange(ngrid_dec):
        dec_min = dec_limits[0] + idec*dec_binsize
        dec_max = dec_min + dec_binsize

        cos_dec_min = np.cos(dec_min)
        cos_dec_max = np.cos(dec_max)

        if cos_dec_min < cos_dec_max:
            min_cos = cos_dec_min
        else:
            min_cos = cos_dec_max

        if min_cos > 0:
            _tmp = sin_half_thetamax/min_cos
            # clamp to range [0.0, 1.0]
            _tmp = max(min(_tmp, 1.0), 0.0)
            ra_binsize = min(2.0 * np.arcsin(_tmp), ra_diff)
            num_ra_cells[idec] = compute_nbins(ra_diff, ra_binsize,
                                              refine_factor=ra_refine_factor,
                                              max_nbins=max_ra_cells)

    totncells = num_ra_cells.sum()
    sphere_grid = np.zeros(totncells, dtype=grid_dtype)
    ra_binsizes = ra_diff/num_ra_cells

    start = 0
    for idec in xrange(ngrid_dec):
        assert start + num_ra_cells[idec] <= totncells
        source_sel = np.s_[start:start+num_ra_cells[idec]]
        for ira, r in enumerate(sphere_grid[source_sel]):
            r['dec_limit'][0] = dec_limits[0] + dec_binsize*idec
            r['dec_limit'][1] = dec_limits[0] + dec_binsize*(idec + 1)
            r['ra_limit'][0] = ra_limits[0] + ra_binsizes[idec] * ira
            r['ra_limit'][1] = ra_limits[0] + ra_binsizes[idec] * (ira + 1)

        start += num_ra_cells[idec]

    if return_num_ra_cells:
        return sphere_grid, num_ra_cells
    else:
        return sphere_grid


def convert_to_native_endian(array, warn=False):
    '''
    Returns the supplied array in native endian byte-order.
    If the array already has native endianness, then the
    same array is returned.

    Parameters
    ----------
    array: np.ndarray
        The array to convert
    warn: bool, optional
        Print a warning if `array` is not already native endian.
        Default: False.

    Returns
    -------
    new_array: np.ndarray
        The array in native-endian byte-order.

    Example
    -------
    >>> import numpy as np
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = sys_is_le and '<' or '>'
    >>> swapped_code = sys_is_le and '>' or '<'
    >>> native_dt = np.dtype(native_code + 'i4')
    >>> swapped_dt = np.dtype(swapped_code + 'i4')
    >>> arr = np.arange(10, dtype=native_dt)
    >>> new_arr = convert_to_native_endian(arr)
    >>> arr is new_arr
    True
    >>> arr = np.arange(10, dtype=swapped_dt)
    >>> new_arr = convert_to_native_endian(arr)
    >>> new_arr.dtype.byteorder == '=' or new_arr.dtype.byteorder == native_code
    True
    >>> convert_to_native_endian(None) is None
    True
    '''

    import warnings

    if array is None:
        return array

    import numpy as np
    array = np.asarray(array, order='C')

    system_is_little_endian = (sys.byteorder == 'little')
    array_is_little_endian = (array.dtype.byteorder == '<')
    if (array_is_little_endian != system_is_little_endian) and not (array.dtype.byteorder == '='):
        if warn:
            warnings.warn("One or more input array has non-native endianness! A copy will"\
                      " be made with the correct endianness.")
        array = np.array(array, order='C')
    return array


def is_native_endian(array):
    '''
    Checks whether the given array is native-endian.
    None evaluates to True.

    Parameters
    ----------
    array: np.ndarray
        The array to check

    Returns
    -------
    is_native: bool
        Whether the endianness is native

    Example
    -------
    >>> import numpy as np
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = sys_is_le and '<' or '>'
    >>> swapped_code = sys_is_le and '>' or '<'
    >>> native_dt = np.dtype(native_code + 'i4')
    >>> swapped_dt = np.dtype(swapped_code + 'i4')
    >>> arr = np.arange(10, dtype=native_dt)
    >>> is_native_endian(arr)
    True
    >>> arr = np.arange(10, dtype=swapped_dt)
    >>> is_native_endian(arr)
    False
    '''

    if array is None:
        return True

    import numpy as np
    array = np.asanyarray(array)

    system_is_little_endian = (sys.byteorder == 'little')
    array_is_little_endian = (array.dtype.byteorder == '<')
    return (array_is_little_endian == system_is_little_endian) or (array.dtype.byteorder == '=')


def process_weights(weights1, weights2, X1, X2, weight_type, autocorr):
    '''
    Process the user-passed weights in a manner that can be handled by
    the C code.  `X1` and `X2` are the corresponding pos arrays; they
    allow us to get the appropriate dtype and length when weight arrays
    are not explicitly given.

    1) Scalar weights are promoted to arrays
    2) If only one set of weights is given, the other is generated with
        weights = 1, but only for weight_type = 'pair_product'.  Otherwise
        a ValueError will be raised.
    3) Weight arrays are reshaped to 2D (shape n_weights_per_particle, n_particles)
    '''
    import numpy as np

    if weight_type is None:
        # Weights will not be used; do nothing
        return weights1, weights2

    # Takes a scalar, 1d, or 2d weights array
    # and returns a 2d array of shape (nweights,npart)
    def prep(weights, x):
        if weights is None:
            return weights

        if not isinstance(weights, (tuple, list)):
            if isinstance(weights, np.ndarray) and weights.ndim == 2:
                weights = list(weights)
            else:
                weights = [weights]

        toret = []
        for w in weights:
            w = np.asarray(w)
            w.shape = (-1,)
            if w.shape[-1] == 1:
                w = np.tile(w, len(x))
            toret.append(w)

        return toret

    weights1 = prep(weights1, X1)

    if not autocorr:
        weights2 = prep(weights2, X2)

        if (weights1 is None) != (weights2 is None):
            if weight_type != 'pair_product':
                raise ValueError("If using a weight_type other than "\
                                 "'pair_product', you must provide "\
                                 "both weight arrays.")

        if weights1 is None and weights2 is not None:
            weights1 = [np.ones_like(X1) for w in weights2]

        if weights2 is None and weights1 is not None:
            weights2 = [np.ones_like(X2) for w in weights1]

    return weights1, weights2


@contextmanager
def sys_pipes():
    '''
    In a Jupyter notebook, Python's ``sys.stdout`` and ``sys.stderr`` are redirected
    so output ends up in cells. But C extensions don't know about that! Wurlitzer
    uses os.dup2 to redirect fds 1 & 2 to the new location and restore them on return,
    but will cause the output to hang if they were not already redirected. It seems
    we can compare Python's ``sys.stdout`` to the saved ``sys.__stdout__`` to tell
    if redirection occurred. We will also check if the output is a TTY as a safety
    net, even though it is probably a subset of the preceeding check.

    Basic usage is:

    >>> with sys_pipes():  # doctest: +SKIP
    ...    call_some_c_function()

    See the Wurlitzer package for usage of `wurlitzer.pipes()`;
    see also https://github.com/manodeep/Corrfunc/issues/157,
    https://github.com/manodeep/Corrfunc/issues/269.
    '''

    kwargs = {}
    if sys.stdout.isatty() or (sys.stdout is sys.__stdout__):
        kwargs['stdout'] = None
    else:
        kwargs['stdout'] = sys.stdout
    if sys.stderr.isatty() or (sys.stderr is sys.__stderr__):
        kwargs['stderr'] = None
    else:
        kwargs['stderr'] = sys.stderr

    # Redirection might break for any number of reasons, like
    # stdout/err already being closed/redirected.  We probably
    # prefer not to crash in that case and instead continue
    # without any redirection.
    try:
        with wurlitzer.pipes(**kwargs):
            yield
    except:
        yield


def check_runtime_env():
    '''
    Detect any computing environment conditions that may cause Corrfunc
    to fail, and inform the user if there is any action they can take.
    '''

    # Check if Cray hugepages is enabled at NERSC, which will crash
    # C Python extensions due to a hugepages bug
    if 'NERSC_HOST' in os.environ and os.getenv('HUGETLB_DEFAULT_PAGE_SIZE'):
        warnings.warn('Warning: Cray hugepages has a bug that may crash '
                      'Corrfunc. You might be able to fix such a crash with '
                      '`module unload craype-hugepages2M` (see '
                      'https://github.com/manodeep/Corrfunc/issues/245 '
                      'for details)')

if __name__ == '__main__':
    import doctest
    doctest.testmod()
