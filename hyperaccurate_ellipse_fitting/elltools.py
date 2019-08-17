"""Sampling tool, conversion between representations e.t.c."""
import functools
from typing import Sized, Tuple

import numpy as np
from pynverse import inversefunc
from scipy.special import ellipeinc as _ellipeinc

# Ellipse parameter as in algebraic representation
# (A, B, C, D, E, F)   <=>   A xx + 2B xy + C yy + 2D x + 2E y + F = 0
EllipseParametersAlgebraic = Tuple[float, float, float, float, float, float]

# Ellipse parameter as affine transformations
# `x = R∑r + c` where |r| = 1
EllipseParametersAffine = Tuple[
    Tuple[float, float],  # center coord
    Tuple[float, float],  # axes length
    float,                # tilt angle
]


def algebraic2affine(param, f0: float) -> EllipseParametersAffine:
    """
    Restore ellipse parameters from dlt parameter vector

    Raises:
        RuntimeError: If not ellipse

    Returns:

    Example 1. Circles:
        >>> algebraic2affine((1, 0, 1, 0, 0, -1), f0=1)
        ((0.0, 0.0), (1.0, 1.0), 0.0)
        >>> algebraic2affine((2, 0, 2, 0, 0, -2), f0=1)
        ((0.0, 0.0), (1.0, 1.0), 0.0)

        # xx + yy - 2x = 0
        >>> algebraic2affine((1, 0, 1, -1, 0, 0), f0=1)
        ((1.0, 0.0), (1.0, 1.0), 0.0)

        # 4 xx + 4 yy - 8x + 3 = 0
        >>> algebraic2affine((4, 0, 4, -4, 0, 3), f0=1)
        ((1.0, 0.0), (0.5, 0.5), 0.0)
    
    Example 2. Some ellipse
        >>> cxy, mm, t = algebraic2affine((10, 6, 10, 0, 0, -1), f0=1)
        >>> np.testing.assert_almost_equal(cxy, (0, 0))
        >>> np.testing.assert_almost_equal(mm, (.25, .5))
        >>> np.testing.assert_almost_equal(t % np.pi, .25 * np.pi)
    """
    a, b, c, d, e, f = param
    d /= f0
    e /= f0
    f /= f0*f0
    mat = np.array(
        (
            (a, b, d),
            (b, c, e),
            (d, e, f),
        ),
        dtype=float
    )
    val2, rot2 = np.linalg.eig(mat[:2, :2])
    n_positive = (np.sign(val2) > 0).sum()
    if n_positive == 2:
        # everyone is happy and rainbows and stuff
        pass
    elif n_positive == 0:
        val2 *= -1
        rot2 *= -1
        mat *= -1
    else:
        raise RuntimeError("Not an ellipse")

    inv2 = np.linalg.inv(mat[:2, :2])
    b = mat[:2, 2]
    k = b.T @ inv2 @ b - mat[2, 2]
    scale2 = 1 / np.sqrt(np.diag(rot2.T @ mat[:2, :2] @ rot2) / k)
    t = - np.linalg.inv(mat[:2, :2]) @ mat[:2, 2]
    major_angle = np.arctan2(rot2[1, 0], rot2[0, 0])

    # quick sanity check
    np.testing.assert_almost_equal(
        k * np.array(affine2algebraic((tuple(t), tuple(scale2), major_angle), f0=f0)),
        param,
    )

    return tuple(t), tuple(scale2), major_angle


def affine2algebraic(params: EllipseParametersAffine, f0: float) -> EllipseParametersAlgebraic:
    """Affine representation to Algebraic representation"""
    trans2, scale2, angle = params
    rot3 = np.array((
        (np.cos(angle), - np.sin(angle), 0),
        (np.sin(angle),   np.cos(angle), 0),
        (0, 0, 1),
    ))
    trans3 = np.identity(3)
    trans3[:2, 2] = trans2
    scale3 = np.diag((*scale2, 1))

    transform = trans3 @ rot3 @ scale3
    inv = np.linalg.inv(transform)

    mat = inv.T @ np.diag((1, 1, -1)) @ inv
    a, b, c = mat[0, 0], mat[1, 0], mat[1, 1]
    d, e = mat[:2, 2] / f0
    f = mat[2, 2] / f0 / f0

    return np.array((a, b, c, d, e, f))


def ell(angle: float, eccentricity2: float) -> float:
    """Get elliptic arc length from 0 to `angle`, given eccentricity squared `eccentricity2`."""
    return _ellipeinc(angle, eccentricity2)


def ell_inv(length: float, eccentricity2: float) -> float:
    """Inverse of `ell`, fixed `eccentricity`."""
    func = functools.partial(ell, eccentricity2=eccentricity2)
    if isinstance(length, Sized):
        return inversefunc(func, y_values=length, domain=[- 2 * np.pi - 1e-7, 2 * np.pi + 1e-7], accuracy=7)
    return inversefunc(func, y_values=[length], domain=[-2 * np.pi - 1e-7, 2 * np.pi + 1e-7], accuracy=7)[0]


def sample_uniform(
    params: EllipseParametersAffine,
    angle_range: Tuple[float, float],
    n: int,
):
    """Sample points from angle range by equal distance"""
    (cx, cy), (major, minor), tilt = params
    angle_start, angle_end = angle_range
    assert major >= minor

    focus        = np.sqrt(major * major - minor * minor)
    ecc          = focus / major
    ecc2         = ecc   * ecc
    start_length = ell(angle_start, ecc2)
    end_length   = ell(angle_end,   ecc2)
    l = np.arange(n) + 0.5
    l /= n
    angles = ell_inv(start_length + (end_length - start_length) * l, ecc2)
    rot = np.array((
        (np.cos(tilt), -np.sin(tilt)),
        (np.sin(tilt),  np.cos(tilt)),
    ))
    pts = rot @ np.diag((major, minor)) @ np.stack((np.sin(angles), np.cos(angles)))
    return pts.T + (cx, cy)  # N x 2
