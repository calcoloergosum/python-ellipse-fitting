"""Main routine; see `fit`"""
from typing import List

import numpy as np
import scipy.linalg as LA

from .elltools import EllipseParametersAlgebraic


def bias_cancel(xi, xixi, mat_pinv, v0, weight):
    """Bias cancelling term introduced by Kanatani 2009"""
    e = np.array((1, 0, 1, 0, 0, 0))
    ret = (weight[:, None, None] * symmetrize(xi[:, :, None] @ e[None, None, :])).sum(axis=0)
    ret -= (weight[:, None, None] * weight[:, None, None] * (
        np.trace(mat_pinv[None, :, :] @ v0, 0, 1, 2)[:, None, None] * xixi + \
        (xi[:, None, :] @ mat_pinv[None] @ xi[:, :, None]) * v0 + \
        symmetrize(v0 @ mat_pinv[None] @ xixi)
    )).sum(axis=0)
    return ret


def dlt(pts: np.ndarray, f0: float = 1) -> np.ndarray:
    """
    Direct Linear Transform; (x,y) |-> (xx, 2xy, yy, 2fx, 2fy, ff)

    >>> dlt(np.array([
    ...    (0, 1),
    ...    (1, 2),
    ...    (2, 3),
    ...    (3, 4),
    ... ]))
    array([[ 0.,  0.,  1.,  0.,  2.,  1.],
           [ 1.,  4.,  4.,  2.,  4.,  1.],
           [ 4., 12.,  9.,  4.,  6.,  1.],
           [ 9., 24., 16.,  6.,  8.,  1.]])
    """
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    xs, ys = pts.T
    return np.stack(
        (
            xs * xs,
            2 * xs * ys,
            ys * ys,
            2 * xs * f0,
            2 * ys * f0,
            f0 * f0 * np.ones(len(pts)),
        ),
        axis=1,
    )  # of shape (N, 6)


def symmetrize(mat):
    """M + M.T"""
    assert mat.ndim in (2, 3)
    assert mat.shape[-2] == mat.shape[-1]
    return mat + np.swapaxes(mat, -2, -1)


def pinv(mat, sanity_check: bool = False):
    """Truncated pseudo-inverse of rank 5
    Example:
        >>> M = np.array((
        ...     (1, 0,),
        ...     (0, 1,),
        ...     (0, 0,),
        ... ))
        >>> y = np.array((1, 1, 1))
        >>> M_inv = np.linalg.pinv(M)
        >>> M_inv @ y
        array([1., 1.])

    # >>> xi0 = dlt(xy0)
    # >>> M0 = (xi0[:, :, None] @ xi0[:, None, :]).mean(axis=0)
    # >>> M0_pinv = pinv(M0, sanity_check=True)
    # >>> np.testing.assert_allclose(M0_pinv @ M0 @ M0_pinv, M0_pinv)
    # >>> np.testing.assert_allclose(M0 @ M0_pinv @ M0, M0)
    """
    # return np.linalg.inv(mat)
    vals, vecs = np.linalg.eig(mat)
    i_min = np.argmin(np.abs(vals))  # should be positive anyway as symmetric matrix
    vals5 = np.delete(vals, i_min, 0)
    vecs5 = np.delete(vecs, i_min, 1)  # 6x5
    inv = vecs5 @ np.diag(1/vals5) @ vecs5.T

    if sanity_check:
        # calculation sanity check
        np.testing.assert_almost_equal(vecs5.T @ vecs5, np.identity(5))
        np.testing.assert_allclose(mat, vecs @ np.diag(vals) @ vecs.T)
    return inv


def taubin_constraint(xys, weight, f0: float):
    """Taubin constraint. Estimated expected deviation of DLT vector in each points"""
    _0 = np.zeros(len(xys))
    _1 = np.ones (len(xys))
    x, y = xys.T
    xx, xy, yy = x*x, x*y, y*y
    x = x * f0
    y = y * f0
    _ff = f0*f0*_1
    v0 = 4 * np.array((
        ( xx,    xy, _0,   x,  _0, _0,),
        ( xy, xx+yy, xy,   y,   x, _0,),
        ( _0,    xy, yy,  _0,   y, _0,),
        (  x,     y, _0, _ff,  _0, _0,),
        ( _0,     x,  y,  _0, _ff, _0,),
        ( _0,    _0, _0,  _0,  _0, _0,),
    )).T
    return (weight[:, None, None] * v0).sum(axis=0), v0


def available_methods() -> List[str]:
    """Enumerates all available methods"""
    return [
        {
            "name": "hyper",
            "description": "Kanatani 2010; Kanatani 2014",
        },{
            "name": "taubin",
            "description": "Taubin 1991",
        },{
            "name": "least-square",
            "description": "Minimize algebraic distance",
        },{
            "name": "force-ellipse",
            "description": "Fitzgibbon 1999; Forces ellipse",
        }
    ]


def fit(xys: np.ndarray, f0: float, method: str, sanity_check: bool = False, test: bool = False,
    n_max_iter: int = 1,
) -> EllipseParametersAlgebraic:
    """
    Fit ellipse

    Args:
        xys (np.ndarray):              Points of shape (N, 2)
        f (float):                     Normalization term, of order as in xys
        method (str):                  one of 'hyperaccuracy, taubin, direct, force-ellipse'
        sanity (bool, optional):         _description_. Defaults to False.
        test (bool, optional):         _description_. Defaults to False.

    Raises:
        KeyError: on unknown method

    Returns:
        _type_: _description_
    """
    assert xys.ndim == 2
    assert xys.shape[-1] == 2
    xi = dlt(xys, f0=f0)

    # Make design matrix
    weight = np.ones(len(xys))
    weight /= weight.sum()
    xixi = xi[:, :, None] @ xi[:, None, :]

    # solution from Kanatani
    if method == 'hyper':
        def get_mat_const():
            nonlocal xys, weight
            mat_const_taubin, v0 = taubin_constraint(xys, weight, f0)
            mat_pinv = pinv(mat, sanity_check)
            mat_const = mat_const_taubin + bias_cancel(xi, xixi, mat_pinv, v0, weight)
            return mat_const, v0

    elif method == 'taubin':
        def get_mat_const():
            nonlocal xys, weight
            mat_const, v0 = taubin_constraint(xys, weight, f0)
            return mat_const, v0

    elif method == 'least-square':
        _mat_const = np.identity(6)
        def get_mat_const():
            nonlocal weight
            _, v0 = taubin_constraint(xys, weight, f0)
            return _mat_const, v0

    elif method == 'force-ellipse':
        _mat_const = np.array((
            ( 0, 0, -1, 0, 0, 0),
            ( 0, 2,  0, 0, 0, 0),
            (-1, 0,  0, 0, 0, 0),
            ( 0, 0,  0, 0, 0, 0),
            ( 0, 0,  0, 0, 0, 0),
            ( 0, 0,  0, 0, 0, 0),
        ))
        def get_mat_const():
            nonlocal weight
            _, v0 = taubin_constraint(xys, weight, f0)
            return _mat_const, v0

    else:
        raise KeyError(f"Unknown method {method}")

    u = 0
    u_new = 1
    n_iter = 0
    while not (np.allclose(u, u_new) or np.allclose(u, -u_new)):  # while not same up to sign
        mat = (weight[:, None, None] * xixi).sum(axis=0)
        mat_const, v0 = get_mat_const()

        # update solution
        u = u_new
        vals, vecs = LA.eig(mat, mat_const)
        i = np.argmin(np.abs(vals))
        u_new = vecs[:, i]
        n_iter += 1
        if n_iter >= n_max_iter:
            break
        weight = 1 / (u_new @ v0 @ u_new)
        weight /= weight.sum()
    return u_new
