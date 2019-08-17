"""Bias / RMS error comparison between some fitting methods"""
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .elltools import (EllipseParametersAffine, EllipseParametersAlgebraic,
                       affine2algebraic, sample_uniform)
from .fit import dlt, fit


def normalize_param(u: EllipseParametersAlgebraic) -> EllipseParametersAlgebraic:
    """Only used for assessing and testing"""
    u = u / np.linalg.norm(u)
    if u[0] < 0:
        u *= -1
    return u


def benchmark_fit(xy0: np.ndarray, u0: EllipseParametersAlgebraic, f0: float) -> None:
    """
    Run benchmark for ellipse fitting

    Args:
        xy0 (np.ndarray): ground truth points of shape (N, 2)
        u0 (np.ndarray):  ground truth ellipse parameter of shape (6,)
        f (float):        Normalizer of same order as xy0
    """
    # input sanity check
    assert xy0.ndim == 2
    assert xy0.shape[-1] == 2
    assert u0.ndim == 1
    assert u0.shape == (6,)

    np.testing.assert_almost_equal(dlt(xy0, f0=f0) @ u0, 0)
    # input sanity check done

    methods = {
        'hyper': ('hyper', 1),
        'LS': ('least-square', 1),
        'taubin': ('taubin', 1),
        'hyper-renorm': ('hyper', 100),
        'LS-renorm': ('least-square', 100),
        'taubin-renorm': ('taubin', 100),
    }
    result = {s:[] for s in methods}

    N_TRIAL = 10_000
    sigmas = (.5 + np.arange(80)) / 100
    # sigmas = (.5 + np.arange(1)) / 10
    for sigma in (pbar_sigma := tqdm(sigmas)):
        for label, (method, n_max_iter) in (pbar_method := tqdm(methods.items(), leave=False)):
            pbar_sigma .set_description(f"sigma={sigma:.2f}")
            pbar_method.set_description(f"method={method}-{n_max_iter}")

            # calculate bias and rms for given sigma
            t_start = time.time()
            errs = []
            for _ in tqdm(range(N_TRIAL), leave=False):
                dxy = sigma * np.random.normal(size=xy0.shape)
                u = fit(xy0 + dxy, f0, method, sanity_check=True, test=False, n_max_iter=n_max_iter)
                err = (np.identity(6) - u0[:, None] @ u0[None, :]) @ normalize_param(u)
                errs.append(err)
            bias = np.linalg.norm(np.mean(errs, axis=0))
            rms = np.mean(np.linalg.norm(errs, axis=1), axis=0)
            duration = time.time() - t_start
            # calculation done

            result[label].append({
                "sigma": sigma,
                "bias": bias,
                "rms": rms,
                "time": duration / N_TRIAL,
            })

    # plot
    fig, ((lu, ru), (ld, rd)) = plt.subplots(2, 2)

    # Left up
    handles = []
    for label in methods.keys():
        xs, ys = list(zip(*[(r["sigma"], r["bias"]) for r in result[label]]))
        handles += lu.plot(xs, ys)
    lu.set_xlabel("sigma")
    lu.set_ylabel("bias")
    lu.set_ylim(0, 0.1)
    lu.legend(handles, methods.keys())

    # Right up
    handles = []
    for label in methods.keys():
        xs, ys = list(zip(*[(r["sigma"], r["rms"]) for r in result[label]]))
        handles += ru.plot(xs, ys)
    ru.set_xlabel("sigma")
    ru.set_ylabel("rms")
    ru.set_ylim(0, 0.3)
    ru.legend(handles, methods)

    # Left down
    handles = []
    for label in methods.keys():
        xs, ys = list(zip(*[(r["sigma"], r["time"]) for r in result[label]]))
        handles += ld.plot(xs, ys)
    ld.set_xlabel("sigma")
    ld.set_ylabel("time")
    ld.legend(handles, methods)

    fig.show()
    input("Done. Enter to finish")
    fig.savefig("bias_and_rms.png")
    # plot done


if __name__ == '__main__':
    def main():
        """Entry point"""
        # Constants
        n  = 30
        f0 = 600
        ellipse_params: EllipseParametersAffine = ((0, 0), (100, 50), 0)
        phi_range = (0, np.pi / 2)

        # Elliptic parameters
        u0 = affine2algebraic(ellipse_params, f0=f0)
        u0 = normalize_param(u0)

        # Generate ground truth points
        xy0 = sample_uniform(ellipse_params, phi_range, n=n)

        benchmark_fit(xy0, u0, f0=f0)
    main()
