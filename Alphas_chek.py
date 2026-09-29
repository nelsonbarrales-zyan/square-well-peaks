# -*- coding: utf-8 -*-
"""
Alphas_chek.py -- raiz de tan(alpha) = alpha antes del umbral entero N.

Para N >= 2 hay una raiz en el intervalo ((N-1) pi, (N-1/2) pi). Es la condicion en que
la longitud de dispersion a_s = a tan(alpha)/alpha iguala al radio del pozo (a_s = a).

Uso:
    python Alphas_chek.py            # N = 2, 3, 6, 9 (valores por omision)
    python Alphas_chek.py 4 7 15     # los N que se pidan
"""
import sys
import numpy as np
from scipy.optimize import brentq

N_POR_OMISION = (2, 3, 6, 9)
EPS = 1e-10


def f(alpha):
    return np.tan(alpha) - alpha


def alpha_critico(N):
    """Raiz de tan(alpha) = alpha en ((N-1) pi, (N-1/2) pi)."""
    if N < 2:
        raise ValueError("N debe ser >= 2: para N=1 no hay raiz distinta de cero.")
    a_left = (N - 1) * np.pi + EPS
    a_right = (N - 0.5) * np.pi - EPS
    return brentq(f, a_left, a_right, xtol=1e-14, rtol=1e-14, maxiter=300)


def main(valores):
    for N in valores:
        alpha_crit = alpha_critico(N)
        alpha_hat = alpha_crit / (np.pi / 2)
        Q_alpha = alpha_crit / np.pi + 0.5
        print("N =", N)
        print("alpha_crit =", alpha_crit)
        print("alpha_hat_crit =", alpha_hat)
        print("Q_alpha_crit =", Q_alpha)
        print("check tan(alpha)-alpha =", f(alpha_crit))
        print()


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]] or N_POR_OMISION)
