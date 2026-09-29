# -*- coding: utf-8 -*-
"""
test_validacion.py -- pruebas del repositorio.

Uso:   pip install pytest  &&  pytest -v
       (o:  python test_validacion.py)

  1) Alphas_chek.py: la raiz de tan(alpha) = alpha cae en su intervalo, anula la ecuacion
     y reproduce la primera raiz conocida (4.4934094579...).
  2) AnchosNewForm.py, formula de l(k): coincide con l = 2 dphi/dk calculada por
     diferencias finitas sobre la fase (calculo independiente).
  3) Estados acotados: hay floor(Q_alpha) por caso, cada uno anula la ecuacion
     x cos x + sqrt(alpha^2 - x^2) sin x = 0 y k_bound_imag = sqrt(alpha^2 - x^2)/L.
  4) Picos: ordenados en k, con el valor de la funcion en el pico igual al recalculado, y
     anchos a media prominencia positivos.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from Alphas_chek import alpha_critico, f      # noqa: E402
from casos import Q_CASOS                     # noqa: E402

L = 2.5


@pytest.fixture(scope="session")
def salida(tmp_path_factory):
    d = tmp_path_factory.mktemp("salida")
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(AQUI / "AnchosNewForm.py")], cwd=d, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    base = d / "salida_Qalpha"
    return {"picos": pd.read_csv(base / "picos_por_funcion.csv"),
            "estados": pd.read_csv(base / "estados_acotados.csv")}


@pytest.fixture(scope="session")
def modulo(tmp_path_factory):
    d = tmp_path_factory.mktemp("import")
    cwd = os.getcwd()
    os.chdir(d)                       # el modulo crea salida_Qalpha/ al importarse
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import AnchosNewForm as m
    finally:
        os.chdir(cwd)
    return m


# ---------------------------------------------------------------- 1) Alphas_chek
def test_alpha_critico_en_su_intervalo_y_anula_la_ecuacion():
    for N in range(2, 12):
        alpha = alpha_critico(N)
        assert (N - 1) * np.pi < alpha < (N - 0.5) * np.pi
        assert abs(f(alpha)) / alpha < 1e-9


def test_alpha_critico_primera_raiz_conocida():
    assert alpha_critico(2) == pytest.approx(4.493409457909064, abs=1e-12)


# ---------------------------------------------------------------- 2) l(k) vs 2 dphi/dk
@pytest.mark.parametrize("Q", Q_CASOS)
def test_ell_coincide_con_derivada_de_la_fase(modulo, Q):
    alpha = modulo.alpha_from_Qalpha(Q)
    V2 = modulo.V2_from_alpha_L(alpha, L)

    def phi(k):
        q = np.sqrt(2 * V2 + k**2)
        return np.arctan((k / q) * np.tan(q * L))

    ks = np.linspace(0.3, 4.0, 60)
    k_x, funcs = modulo.process_dispersion_functions(alpha, L, ks * L)
    h = 1e-6
    d = phi(ks + h) - phi(ks - h)
    d = (d + np.pi / 2) % np.pi - np.pi / 2          # quita el salto de rama de arctan
    ell_num = 2 * d / (2 * h)
    ok = np.isfinite(funcs["lk"])
    assert np.allclose(funcs["lk"][ok], ell_num[ok], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- 3) estados acotados
def test_estados_acotados(salida):
    e = salida["estados"]
    for Q in Q_CASOS:
        g = e[np.isclose(e.Qalpha, Q)].sort_values("bound_order")
        alpha = np.pi * (Q - 0.5)
        assert len(g) == int(np.floor(Q))
        x = g.xq_bound.to_numpy()
        assert (np.diff(x) > 0).all() and (x < alpha).all()
        res = x * np.cos(x) + np.sqrt(alpha**2 - x**2) * np.sin(x)
        assert np.abs(res).max() < 1e-8
        assert np.allclose(g.k_bound_imag, np.sqrt(alpha**2 - x**2) / L, rtol=1e-9)


# ---------------------------------------------------------------- 4) picos y anchos
def test_picos(salida, modulo):
    p = salida["picos"]
    assert set(p.Qalpha.round(6)) == set(np.round(Q_CASOS, 6))
    for _, g in p.groupby(["Qalpha", "funcion"]):
        g = g.sort_values("peak_order")
        assert (np.diff(g.k_peak) > 0).all()
    w = p.width_k_halfprom.dropna()
    assert (w > 0).all()
    # el valor de la funcion en cada pico coincide con el recalculado en k_peak
    for (Q, fn), g in p.groupby(["Qalpha", "funcion"]):
        alpha = modulo.alpha_from_Qalpha(Q)
        k, funcs = modulo.process_dispersion_functions(alpha, L, g.k_peak.to_numpy() * L)
        assert np.allclose(funcs[fn], g.y_peak.to_numpy(), rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
