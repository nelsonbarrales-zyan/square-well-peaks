# -*- coding: utf-8 -*-
"""
casos.py -- lista unica de casos de prueba para los tres programas.

Q_alpha = alpha/pi + 1/2, con alpha = sqrt(2|V0|) a (profundidad adimensional del pozo).
La parte entera de Q_alpha es el numero de estados acotados del pozo cuadrado.

Los valores son arbitrarios: no estan puestos cerca de un umbral (Q_alpha entero),
ni de casos especiales. Cubren de 1 a 20 estados acotados. Para probar otros pozos,
cambia esta lista; los tres programas la leen desde aqui.
"""
Q_CASOS = [1.70, 2.35, 3.60, 5.15, 8.45, 12.80, 20.30]
