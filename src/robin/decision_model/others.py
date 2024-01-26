""" Módulo con funciones varias."""
from __future__ import annotations


# FUNCIONES DE NORMA Y T-CONORMA
# Se puede ampliar con más funciones dando más generalidad
def and_t_norma(val1:float, val2:float):
    """
    T-norma del producto
    """
    return val1*val2


def or_t_conorma(val1:float, val2:float):
    """
    T-conorma de la suma
    """
    return val1+val2


funciones = {'&': and_t_norma,
             '|': or_t_conorma}
