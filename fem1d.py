from typing import Callable

class Element1D:
    def __init__(self):
        self.x0
        self.x1

type BasisFunc = Callable[float, [float]]

def linear_basis(e: list[Element1D]) -> list[(BasisFunc, BasisFunc)]:
    return list(map(lambda _e: (lambda x, x0=_e.x0, x1=_e.x1: (x - x0) / (x1 - x0), lambda x, x0=_e.x0, x1=_e.x1: (x - x1) / (x0 - x1)), e))


