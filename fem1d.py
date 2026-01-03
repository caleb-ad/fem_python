from typing import Callable, Sequence
from scipy.integrate import quad
from scipy.linalg import inv
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
import matplotlib.pyplot as plt

BasisFunc = list[Callable[[float], float]] #function and its derivatives
type Matrix[T] = Sequence[Sequence[T]]

@dataclass
class Basis1D:
    f: list[BasisFunc]

    @staticmethod
    def linear_basis(x0: float, x1: float) -> Basis1D:
        _1 = [lambda x, _x0=x0, _x1=x1: (x - _x1) / (_x0 - _x1), lambda x, _x0=x0, _x1=x1: 1 / (_x0 - _x1)]
        _2 = [lambda x, _x0=x0, _x1=x1: (x - _x0) / (_x1 - _x0), lambda x, _x0=x0, _x1=x1: 1 / (_x1 - _x0)]
        return Basis1D([_1, _2])

BilinearForm = Callable[[BasisFunc, BasisFunc, float, float], float]

@dataclass
class Element1D:
    x0: float
    x1: float
    n0: int #global node
    n1: int #global node
    basis: Basis1D
    op: BilinearForm

    def element_matrix(self) -> Matrix[float]:
        M = []
        for f2 in self.basis.f:
            row = []
            for f1 in self.basis.f:
                val = self.op(f1, f2, self.x0, self.x1)
                row.append(val)
            M.append(row)
        return M


def assemble_1d(e: list[Element1D]) -> Matrix[float]:
    M = []
    element_matrices = [e_.element_matrix() for e_ in e]
    #left boundary
    row = [element_matrices[0][0][0], element_matrices[0][0][1]] + [0 for _ in range(len(e) - 1)]
    M.append(row)

    #interior
    for e_ in range(0, len(e) - 1):
        row = [0.0 for _ in range(e_)]
        row.append(element_matrices[e_][1][0])
        row.append(element_matrices[e_][0][0] + element_matrices[e_ + 1][1][1])
        row.append(element_matrices[e_ + 1][0][1])
        row += [0.0 for _ in range(len(e) - e_ - 2)]
        M.append(row)

    #right boundary
    row = row = [0.0 for _ in range(len(e) - 1)] + [element_matrices[-1][1][0], element_matrices[-1][1][1]]
    M.append(row)

    return M


def result_1d(x: float, u: list[float], elements: list[Element1D]) -> float:
    element_coeffs = zip(elements, zip(u[:-1], u[1:]))
    for e, (u0, u1) in element_coeffs:
        if x <= e.x1 and x >= e.x0:
            return e.basis[0](x) * u0 + e.basis[1](x) * u1
    raise ValueError


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    L = 2 * np.pi #length of section
    N = 1000 #number of elements
    k = 1
    points = np.linspace(0, L, N + 1)
    def a(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
        return quad(lambda x: u[1](x)*v[1](x) + k*k*u[0](x)*v[0](x), a, b)[0]
    elements = [Element1D(x0, x1, i, i+1, Basis1D.linear_basis(x0, x1), a) for (i, (x0, x1)) in enumerate(zip(points[0:-1], points[1:]))]
    M = assemble_1d(elements)
    print(np.array(M))

    b = [-2] + [0 for _ in range(N - 1)] + [2]
    u = inv(M) @ b

    domain = np.linspace(0, L, 3000)
    fig, axs =plt.subplots()
    axs.plot(domain, [result_1d(x, u, elements) for x in domain])
    # axs.plot(domain, [result_1d(x, [pow(-1, i) for i in range(11)], elements) for x in domain])
    plt.show()

