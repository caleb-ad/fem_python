from typing import Callable, Sequence, Any
from scipy.integrate import quad
from scipy.linalg import inv
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
import matplotlib.pyplot as plt
import enum

BasisFunc = Sequence[Callable[[float], float]] # function and its derivatives
ForcingFunc = Callable[[float], float]
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
class GlobalNode:
    idx: int
    is_boundry: bool

    @staticmethod
    def from_ints(ints: list[int]) -> list[GlobalNode]:
        return [GlobalNode(idx=i, is_boundry=False) for i in ints]

@dataclass
class Element1D:
    x0: float
    x1: float
    n: list[GlobalNode] #global nodes
    basis: Basis1D

    # computes the matrix coefficients corresponding to `self.basis` under `op`
    def element_matrix(self, op: BilinearForm) -> Matrix[float]:
        M = []
        for f2 in self.basis.f:
            row = []
            for f1 in self.basis.f:
                val = op(f1, f2, self.x0, self.x1)
                row.append(val)
            M.append(row)
        return M

    # compute the inner product of each basis function and `f`
    def inner_product(self, f: ForcingFunc) -> list[float]:
        return [quad(lambda x: bf[0](x)*f(x), self.x0, self.x1)[0] for bf in self.basis.f]

    def order(self) -> int:
        return len(self.basis.f)

class BoundryType(enum.Enum):
    Homogenous = enum.auto
    Dirichlet = enum.auto
    Mixed = enum.auto

@dataclass
class Boundry1D:
    btype: BoundryType
    vals: list[float]

    # true if boundry adds conditions to the function
    def is_func_condition(self) -> bool:
        return  (self.btype == BoundryType.Dirichlet or self.btype == BoundryType.Homogenous)

    # true if boundry adds conditions to the functions deriviative
    def is_deriv_condition(self) -> bool:
        return (self.btype == BoundryType.Mixed) and len(self.vals) == 3

class BoundryValueProblem1D:
    M: Matrix[float]
    elements: Sequence[Element1D]
    b: Sequence[float]
    op: BilinearForm

    def __init__(self, op: BilinearForm, elements: Sequence[Element1D], f: ForcingFunc, boundary: Boundry1D):
        self.M = [[0.0 for _ in range(len(elements) + 1)] for _ in range(len(elements) + 1)]
        self.b = [0.0 for _ in range(len(elements) + 1)]
        self.op = op
        for e in elements:
            # compute matrix elements
            ematrix = e.element_matrix(self.op)
            for i in range(e.order()):
                for j in range(e.order()):
                    if e.n[i].is_boundry and boundary.is_func_condition():
                        continue
                    self.M[e.n[i].idx][e.n[j].idx] += ematrix[i][j]

            # compute forcing element
            products = e.inner_product(f)
            for i in range(e.order()):
                self.b[e.n[i].idx] += products[i]

        # modify `b` to implement general Dirichlet condition
        if boundary.btype == BoundryType.Dirichlet:
            ematrix_0_1_0 = self.op(elements[0].basis.f[0], elements[0].basis.f[1], elements[0].x0, elements[0].x1)
            self.b[elements[0].n[1].idx] -= boundary.vals[0] * ematrix_0_1_0
            ematrix_N_0_1 = self.op(elements[-1].basis.f[1], elements[-1].basis.f[0], elements[-1].x0, elements[-1].x1)
            self.b[elements[-1].n[0].idx] -= boundary.vals[1] * ematrix_N_0_1
        # modify `b` and `M` to implement Neumann conditions
        elif boundary.btype == BoundryType.Mixed:
            self.b[elements[0].n[0].idx] -= boundary.vals[0] * elements[0].basis.f[0][0](0)
            self.b[elements[-1].n[1].idx] += boundary.vals[1] * elements[-1].basis.f[1][0](0)
            self.M[elements[0].n[0].idx][elements[0].n[0].idx] -= boundary.vals[2] * elements[0].basis.f[0][0](0)
            self.M[elements[-1].n[1].idx][elements[-1].n[1].idx] += boundary.vals[2] * elements[-1].basis.f[1][0](0)

        self.elements = elements
        self.b = [-3.0] + [0.0 for _ in range(N - 1)] + [3.0]

    # solve for coefficients
    def solve(self) -> Sequence[float]:
        return inv(self.M) @ self.b

    # construct solution function from coefficients
    def result(self, x: float, u: Sequence[float]) -> float:
        for e in elements:
            if x <= e.x1 and x >= e.x0:
                return sum([f[0](x) * u[n.idx] for (f, n) in zip(e.basis.f, e.n)])
        raise ValueError


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    L = 2 * np.pi #length of section
    N = 500 #number of elements
    k = 1
    points = np.linspace(0, L, N + 1)
    def a(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
        return quad(lambda x: u[1](x)*v[1](x) + k*k*u[0](x)*v[0](x), a, b)[0]
    elements = [Element1D(x0, x1, GlobalNode.from_ints([i, i+1]), Basis1D.linear_basis(x0, x1)) for (i, (x0, x1)) in enumerate(zip(points[0:-1], points[1:]))]
    elements[0].n[0].is_boundry = True
    elements[-1].n[1].is_boundry = True

    problem = BoundryValueProblem1D(a, elements, lambda x: 0.0, Boundry1D(BoundryType.Dirichlet, [1,0]))
    print(np.array(problem.M))

    u = problem.solve()

    domain = np.linspace(0, L, 3000)
    fig, axs =plt.subplots()
    axs.plot(domain, [problem.result(x, u) for x in domain])
    # axs.plot(domain, [result_1d(x, [pow(-1, i) for i in range(11)], elements) for x in domain])
    plt.show()

