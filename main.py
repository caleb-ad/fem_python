from fem1d import *

def propagating(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
    k = 1
    return quad(lambda x: u[1](x)*v[1](x) - k*k*u[0](x)*v[0](x), a, b)[0]

def attenuating(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
    k = 1
    return quad(lambda x: u[1](x)*v[1](x) + k*k*u[0](x)*v[0](x), a, b)[0]

def inhomogenous(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
    def k(x):
        return 10*x
    return quad(lambda x: u[1](x)*v[1](x) - k(x)*k(x)*u[0](x)*v[0](x), a, b)[0]

def nonlinear(u: BasisFunc, v: BasisFunc, a: float, b: float) -> float:
    def k(x):
        return np.sin(x)
    return quad(lambda x: u[1](x)*v[1](x) - k(u[0](x))*k(u[0](x))*u[0](x)*v[0](x), a, b)[0]


def main():
    np.set_printoptions(precision=3)
    L = 7 * np.pi #length of section
    N = 5000 #number of elements
    points = np.linspace(0, L, N + 1)

    elements = [Element1D(x0, x1, GlobalNode.from_ints([i, i+1]), Basis1D.linear_basis(x0, x1)) for (i, (x0, x1)) in enumerate(zip(points[0:-1], points[1:]))]
    elements[0].n[0].is_boundary = True
    elements[-1].n[1].is_boundary = True

    # problem = BoundaryValueProblem1D(a, elements, lambda x: 0.0, Boundary1D(BoundaryType.Dirichlet, [1, 1]))
    propagating_problem = BoundaryValueProblem1D(propagating, elements, lambda x: 5 * np.exp(-(x-L/2)**2), Boundary1D(BoundaryType.Neumann, [1, -1, 0]))
    attenuating_problem = BoundaryValueProblem1D(attenuating, elements, lambda x: 5 * np.exp(-(x-L/2)**2), Boundary1D(BoundaryType.Neumann, [1, -1, 0]))
    dispersive_problem = BoundaryValueProblem1D(nonlinear, elements, lambda x: 5 * np.exp(-(x-L/2)**2), Boundary1D(BoundaryType.Neumann, [1, -1, 0]))
    inhomogenous_problem = BoundaryValueProblem1D(inhomogenous, elements, lambda x: 5 * np.exp(-(x-L/2)**2), Boundary1D(BoundaryType.Neumann, [1, -1, 0]))

    u1 = propagating_problem.solve()
    u2 = attenuating_problem.solve()
    u3 = dispersive_problem.solve()
    u4 = inhomogenous_problem.solve()

    domain = np.linspace(0, L, 3000)
    fig, axs =plt.subplots()
    axs.plot(domain, [propagating_problem.result(x, u1) for x in domain])
    axs.plot(domain, [attenuating_problem.result(x, u2) for x in domain])
    axs.plot(domain, [dispersive_problem.result(x, u3) for x in domain])
    axs.plot(domain, [inhomogenous_problem.result(x, u4) for x in domain])
    # axs.plot(domain, [result_1d(x, [pow(-1, i) for i in range(11)], elements) for x in domain])
    plt.show()


if __name__ == "__main__":
    main()
