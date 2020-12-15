#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
import collocation

MESH_SIZE = 3
POLYNOMIAL_ORDER = 3
INITIAL_PARS = (1, 2)
PAR_RANGE = [1, 10]


def estimate_initial_continuation_vector(continuation_obj, parameter, period, initial_cond):
    """
    Given a parameter value, an initial condition assumed to be
    close to the limit cycle for that parameter value, and an
    estimate of the period of that cycle, construct a continuation
    vector that encodes these assumptions. This is done by
    simulating the system from the stated initial condition, for
    the stated time period. The output is sampled on each of the
    meshpoints. As we are using interpolating polynomials as basis
    functions, the appropriate coefficients are given by the
    trajectory values at the meshpoints. These state values are
    then reshaped to get the continuation vector.
    """
    def ode(t, y):
        return continuation_obj.ode_RHS(y, parameter)
    soln = scipy.integrate.solve_ivp(ode, [0, period], initial_cond, dense_output=True).sol
    coeffs_list = soln(period * continuation_obj.col_func.full_mesh).T
    target_shape = (continuation_obj.col_func.n_subintervals, -1, coeffs_list.shape[1])
    coeffs_list.reshape(target_shape)
    return np.hstack((parameter, period, coeffs_list.reshape(-1)))


def get_starters(guess_vec_1, guess_vec_2, continuer):
    continuation_vecs = [guess_vec_1]
    for i, vec in enumerate((guess_vec_1, guess_vec_2)):
        # Use previous continuation vector as a phase reference and initial condition
        last_vec = continuation_vecs[i]
        last_coefficients = continuer.get_discretisation(last_vec)
        par = continuer.get_parameter(vec)

        def v_dot(t):
            return continuer.col_func.derivative(last_coefficients, t)

        def f(v):
            continuation_vec = np.hstack((par, v))
            return continuer.collocation_system(continuation_vec, v_dot)

        # Refine guess with Newton corrector
        initial_guess = last_vec[1:]
        soln = np.hstack((par, scipy.optimize.root(f, initial_guess).x))
        continuation_vecs.append(soln)
    # First entry was our initial guess, so drop it
    return continuation_vecs[1:]


def hopf(state, par, stable_LC=True):
    sigma = -1 if stable_LC else 1
    x, y = state.T
    x_dot = par * x - y + sigma * x * (x**2 + y**2)
    y_dot = x + par * y + sigma * y * (x**2 + y**2)
    return np.vstack((x_dot, y_dot)).T


def solver(sys, x0):
    solution = scipy.optimize.root(sys, x0)
    print("Solution vector: ", solution.x)
    print("Solution value: ", solution.fun)
    print("Parameter: ", solution.x[0:1])
    if solution.success:
        return solution.x
    return None


def main():
    krogh = collocation.KroghMesh(MESH_SIZE, POLYNOMIAL_ORDER)
    continuer = collocation.NumericalContinuation(hopf, krogh)

    initial_vec_1 = estimate_initial_continuation_vector(continuer, INITIAL_PARS[0], 6.2, [0, np.sqrt(INITIAL_PARS[0])])
    initial_vec_2 = estimate_initial_continuation_vector(continuer, INITIAL_PARS[1], 6.2, [0, np.sqrt(INITIAL_PARS[1])])
    starters = get_starters(initial_vec_1, initial_vec_2, continuer)
    print("Starters:")
    print(starters)

    continuation_vectors, _ = continuer.run_continuation(starters, solver, par_range=PAR_RANGE)

    fig, ax = plt.subplots()
    ts = np.linspace(0, 1)
    for vec in continuation_vectors:
        coeffs = continuer.get_discretisation(vec)
        par = continuer.get_parameter(vec)
        states = continuer.col_func.eval(coeffs, ts).T
        ax.plot(states[0], states[1], label=par)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
