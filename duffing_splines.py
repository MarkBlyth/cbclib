#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import scipy.misc
import matplotlib.pyplot as plt
import continuation
import discretise
import system

STARTER_PARAMS = [0.5, 0.51]
KP = 1
EVALUATION_TIME = 30
TRANSIENT_TIME = 100
###
DSIZE = 5
STEPSIZE = 1
FINITE_DIFFERENCES_STEPSIZE = 0.2

def duffing(t, x, omega):
    """Defines the RHS of the model"""
    gamma = 1
    alpha = 1
    beta = 0.04
    delta = 0.1
    v, w = x  # Unpack state
    vdot = w
    wdot = (
        gamma * np.cos(omega * t)
        - beta * v ** 3
        - alpha * v
        - delta * w
    )
    return [vdot, wdot]


def get_analytic_amplitude(
    initial_guess, omega, alpha=1, gamma=1, beta=0.04, delta=0.1
):
    def f(z):
        return (
            (omega ** 2 - alpha - 0.75 * beta * (z ** 2)) ** 2 + (delta * omega) ** 2
        ) * (z ** 2) - gamma ** 2

    def fprime(z):
        return scipy.misc.derivative(f, z, dx=1e-6)

    return scipy.optimize.root_scalar(
        f, fprime=fprime, x0=initial_guess, method="newton"
    ).root


def build_continuation_vector(signal, discretisor, parameter):
    """ TODO comments """
    # TODO replace this with the method defined in NonautonymousCBC
    period = 2 * np.pi / parameter
    discretisation = discretisor.discretise(signal, period)
    return np.hstack((parameter, discretisation))


def get_amplitude(solution):
    """
    Calculate a loose estimate of the signal amplitude. Evaluate the
    undiscretised model at 1000 test points; find the maximum and
    minimum evaluations; return the difference between them. This is
    not an exact amplitude, as there is no guarantee the tested
    timepoints will line up with the maxima and minima; nevertheless,
    it will give a very good estimate.
    """
    ts = np.linspace(0, solution.period, 1000)
    ys = solution.control_target(ts)
    return (np.max(ys) - np.min(ys)) / 2


class DuffingContinuation(continuation.NonautonymousCBC):
    def get_period(self, continuation_vec):
        return 2 * np.pi / continuation_vec[0]


def main():
    solver = lambda sys, x0: continuation.newton_solver(
        sys, x0, finite_differences_stepsize=FINITE_DIFFERENCES_STEPSIZE
    )

    """ SCIPY SOLVER """
    # def solver(sys, x0):
    #     solution = scipy.optimize.root(sys, x0, tol=1e-6)
    #     print("Solution vector: ", solution.x)
    #     print("Solution value: ", solution.fun)
    #     print("Parameter: ", solution.x[0])
    #     if solution.success:
    #         return solution.x
    #     return None

    controller = system.ProportionalController(KP, [1, 0], [0, 1])
    blackbox_system = system.System([1, 0], duffing, TRANSIENT_TIME, EVALUATION_TIME, 10, controller)
    par_0, par_1 = STARTER_PARAMS
    signal_0 = blackbox_system(par_0)
    signal_1 = blackbox_system(par_1)
    # Discretisation size, initial signal, period of initial signal
    discretisor = discretise.SplinesDiscretisor(DSIZE)
    starters = [
        build_continuation_vector(signal_0, discretisor, par_0),
        build_continuation_vector(signal_1, discretisor, par_1),
    ]
    continuer = DuffingContinuation(blackbox_system, discretisor)

    continuation_solutions, message = continuer.run_continuation(
        starters, solver=solver, stepsize=STEPSIZE, par_range=[0.5, 2]
    )
    print(message)

    # Parse output
    parameter_points = [s.parameter for s in continuation_solutions]
    cbc_amplitudes = [get_amplitude(s) for s in continuation_solutions]
    analytic_amplitudes = [
        get_analytic_amplitude(a, w) for a, w in zip(cbc_amplitudes, parameter_points)
    ]

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(parameter_points, cbc_amplitudes, color="k")
    ax.set_xlabel("Forcing frequency")
    ax.set_ylabel("Response amplitude")
    plt.show()

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(
        parameter_points, cbc_amplitudes, marker="x", color="k", label="CBC results",
    )
    ax.plot(
        parameter_points, analytic_amplitudes, marker="x", label="Analytic results",
    )
    ax.legend()
    ax.set_xlabel("Forcing amplitude")
    ax.set_ylabel("Response amplitude")
    plt.show()


if __name__ == "__main__":
    main()
