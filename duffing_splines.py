#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.interpolate
import scipy.stats
import scipy.misc
import matplotlib.pyplot as plt
import continuation
import discretise

STARTER_PARAMS = [0.5, 0.51]
KP = 1
INTEGRATION_TIME = 30
TRANSIENT_TIME = 100
N_SAMPLES = INTEGRATION_TIME * 10
###
DSIZE = 5
STEPSIZE = 1
FINITE_DIFFERENCES_STEPSIZE = 0.2


# Things to do:
# Understand why it's jumping!
# Try different numerical solver?


def blackbox_system(control_target, parameter):
    """ TODO comments """

    def duffing(x, t, omega, control_target, kp):
        """Defines the RHS of the model"""
        gamma = 1
        alpha = 1
        beta = 0.04
        delta = 0.1
        v, w = x  # Unpack state
        if control_target is not None:
            control_action = kp * (control_target(t) - v)
        else:
            control_action = 0
        vdot = w
        wdot = (
            gamma * np.cos(omega * t)
            - beta * v ** 3
            - alpha * v
            - delta * w
            + control_action
        )
        return [vdot, wdot]

    t_span = [0, TRANSIENT_TIME + INTEGRATION_TIME]
    t_eval = np.linspace(TRANSIENT_TIME, TRANSIENT_TIME + INTEGRATION_TIME, N_SAMPLES)
    if control_target is not None:
        # Speed up transient decay
        y0 = [control_target(0), control_target(0)]
    else:
        y0 = [1, 1]  # Arbitrarily
    soln = scipy.integrate.solve_ivp(
        lambda t, x: duffing(x, t, parameter, control_target, KP),
        t_span=t_span,
        t_eval=t_eval,
        y0=y0,
    )
    return np.vstack((soln.t, soln.y[0]))


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
    ts = np.linspace(0, solution.period/2, 1000)
    ys = solution.control_target(ts)
    return (np.max(ys) - np.min(ys)) / 2


class DuffingContinuation(continuation.AutonymousCBC):
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

    par_0, par_1 = STARTER_PARAMS
    signal_0 = blackbox_system(None, par_0)
    signal_1 = blackbox_system(None, par_1)
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
