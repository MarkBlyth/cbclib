#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.stats
import scipy.misc
import matplotlib.pyplot as plt
import continuation
import discretise

STARTER_PARAMS = [0.5, 0.51]
N_HARMONICS = 3
KP = 1
INTEGRATION_TIME = 30
TRANSIENT_TIME = 100
###
N_SAMPLES = INTEGRATION_TIME * 10
STEPSIZE = 0.2
FINITE_DIFFERENCES_STEPSIZE = 0.1


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


def build_continuation_vector(signal, discretisor, parameter):
    """ TODO comments """
    period = 2 * np.pi / parameter
    discretisation = discretisor.discretise(signal, period)
    return np.hstack((parameter, discretisation))


def get_amplitude(continuation_solution):
    """ TODO comments """
    a1 = continuation_solution.discretisation[1]
    b1 = continuation_solution.discretisation[(len(continuation_solution.discretisation) + 1) // 2]
    f0_amplitude = np.sqrt(a1 ** 2 + b1 ** 2)
    return f0_amplitude


class DuffingContinuation(continuation.NonautonymousCBC):
    """ TODO comments """

    def get_period(self, continuation_vec):
        return 2 * np.pi / continuation_vec[0]


def main():
    discretisor = discretise.FourierDiscretisor(N_HARMONICS)
    continuer = DuffingContinuation(blackbox_system, discretisor)

    par_0, par_1 = STARTER_PARAMS
    signal_0 = blackbox_system(None, par_0)
    signal_1 = blackbox_system(None, par_1)
    starters = [
        build_continuation_vector(signal_0, discretisor, par_0),
        build_continuation_vector(signal_1, discretisor, par_1),
    ]

    solver = lambda sys, x0: continuation.newton_solver(
        sys, x0, finite_differences_stepsize=FINITE_DIFFERENCES_STEPSIZE
    )
    continuation_solutions, message = continuer.run_continuation(
        starters, solver=solver, stepsize=STEPSIZE, par_range=[0.5, 2]
    )
    print(message)

        # Plot results
    fig, ax = plt.subplots()
    ax.plot(
        [s.parameter for s in continuation_solutions],
        [get_amplitude(s) for s in continuation_solutions],
        color="k",
    )
    ax.set_xlabel("Forcing frequency")
    ax.set_ylabel("Response amplitude")
    plt.show()


if __name__ == "__main__":
    main()
