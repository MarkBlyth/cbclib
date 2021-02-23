#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import continuation
import discretise
import system

STARTER_PARAMS = [0.5, 0.51]
N_HARMONICS = 3
KP = 1
EVALUATION_TIME = 30
TRANSIENT_TIME = 100
###
STEPSIZE = 0.2
FINITE_DIFFERENCES_STEPSIZE = 0.1


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


def build_continuation_vector(signal, discretisor, parameter):
    """ TODO comments """
    # TODO replace this with the method defined in NonautonymousCBC
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

    controller = system.ProportionalController(KP, [1, 0], [0, 1])
    blackbox_system = system.System([1, 0], duffing, TRANSIENT_TIME, EVALUATION_TIME, 10, controller)
    discretisor = discretise.FourierDiscretisor(N_HARMONICS)
    continuer = DuffingContinuation(blackbox_system, discretisor)

    par_0, par_1 = STARTER_PARAMS
    signal_0 = blackbox_system(par_0)
    signal_1 = blackbox_system(par_1)
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
