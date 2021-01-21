#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.stats
import scipy.misc
import matplotlib.pyplot as plt
import continuation
import discretise

STARTER_PARAMS = [(0.5, 0.51)]
N_HARMONICS = 3
KP = 1
INTEGRATION_TIME = 30
TRANSIENT_TIME = 100
###
N_SAMPLES = INTEGRATION_TIME * 10
STEPSIZE = 0.2
STEP_CONTROL = [0.2, 0.2, 0.2, 0.05, np.inf] ## Initial step; min., max. step; nominal step, contraction
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


def convergence_criteria(y0, y1, system_evaluation):
    """ TODO comments """
    # return np.linalg.norm(y1 - y0) < 1e-9
    return np.linalg.norm(system_evaluation) < 1e-5


def build_continuation_vector(signal, discretisor, parameter):
    """ TODO comments """
    period = 2 * np.pi / parameter
    discretisation = discretisor.discretise(signal, period)
    return np.hstack((parameter, discretisation))


def get_amplitude(discretisation):
    """ TODO comments """
    a1 = discretisation[1]
    b1 = discretisation[(len(discretisation) + 1) // 2]
    f0_amplitude = np.sqrt(a1 ** 2 + b1 ** 2)
    return f0_amplitude


class DuffingContinuation(continuation.ControlBasedContinuation):
    """ TODO comments """

    def get_period(self, continuation_vec):
        return 2 * np.pi / continuation_vec[0]


def main():
    discretisor = discretise.FourierDiscretisor(N_HARMONICS)
    continuer = DuffingContinuation(blackbox_system, discretisor)
    results = []

    for par_0, par_1 in STARTER_PARAMS:
        # Run forward, then backward
        signal_0 = blackbox_system(None, par_0)
        signal_1 = blackbox_system(None, par_1)
        starters = [
            build_continuation_vector(signal_0, discretisor, par_0),
            build_continuation_vector(signal_1, discretisor, par_1),
        ]

        # solver = lambda sys, x0: continuation.newton_solver(
        #     sys, x0, finite_differences_stepsize=FINITE_DIFFERENCES_STEPSIZE
        # )

        """ SCIPY SOLVER """
        solver = continuation.scipy_broyden_solver

        continuation_vectors, message = continuer.run_continuation(
            starters, solver=solver, step_control=STEPSIZE, par_range=[0.5, 2]
        )
        print(message)
        results.append(continuation_vectors)

        # Plot results
    fig, ax = plt.subplots()
    for vec_list in results:
        ax.plot(
            [continuer.get_parameter(v) for v in vec_list],
            [get_amplitude(continuer.get_discretisation(v)) for v in vec_list],
            color="k",
        )
    ax.set_xlabel("Forcing amplitude")
    ax.set_ylabel("Response amplitude")
    plt.show()


if __name__ == "__main__":
    main()
