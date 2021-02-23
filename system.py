import numpy as np
import scipy.integrate


class System:
    """
    A model of a physical system. The model is governed by a set of
    ODEs. The system is evaluated using some control target and some
    parameter value. The system is first initialised at the provided
    initial condition. Thereafter, it is initialised from the final
    state of the previous run, to model a real system. When a period
    is provided, the integration time is extended so that the final
    state is always at phase=0, which avoids any discontinuity between
    the current and next control target. The system is controlled
    using a controller object, and returned observations are obtained
    by observing the states according to the controller's observations.
    """
    def __init__(
        self,
        initial_cond,
        ode_rhs,
        transient_time,
        evaluation_time,
        sample_rate,
        controller,
    ):
        """
            initial_cond : initial condition for the first integration

            ode_rhs : func(t, x, parameter)
                ODE to integrate

            transient_time : float > 0
                Timespan of discarded system outputs

            evaluation_time : float > 0
                Timespan of measured system outputs

            sample_rate : float > 0
                Number of solution samples per integration time

            controller : func(state, target)
                As returned from get_proportional_controller
        """
        self.initial_cond = initial_cond
        self.ode_rhs = ode_rhs
        self.evaluation_time = evaluation_time
        self.transient_time = transient_time
        self.sample_rate = sample_rate
        self.controller = controller

    def __call__(self, parameter, control_target=None, period=None):
        """
        Run the system, controlled or uncontrolled.

            parameter : float
                Parameter value at which to run the system
        
            control_target : func(t)
                Get the target system output at time t. If None, the
                system is ran uncontrolled.

            period : float
                The period of a periodic control target. If set, the
                integration time is extended so that the last
                measurement is taken at control phase = 0
        """
        end_time = self.transient_time + self.evaluation_time
        # If we know the period, add some extra integration time
        extra_time = 0 if period is None else (period - np.mod(end_time, period))
        end_time += extra_time

        t_span = [0, end_time]
        t_eval = np.linspace(
            self.transient_time,
            end_time,
            int((end_time - self.transient_time) * self.sample_rate),
        )
        if control_target is None:
            system = lambda t, x: self.ode_rhs(t, x, parameter)
        else:
            system = lambda t, x: self.ode_rhs(t, x, parameter) + self.controller(
                x, control_target(t)
            )
        soln = scipy.integrate.solve_ivp(
            system,
            t_span=t_span,
            t_eval=t_eval,
            y0=self.initial_cond,
        )
        self.initial_cond = np.squeeze(soln.y[:, -1])
        return np.vstack((soln.t, self.controller.observe(soln.y)))


class ProportionalController:
    """
    Callable class whose function gives the control action for a given
    state and target.
    """

    def __init__(self, kp, observed_var, controlled_var):
        """
        Assume only the i'th variable is measured as the system output;
        this is encoded with observed_var, a vector with the same
        dimension as the state vector, whose i'th entry is one and all
        other entries are zero. Similarly, assume control is applied only
        to the j'th variable; this is encoded with controlled_var, a
        vector with the same dimension as the state vector, whose j'th
        entry is one and all other entries are zero. Control gain is given
        by kp.
        """
        self.kp = kp
        self.observed_var = np.array(observed_var)
        self.controlled_var = np.array(controlled_var)

    def observe(self, states):
        """
        Given an array where each row is a state variable, return the
        row corresponding to observations of the system. TODO this
        method is inefficient. It would be computationally better to
        simply index the data.
        """
        return np.dot(self.observed_var, states)

    def __call__(self, state, target):
        return self.kp * self.controlled_var * (target - self.observe(state))
