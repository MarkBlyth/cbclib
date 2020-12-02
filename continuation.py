import abc
import numpy as np
import scipy.optimize
import discretise
import copy
import numdifftools as ndt

"""
GOAL: scripts for continuing periodic orbits in a controlled system.
   * Allows different systems
   * Allows different discretisors
   * Allows inclusion or exclusion of phase conditions
   * SHOULD allow different numerical solvers, especially surrogate-based solvers
   * DOES NOT allow different augmentations, eg. to track folds

TODOs
   * Eventually, build into cbc_lib, to make it easier to define systems and controllers
   * Replace the solver with something more portable, eg. more standard max_iter, convergence_criteria
   * Simplify API -- make it easier to set up and run the code
"""


def newton_solver(
    system,
    starter,
    max_iter=100,
    finite_differences_stepsize=None,
    convergence_criteria=None,
    silent=False,
):
    jacobian_func = ndt.Jacobian(system, step=finite_differences_stepsize)

    if convergence_criteria is None:
        def convergence_criteria(vec0, vec1, system_evaluation):
            return np.linalg.norm(vec1 - vec0) < 1e-6

    def modified_convergence_criteria(
        vec0, vec1, system_evaluation, convergence_criteria
    ):
        if vec0 is None:
            return False
        return convergence_criteria(vec0, vec1, system_evaluation)

    def new_print(*args):
        if not silent:
            print(*args)

    # Solve
    i, last_step, this_step = 0, None, starter
    while not modified_convergence_criteria(
        last_step, this_step, system(this_step), convergence_criteria
    ):
        if i > max_iter:
            return None
        last_step = this_step
        jacobian = jacobian_func(this_step)
        step = np.linalg.solve(jacobian, -system(this_step))
        this_step += step
        i += 1
        new_print("Solver iteration ", i)
        new_print("Jacobian:\n", jacobian)
        new_print("\nJacobian condition number: ",
                  np.linalg.cond(jacobian), "\n")
        new_print("\nNew continuation vector:\n", this_step)
        new_print("\nSystem evaluation:\n", system(this_step), "\n")
    new_print("Converged in {0} step(s)".format(i))
    new_print("\n\n")
    return this_step


class Continuation(abc.ABC):
    """
    TODO docstring explaining how and when to use this
    """

    def __init__(self, blackbox_system, discretisor):
        """
        Initialize a Continuation object. This object then allows
        periodic orbits of the specified system to be continued with
        CBC. Requires a system to continue, a discretisation method,
        and optionally a phase condition.

            blackbox_system : function(control target, system parameter)
                Something resembling a physical system. Given only a
                control target and a system parameter, it returns a signal
                of form [[signal ts], [signal ys]]. The exact method for
                doing this is left to the system implementer. It is
                suggested that the system uses the previous run's
                end-state as the initial conditions for the next run; that
                the system integrates out and omits all transients; that
                the system ensures a sufficiently long integration time to
                fully represent the signal, eg. sufficient periods.

            discretisor : _Discretisor obj
                Instantiation of a class implementing the _Discretisor
                API.

            phase_condition : function
                Function of signature func(self, current_vec,
                last_vec). Evaluates a phase condition if desired. If
                no phase condition is desired, returns None.
        """
        self.blackbox_system = blackbox_system
        self.discretisor = discretisor

    """
    METHODS, HELPERS USED TO RUN A CONTINUATION PROBLEM.
    """

    def _modified_convergence_criteria(self, vec0, vec1, system_evaluation, convergence_criteria):
        """
        Override the provided convergence_criteria function so that it
        returns False on the first step. This allows the solver to
        avoid converging on the very first iteration.

            vec0 : continuation vector
                Previous iteration vector.

            vec1 : continuation vector
                Current iteration vector.

            convergence_criteria : function
                Function of signature func(v0, v1). Returns True if
                the solver iteration scheme has converged, based on
                the current and previous solution estimates. False
                otherwise.
        """
        ###########################
        # import matplotlib.pyplot as plt
        # signal = self._run_system_from_continuation_vector(vec1)
        # control_target = self.discretisor.undiscretise(self.get_discretisation(vec1), self.get_period(vec1))
        # fig, ax = plt.subplots()
        # ax.plot(signal[0], signal[1], label="Signal (output)")
        # ax.plot(signal[0], control_target(signal[0]), label="Target (input)")
        # ax.legend()
        # ax.set_title("After a Newton step")
        # plt.show()
        # ##########################

        if vec0 is None:
            return False
        return convergence_criteria(vec0, vec1, system_evaluation)

    def _prediction_correction_step(
        self,
        y0,
        y1,
        stepsize,
        max_iter,
        convergence_criteria,
        finite_differences_stepsize=None,
    ):
        """
        Perform a predictor-corrector step using the psuedo-arclength
        method with a secant predictor, and Newton-iteration orthogonal
        corrector.

            y0, y1 : continuation vector
                Continuation vectors to initialize the continuation
                procedure from. Used in calculating the secant
                prediction vector.

            stepsize : float > 0
                Size of the secant vector used when making a prediction of
                the next periodic orbit. Needs to be sufficiently small to
                be able to accurately represent the continuation curve.

            max_iter : int > 0
                Maximum number of Newton iterations before the correction
                step should be deemed non-convergent.

            convergence_criteria : func(x_{n-1}, x_n)
                Function to decide whether the iteration scheme has
                converged. Takes the current and the previous step values.
                Returns True if the iteration is deemed converged, False
                otherwise.

            finite_differences_stepsize : float > 0
                Stepsize for finite differences Jacobian estimate. If
                None, stepsize is determined automatically.

        Returns the next solution vector to the continuation system, if
        convergence is achieved. If convergence is not reached within
        max_iter steps, returns None.
        """
        # Predict
        secant = (y1 - y0) / np.linalg.norm(y1 - y0)
        prediction = y1 + stepsize * secant
        bound_continuation_system = lambda y_in: self._continuation_system(
            y_in, y1, prediction, secant
        )
        jacobian_func = ndt.Jacobian(bound_continuation_system, step=finite_differences_stepsize)
        # Solve
        i, last_step, this_step = 0, None, prediction
        while not self._modified_convergence_criteria(
                last_step, this_step, bound_continuation_system(this_step), convergence_criteria
        ):
            if i > max_iter:
                return None
            last_step = this_step
            jacobian = jacobian_func(this_step)
            step = np.linalg.solve(jacobian, -bound_continuation_system(this_step))
            this_step += step
            i += 1
            print("Solver iteration ", i)
            print("Jacobian:\n", jacobian)
            print("\nJacobian condition number: ", np.linalg.cond(jacobian), "\n")
            print("\nNew continuation vector:\n", this_step)
            print("\nSystem evaluation:\n", bound_continuation_system(this_step), "\n")
        print("Converged in {0} step(s), at parameter value {1}".format(i, self.get_parameter(this_step)))
        print("\n\n")

        ###########################
        # import matplotlib.pyplot as plt
        # signal = self._run_system_from_continuation_vector(this_step)
        # control_target = self.discretisor.undiscretise(self.get_discretisation(this_step), self.get_period(this_step))
        # fig, ax = plt.subplots()
        # ax.plot(signal[0], signal[1], label="Signal (output)")
        # ax.plot(signal[0], control_target(signal[0]), label="Target (input)")
        # ax.legend()
        # ax.set_title("Accepted continuation result")
        # plt.show()
        # ##########################
        return this_step

    def _new_prediction_correction_step(
        self,
        y0,
        y1,
        stepsize,
        solver=newton_solver,
    ):
        """
        Perform a predictor-corrector step using the psuedo-arclength
        method with a secant predictor, and Newton-iteration orthogonal
        corrector.

            y0, y1 : continuation vector
                Continuation vectors to initialize the continuation
                procedure from. Used in calculating the secant
                prediction vector.

            stepsize : float > 0
                Size of the secant vector used when making a prediction of
                the next periodic orbit. Needs to be sufficiently small to
                be able to accurately represent the continuation curve.

            solver : func
                Function of signature solver(system, initial_guess,
                max_iter). Returns the solution to the system,
                starting from initial point initial_guess.

        Returns the next solution vector to the continuation system, if
        convergence is achieved. If convergence is not reached within
        max_iter steps, returns None.
        """
        # Predict
        secant = (y1 - y0) / np.linalg.norm(y1 - y0)
        prediction = y1 + stepsize * secant
        bound_continuation_system = lambda y_in: self._continuation_system(
            y_in, y1, prediction, secant
        )
        return solver(bound_continuation_system, prediction)

    def _continuation_system(
        self, continuation_vector, last_solution, prediction, secant
    ):
        """
        Minimally augmented system for the control-based continuation
        of periodic orbits. Solves for discretised input = discretised
        output, orthogonality between the prediction and correction
        vectors, and optionally a phase constraint.

            continuation_vector : ndarray
                A vector following the specification set out by the
                API implementor, ie. the user decides how to format
                the continuation vector, and self.get_discretisation,
                self.get_period, and self.get_parameter implement the
                relevant parsing of the vector.

            last_solution : ndarray
                Last accepted continuation vector; provides a phase
                reference.

            prediction : ndarray
                Secant-derived predictor of the current continuation
                step solution. Obtained from the previous two accepted
                solutions.

            secant : ndarray
                Secant vector used for the current continuation
                prediction. Obtained from the previous two accepted
                solutions.

        Returns an evaluation of the zero-problem.
        """
        IOMap_output = self._IO_map(continuation_vector)
        F = self.get_discretisation(continuation_vector) - IOMap_output
        g = np.dot(continuation_vector - prediction, secant)
        h = self.phase_condition(continuation_vector, last_solution)
        return np.hstack((F, g)) if h is None else np.hstack((F, g, h))

    def _IO_map(self, y):
        """
        Evaluate a single iteration of the IO-map. Takes a
        continuation vector object, runs the black-box system based on
        the information provided in the continuation vector, then
        returns the discretised result.

            y : ndarray
                The continuation vector to evaluate the IO map at.

        Returns a discretised system output, from the control target
        and period described by the continuation vector.
        """
        signal = self._run_system_from_continuation_vector(y)
        period = self.get_period(y)
        discretised_signal = self.discretisor.discretise(signal, period)
        ##########
        # import matplotlib.pyplot as plt
        # signal = self._run_system_from_continuation_vector(y)
        # control_target = self.discretisor.undiscretise(self.get_discretisation(y), self.get_period(y))
        # fig, ax = plt.subplots()
        # ax.plot(signal[0], signal[1], label="Signal (output)")
        # ax.plot(signal[0], control_target(signal[0]), label="Target (input)")
        # ax.set_title("During a Newton step")
        # ax.legend()
        # plt.show()
        # ###############
        return discretised_signal

    def _run_system_from_continuation_vector(self, continuation_vector):
        """
        Given a continuation vector, parse the vector into period,
        parameter, and discretisation; undiscretise the control
        target; then run the system with the specified control target
        and parameter.

            continuation_vector : ndarray
                A vector following the specification set out by the
                API implementor, ie. the user decides how to format
                the continuation vector, and self.get_discretisation,
                self.get_period, and self.get_parameter implement the
                relevant parsing of the vector.

        Returns an output signal of form [[ts], [ys]].
        """
        discretisation = self.get_discretisation(continuation_vector)
        period = self.get_period(continuation_vector)
        parameter = self.get_parameter(continuation_vector)
        control_target = self.discretisor.undiscretise(discretisation, period)
        return self.blackbox_system(control_target, parameter)

    def run_continuation(
        self,
        starters,
        convergence_criteria,
        finite_differences_stepsize=None,
        stepsize=0.01,
        par_range=[-np.inf, np.inf],
        n_steps=1000,
        max_period=np.inf,
        max_iters=50,
    ):
        """Run a continuation experiment on a given system, using a given
        discretisor. The continuation continues until one of the following
        exit criteria are met:
            * The Newton iterations fail to converge
            * The number of steps taken exceeds n_steps
            * The continuation parameter leaves the target interval
            par_range
            * The signal reconstruction error exceeds max_error
            * The signal period exceeds max_period

            starters : list of vectors
                A list of form [continuation vector 1, continuation vector
                2].

            convergence_criteria : func x,y -> bool
                Function of signature f(y0, y1). Returns True if given
                continuation vectors y0 and y1, the solver procedure
                can be deemed converged, False otherwise.

            finite_differences_stepsize : float > 0
                Size of the perturbations to make when computing
                finite differences Jacobian. If None, stepsize is
                chosen automatically.

            stepsize : float > 0
                Size of the secant vector used when making a prediction of
                the next periodic orbit. Needs to be sufficiently small to
                be able to accurately represent the continuation curve.

            par_range : 2-tuple
                Minimum and maximum values of the continuation parameter.
                Default +/- infinity.

            n_steps : int > 0
                Number of predictor-corrector steps to take. Default 1000.

            max_period : float > 0
                Maximum period of a signal. Default infinity.

            max_iters : int > 0
                Maximum number of Newton iterations in the correction
                step, before a failure-to-converge is declared.

        """
        message = None
        # Initialise the first results in the continuation, for secant prediction
        solution_vecs = [copy.deepcopy(s) for s in starters]
        # Can exit the continuation steps with CTRL-C
        try:
            for i in range(n_steps):
                # Keep stepping until an exit criterion is met
                print("Step {0}".format(i + 1))
                new_vec = self._prediction_correction_step(
                    solution_vecs[-2],
                    solution_vecs[-1],
                    stepsize,
                    max_iters,
                    convergence_criteria,
                    finite_differences_stepsize,
                )
                if new_vec is None:
                    message = "Continuation terminated as last correction step did not converge"
                    break
                solution_vecs.append(new_vec)
                if isinstance(self.discretisor, discretise._AdaptiveDiscretisor):
                    self.discretisor.update_discretisation_scheme(
                        self._run_system_from_continuation_vector(solution_vecs[-1]),
                        self.get_period(solution_vecs[-1])
                    )
                # Exit criteria:
                if (self.get_parameter(solution_vecs[-1]) > max(par_range)) or (
                    self.get_parameter(solution_vecs[-1]) < min(par_range)
                ):
                    message = "Continuation terminated as parameter left target range"
                    break
                if self.get_period(solution_vecs[-1]) > max_period:
                    message = "Continuation terminated as period exceeded maximum value"
                    break
            if message is None:
                message = "Continuation terminated as maximum number of iterations were reached"
        except KeyboardInterrupt:
            # Can exit the continuation steps with CTRL-C
            message = "Continuation terminated though keyboard interrupt"
        return solution_vecs, message

    def new_run_continuation(
        self,
        starters,
        solver=newton_solver,
        stepsize=1,
        par_range=[-np.inf, np.inf],
        n_steps=1000,
        max_period=np.inf,
    ):
        """Run a continuation experiment on a given system, using a given
        discretisor. The continuation continues until one of the following
        exit criteria are met:
            * The Newton iterations fail to converge
            * The number of steps taken exceeds n_steps
            * The continuation parameter leaves the target interval
            par_range
            * The signal reconstruction error exceeds max_error
            * The signal period exceeds max_period

            starters : list of vectors
                A list of form [continuation vector 1, continuation vector
                2].

            solver : func
                Function of signature solver(system, initial_guess,
                max_iter). Returns the solution to the system,
                starting from initial point initial_guess.

            stepsize : float > 0
                Size of the secant vector used when making a prediction of
                the next periodic orbit. Needs to be sufficiently small to
                be able to accurately represent the continuation curve.

            par_range : 2-tuple
                Minimum and maximum values of the continuation parameter.
                Default +/- infinity.

            n_steps : int > 0
                Number of predictor-corrector steps to take. Default 1000.

            max_period : float > 0
                Maximum period of a signal. Default infinity.
        """
        message = None
        # Initialise the first results in the continuation, for secant prediction
        solution_vecs = [copy.deepcopy(s) for s in starters]
        # Can exit the continuation steps with CTRL-C
        try:
            for i in range(n_steps):
                # Keep stepping until an exit criterion is met
                print("Step {0}".format(i + 1))
                new_vec = self._new_prediction_correction_step(
                    solution_vecs[-2],
                    solution_vecs[-1],
                    stepsize,
                    solver,
                )
                if new_vec is None:
                    message = "Continuation terminated as last correction step did not converge"
                    break
                solution_vecs.append(new_vec)
                if isinstance(self.discretisor, discretise._AdaptiveDiscretisor):
                    self.discretisor.update_discretisation_scheme(
                        self._run_system_from_continuation_vector(solution_vecs[-1]),
                        self.get_period(solution_vecs[-1])
                    )
                # Exit criteria:
                if (self.get_parameter(solution_vecs[-1]) > max(par_range)) or (
                    self.get_parameter(solution_vecs[-1]) < min(par_range)
                ):
                    message = "Continuation terminated as parameter left target range"
                    break
                if self.get_period(solution_vecs[-1]) > max_period:
                    message = "Continuation terminated as period exceeded maximum value"
                    break
            if message is None:
                message = "Continuation terminated as maximum number of iterations were reached"
        except KeyboardInterrupt:
            # Can exit the continuation steps with CTRL-C
            message = "Continuation terminated though keyboard interrupt"
        return solution_vecs, message

    """
    METHODS SET BY THE USER, TO DEFINE A CONTINUATION PROBLEM.
    """

    @abc.abstractmethod
    def get_parameter(self, continuation_vec):
        """
        Given a continuation vector, parse the vector and extract the
        bifurcation parameter from it.

            continuation_vec : float array
                1d array containing the discretisation vector and any
                regularisation terms.

        Returns the bifurcation parameter.
        """
        pass

    @abc.abstractmethod
    def get_period(self, continuation_vec):
        """
        Given a continuation vector, parse the vector and extract the
        signal period from it.

            continuation_vec : float array
                1d array containing the discretisation vector and any
                regularisation terms.

        Returns the signal period.
        """
        pass

    @abc.abstractmethod
    def get_discretisation(self, continuation_vec):
        """
        Given a continuation vector, parse the vector and extract the
        signal discretisation from it.

            continuation_vec : float array
                1d array containing the discretisation vector and any
                regularisation terms.

        Returns the signal discretisation.
        """
        pass


class AutonymousContinuation(Continuation):
    """
    TODO docstring explaining how and when to use this
    """

    def phase_condition(self, current_vec, last_vec):
        """
        Implements a phase condition for a continuation procedure. This
        implementation returns the integrated product of the current
        signal, and the previous signal. This phase condition is
        implemented as standard in AUTO, and generally accepted as being
        the best choice. It will work out-the-box for any system.
        Nevertheless, it may be possible to produce a tailor-made, more
        computationally efficient phase condition, depending on the
        specific problem of interest.

            current_vec : continuation vector
                Continuation vector at the current predictor/corrector
                step

            last_vec : continuation vector
                Continuation vector at the previous predictor/corrector
                step

        Returns some float which is zero only when the phase shift between
        the current and reference signal is minimized.
        """
        current_model = self.discretisor.undiscretise(
            self.get_discretisation(current_vec), 1
        )
        reference_model = self.discretisor.undiscretise(
            self.get_discretisation(last_vec), 1
        )
        return scipy.integrate.quad(
            lambda t: current_model(t) * scipy.misc.derivative(reference_model, t), 0, 1
        )[0]


class NonAutonymousContinuation(Continuation):
    """
    TODO docstring explaining how and when to use this
    """

    def phase_condition(self, last_vec, current_vec):
        """
        If the Coninuation.phase_condition function returns None, the
        phase condition is ignored. This function implements a phase
        condition that is ignored, for use eg. in autonymous systems, when
        the phase is determined by the system in question.
        """
        return None
