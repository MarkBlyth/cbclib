# TODO        Move adaptive-stepsize Jacobian finder, stepsize to changeable parameters
# TODO        Integrate finite-differences jacobian into Broyden method, so we can reuse the initial f(x0) and save a computation
# TODO        Rewrite Newton solver as something more tidy

import abc
import numpy as np
import discretise
import copy
import numdifftools as ndt
import scipy.integrate
import warnings

DEFAULT_CONVERGENCE_TOL = 5e-4

"""
GOAL: scripts for continuing periodic orbits in a controlled system.
   * Allows different systems
   * Allows different discretisors
   * Allows inclusion or exclusion of phase conditions
   * SHOULD allow different numerical solvers, especially surrogate-based solvers
   * DOES NOT allow different augmentations, eg. to track folds

TODOs
   * Autonymous continuation is untested; give it a go!
   * Eventually, build into cbc_lib, to make it easier to define systems and controllers
   * Replace the solver with something more portable, eg. more standard max_iter, convergence_criteria
   * Simplify API -- make it easier to set up and run the code
"""


class StepsizeError(Exception):
    pass


def finite_differences_jacobian(f, x, stepsize=1e-3, central=False):
    if np.isscalar(stepsize):
        stepsize = stepsize * np.ones(x.shape)
    perturbations = np.diag(stepsize)
    if central:
        jac_transpose = [(f(x + h) - f(x - h)) / (2 * np.max(h)) for h in perturbations]
    else:
        f_x = f(x)
        jac_transpose = [(f(x + h) - f_x) / np.max(h) for h in perturbations]
    return np.array(jac_transpose).T


def scipy_broyden_solver(sys, x0):
    solution = scipy.optimize.root(
        sys, x0, tol=DEFAULT_CONVERGENCE_TOL, method="broyden1"
    )
    print("Solution vector: ", solution.x)
    print("Solution value: ", solution.fun)
    print("Parameter: ", solution.x[0])
    if solution.success:
        return solution.x
    return None


def broyden_solver(
    system,
    starter,
    max_iter=25,
    finite_differences_stepsize=None,
    convergence_tol=DEFAULT_CONVERGENCE_TOL,
    silent=False,
    callback=None,
):
    def new_print(*args):
        if not silent:
            print(*args)

    jacobian_func = ndt.Jacobian(system, step=finite_differences_stepsize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jacobian = jacobian_func(starter)
    #  jacobian = finite_differences_jacobian(system, starter)
    min_iters = 1 if callback is None else 2
    i, soln, func_evaluation = 0, starter, system(starter)
    results = [(starter, np.linalg.norm(func_evaluation))]

    while (
        i < min_iters
        or (
            np.linalg.norm(step) > convergence_tol
            and np.linalg.norm(func_evaluation) > convergence_tol
        )
    ) and i < max_iter:
        step = np.linalg.solve(jacobian, func_evaluation)
        soln -= step
        last_func_evaluation = func_evaluation
        func_evaluation = system(soln)
        jacobian_step = (
            (func_evaluation - last_func_evaluation) - jacobian.dot(step)
        ).dot(step.T) / np.linalg.norm(step) ** 2
        jacobian += jacobian_step
        results.append((soln, np.linalg.norm(func_evaluation)))

        new_print("Solver iteration ", i)
        new_print("Jacobian condition number: ", np.linalg.cond(jacobian))
        new_print("New continuation vector:\n", soln)
        new_print("System evaluation:\n", func_evaluation, "\n")
        i += 1
        if callback is not None:
            callback(step)

    if i != max_iter:
        new_print("Converged in {0} step(s)".format(i), "\n")
    return min(results, key=lambda x: x[1])[0]


def newton_solver(
    system,
    starter,
    max_iter=5,
    finite_differences_stepsize=None,
    convergence_tol=DEFAULT_CONVERGENCE_TOL,
    silent=False,
    callback=None,
):
    # jacobian_func = ndt.Jacobian(system, step=finite_differences_stepsize)

    def new_print(*args):
        if not silent:
            print(*args)

    # Solve
    i, last_step, this_step = 0, None, starter
    f_eval = system(starter)
    results = [(starter, np.linalg.norm(f_eval))]
    min_iters = 1 if callback is None else 2
    while (
        i < min_iters
        or (
            np.linalg.norm(system(this_step)) > convergence_tol
            and np.linalg.norm(step) > convergence_tol
        )
    ) and i < max_iter:
        if i > max_iter:
            return None
        last_step = this_step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jacobian = jacobian_func(this_step)
        #  jacobian = finite_differences_jacobian(
        #     system, starter
        # )  
        f_eval = system(this_step)
        step = np.linalg.solve(jacobian, -f_eval)
        this_step += step
        i += 1
        results.append((this_step, np.linalg.norm(f_eval)))

        new_print("Solver iteration ", i)
        new_print("Jacobian condition number: ", np.linalg.cond(jacobian))
        new_print("New continuation vector:\n", this_step)
        new_print("System evaluation:\n", system(this_step), "\n")
        if callback is not None:
            callback(step)

    if i != max_iter:
        new_print("Converged in {0} step(s)".format(i), "\n")
    return min(results, key=lambda x: x[1])[0]


class Continuation(abc.ABC):
    """
    TODO docstring explaining how and when to use this
    """

    def __init__(self, continuation_target, discretisor, autonymous=False):
        # TODO modify this for consistency with collocation continuation
        """
        Initialize a Continuation object. This object then allows
        periodic orbits of the specified system to be continued with
        CBC. Requires a system to continue, a discretisation method,
        and optionally a phase condition.

            continuation_target : function( [arg] , system parameter)
                The system that we wish to perform numerical
                continuation on. This could either be an ODE model, or
                a physical system, depending on the type of
                continuation in use. [arg] is some
                implementation-dependent argument, such as initial
                condition or control target. Returns a signal of form
                [[signal ts], [signal ys]]. The exact method for doing
                this is left to the system implementer.

            discretisor : _Discretisor obj
                Instantiation of a class implementing the _Discretisor
                API.

            autonymous : bool or function
                If True, the default phase condition is used. This is
                given by the numerically-computed integral phase
                condition. If False, no phase condition is used. If a
                function is passed, it must be of signature func(self,
                current_vec, last_vec). This function specifies a
                custom phase condition, and is used for the
                continuation.
        """
        self.continuation_target = continuation_target
        self.discretisor = discretisor
        if isinstance(autonymous, bool):
            self.phase_condition = (
                self.__default_phase_condition
                if autonymous
                else self.__no_phase_condition
            )
        else:
            self.phase_condition = autonymous

    """
    METHODS, HELPERS USED TO RUN A CONTINUATION PROBLEM.
    """

    def _nonadaptive_prediction_correction_step(
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

        def bound_continuation_system(y_in):
            return self._continuation_system(y_in, y1, prediction, secant)

        return solver(bound_continuation_system, prediction)

    def _adaptive_stepsize_prediction_correction_step(
        self,
        y0,
        y1,
        stepsize,
        nominal_steplength,
        nominal_contraction,
        min_stepsize,
        max_stepsize,
    ):

        secant = (y1 - y0) / np.linalg.norm(y1 - y0)
        prediction = y1 + stepsize * secant
        newton_steps = []

        print("Starting solver with a stepsize of ", stepsize)
        print("Prediction: ", prediction)

        def bound_continuation_system(y_in):
            return self._continuation_system(y_in, y1, prediction, secant)

        def callback(step):
            """Store solutions as they become available. If we need to
            retry with a smaller stepsize, raise a StepsizeError.
            Otherwise use the first two steps to update the
            stepsize."""
            nonlocal stepsize
            if len(newton_steps) > 1:
                return
            newton_steps.append(step)
            if len(newton_steps) == 2:
                delta = np.linalg.norm(newton_steps[0])
                kappa = np.linalg.norm(newton_steps[1]) / delta
                print(
                    "Steplength: nominal, actual, multiplier: {0}, {1}, {2}".format(
                        nominal_steplength, delta, np.sqrt(nominal_steplength / delta)
                    )
                )
                print(
                    "Contraction: nominal, actual, multiplier: {0}, {1}, {2}".format(
                        nominal_contraction, kappa, np.sqrt(nominal_contraction / kappa)
                    )
                )
                f = max(delta / nominal_steplength, kappa / nominal_contraction)
                if (f > 4) and (stepsize / 2 > min_stepsize):
                    print("Retrying with a smaller stepsize")
                    stepsize /= 2
                    print("New stepsize: ", stepsize, "\n")
                    raise StepsizeError
                f_floor_ceil = max(0.25, min(f, 4))
                stepsize /= np.sqrt(f_floor_ceil)
                stepsize = min(max(stepsize, min_stepsize), max_stepsize)
                print("New stepsize: ", stepsize, "\n")

        try:
            # Attempt to solve with the current stepsize
            solution = newton_solver(
                bound_continuation_system,
                prediction,
                callback=callback,
            )
            # BROYDEN SOLVER ALWAYS RETURNS A SOLUTION GUESS SO THIS ISN'T ACTUALLY DOING ANYTHING THERE
            if solution is None:
                # Solver returns None on failure to converge
                print("Convergence was not reached within the iteration limit")
                return -1, None
            print("New stepsize: ", stepsize)
            print("Solution vector: ", solution)
            print("Parameter: ", solution[0], "\n\n")
            return stepsize, solution
        except StepsizeError:
            # Raised if we need to try again with a smaller step
            # Compute a new prediction using the updated stepsize, and solve
            return self._adaptive_stepsize_prediction_correction_step(
                y0,
                y1,
                stepsize,
                nominal_steplength,
                nominal_contraction,
                min_stepsize,
                max_stepsize,
            )

    def run_continuation(
        self,
        starters,
        solver=scipy_broyden_solver,
        step_control=1,
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
                Function of signature solver(system, initial_guess).
                Returns the solution to the system, starting from
                initial point initial_guess.

            step_control : float > 0
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
            (
                stepsize,
                min_stepsize,
                max_stepsize,
                nominal_steplength,
                nominal_contraction,
            ) = step_control

            def take_step(self, y0, y1, stepsize):
                return self._adaptive_stepsize_prediction_correction_step(
                    y0,
                    y1,
                    stepsize,
                    nominal_steplength,
                    nominal_contraction,
                    min_stepsize,
                    max_stepsize,
                )

        except TypeError:

            def take_step(self, y0, y1, stepsize):
                return stepsize, self._nonadaptive_prediction_correction_step(
                    y0, y1, stepsize, solver
                )

            stepsize = step_control
        try:
            for i in range(n_steps):
                # Keep stepping until an exit criterion is met
                print("Step {0}".format(i + 1))
                stepsize, new_vec = take_step(
                    self, solution_vecs[-2], solution_vecs[-1], stepsize
                )
                if new_vec is None:
                    message = "Continuation terminated as last correction step did not converge"
                    break
                solution_vecs.append(new_vec)
                try:
                    if isinstance(self.discretisor, discretise._AdaptiveDiscretisor):
                        self.discretisor.update_discretisation_scheme(
                            self._evaluate_system_from_continuation_vector(
                                solution_vecs[-1]
                            ),
                            self.get_period(solution_vecs[-1]),
                        )
                except AttributeError:
                    # TODO add support for adaptive-mesh collocation
                    pass
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

    def __no_phase_condition(self, last_vec, current_vec):
        """
        If the Coninuation.phase_condition function returns None, the
        phase condition is ignored. This function implements a phase
        condition that is ignored, for use eg. in autonymous systems, when
        the phase is determined by the system in question.
        """
        return None

    def __default_phase_condition(self, current_vec, last_vec):
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
        reference_gradient = ndt.Gradient(reference_model)
        return scipy.integrate.quad(
            lambda t: np.inner(current_model(t), reference_gradient(t)), 0, 1
        )[0]

    """
    METHOD SET BY THE SPECIFIC CONTINUATION SCHEME.
    """

    @abc.abstractmethod
    def _evaluate_system_from_continuation_vector(self, continuation_vector):
        """
        Given a continuation vector, parse the vector into period,
        parameter, and discretisation, and produce a system evaluation
        from this. Eg. a simulation of the modelled periodic orbit, or
        an output from the controlled system.

            continuation_vector : ndarray
                A vector following the specification set out by the
                API implementor, ie. the user decides how to format
                the continuation vector, and self.get_discretisation,
                self.get_period, and self.get_parameter implement the
                relevant parsing of the vector.

        Returns an output signal of form [[ts], [ys]].
        """
        pass

    @abc.abstractmethod
    def _continuation_system(
        self, continuation_vector, last_solution, prediction, secant
    ):
        """
        The equations defining the system whose solution we wish to
        numerically continue. This could be a CBC fixed-point problem,
        or a collocation system, etc.

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
        pass

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


class ControlBasedContinuation(Continuation):
    """
    TODO how does the user do something with this? What does the constructor look like? How do they initialise? Which abstract methods need implementing?


            continuation_target : function(control target, system parameter)
                Something resembling a physical system. Given only a
                control target and a system parameter, it returns a signal
                of form [[signal ts], [signal ys]]. The exact method for
                doing this is left to the system implementer. It is
                suggested that the system uses the previous run's
                end-state as the initial conditions for the next run; that
                the system integrates out and omits all transients; that
                the system ensures a sufficiently long integration time to
                fully represent the signal, eg. sufficient periods.
    """

    def get_parameter(self, continuation_vec):
        return continuation_vec[0]

    def get_discretisation(self, continuation_vec):
        return continuation_vec[1:]

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
        signal = self._evaluate_system_from_continuation_vector(y)
        period = self.get_period(y)
        discretised_signal = self.discretisor.discretise(signal, period)
        return discretised_signal

    def _evaluate_system_from_continuation_vector(self, continuation_vector):
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
        return self.continuation_target(control_target, parameter)
