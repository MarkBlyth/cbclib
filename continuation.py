import abc
import numpy as np
import discretise
import copy
import numdifftools as ndt
import scipy.integrate
import collections

"""
GOAL: scripts for continuing periodic orbits in a controlled system.
   * Allows different systems
   * Allows different discretisors
   * Allows inclusion or exclusion of phase conditions
   * SHOULD allow different numerical solvers, especially surrogate-based solvers
   * DOES NOT allow different augmentations, eg. to track folds

TODOs
   * Make a separate solvers package, with Newton, Broyden, SciPy, (adaptive Newton?); different convergence criteria; different Jacobian estimations; so that solvers can be easily mixed and matched.
   * Add f_eval, n_fev, and other interesting solver data into its return (maybe make a solver solution class?)
   * Store all the nice solver data in the ContinuationSolution
   * Move Continuation class, collocation classes into a separate Continuation Core package
     * Have this as a continuation-runners class (for numerical, cbc, eventually EPC)
   * Some docs on how to use each continuation driver
"""

N_UPDATE_POINTS = 1000  # Number of timepoint samples to use when simulating a control target to re-discretise under a new discretisation scheme

ContinuationSolution = collections.namedtuple(
    "ContinuationSolution",
    "vector, parameter, period, discretisation, control_target, discretisor",
)


def finite_differences_jacobian(f, x, f_x=None, stepsize=1e-3, central=False):
    """
       f : func(x) whose derivative we want to find
       x : the location at which to evaluate the derivative
       f_x : optional; f(x), used for forward FD; computed if None
       central : bool; if False, use forward FD; if True, use central
    """
    if np.isscalar(stepsize):
        stepsize = stepsize * np.ones(x.shape)
    perturbations = np.diag(stepsize)
    if central:
        jac_transpose = [(f(x + h) - f(x - h)) / (2 * np.max(h)) for h in perturbations]
    else:
        f_x = f_x if f_x is not None else f(x)
        jac_transpose = [(f(x + h) - f_x) / np.max(h) for h in perturbations]
    return np.array(jac_transpose).T


def newton_solver(
    system,
    starter,
    max_iter=20,
    finite_differences_stepsize=None,
    convergence_criteria=None,
    silent=False,
    hardstop=False,  ### If true, return None to indicate convergence failure when max_iter is exceeded; if false, return whatever solution we have after max_iter
):
    """
    TODO this shouldn't need hardstop. Instead, return a dict
    containing soln, n_iter, f_eval, converged, and let the caller
    decide whether they're happy to use the solution or not
    """
    # jacobian_obj = ndt.Jacobian(system, step=finite_differences_stepsize)
    # jacobian_func = lambda x, f_x: jacobian_obj(x)
    jacobian_func = lambda x, f_x: finite_differences_jacobian(
        system, x, f_x, finite_differences_stepsize
    )

    if convergence_criteria is None:

        def convergence_criteria(step, system_evaluation):
            return np.linalg.norm(system_evaluation) < 1e-3

    def modified_convergence_criteria(step, system_evaluation, convergence_criteria):
        return convergence_criteria(step, system_evaluation)

    def new_print(*args):
        if not silent:
            print(*args)

    # Solve
    i, step, soln, system_evaluation = 0, np.inf, starter, None
    while True:
        # Check termination conditions. Doing it this way lets us run
        # with the fewest possible function calls
        if i >= max_iter:
            return None if hardstop else soln
        if system_evaluation is None:
            system_evaluation = system(soln)
        if modified_convergence_criteria(step, system_evaluation, convergence_criteria):
            break
        # Haven't terminated, so update the solution
        jacobian = jacobian_func(soln, system_evaluation)
        step = np.linalg.solve(jacobian, -system_evaluation)
        soln += step
        i += 1
        system_evaluation = system(soln)
        # Print the results so far
        new_print("Solver iteration ", i)
        new_print("Jacobian condition number: ", np.linalg.cond(jacobian))
        new_print("New continuation vector:\n", soln)
        new_print("System evaluation:\n", system_evaluation, "\n")
    new_print("Terminated in {0} step(s)".format(i))
    return soln


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

            continuation_target : function( parameter, [args])
                The system that we wish to perform numerical
                continuation on. This could either be an ODE model, or
                a physical system, depending on the type of
                continuation in use. [args] are some
                implementation-dependent argument, such as initial
                condition or control target, and period. Returns a
                signal of form [[signal ts], [signal ys]]. The exact
                method for doing this is left to the system
                implementer.

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
        self.discretisor_is_adaptive = isinstance(
            self.discretisor, discretise._AdaptiveDiscretisor
        )

    """
    METHODS, HELPERS USED TO RUN A CONTINUATION PROBLEM.
    """

    def _prediction_correction_step(
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
        solution = solver(bound_continuation_system, prediction)
        return None if solution is None else self._construct_continuation_result(solution)

    def run_continuation(
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
                Function of signature solver(system, initial_guess).
                Returns the solution to the system, starting from
                initial point initial_guess.

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
        solutions = [self._construct_continuation_result(s) for s in starters]
        # Can exit the continuation steps with CTRL-C
        try:
            for i in range(n_steps):
                # Keep stepping until an exit criterion is met
                print("Step {0}".format(i + 1))
                if self.discretisor_is_adaptive:
                    soln1 = self._update_solution_to_new_discretisor(solutions[-2])
                    soln2 = self._update_solution_to_new_discretisor(solutions[-1])
                else:
                    soln1 = solutions[-2].vector
                    soln2 = solutions[-1].vector
                new_soln = self._prediction_correction_step(
                    soln1, soln2, stepsize, solver
                )
                if new_soln is None:
                    message = "Continuation terminated as last correction step did not converge"
                    break
                solutions.append(new_soln)
                try:
                    if self.discretisor_is_adaptive:
                        ts = np.linspace(0, solutions[-1].period, N_UPDATE_POINTS)
                        ys = solutions[-1].control_target(ts)
                        self.discretisor.update_discretisation_scheme(
                            np.vstack((ts, ys)),
                            solutions[-1].period,
                        )
                except AttributeError:
                    # TODO add support for adaptive-mesh collocation
                    pass
                # Exit criteria:
                if (solutions[-1].parameter > max(par_range)) or (
                    solutions[-1].parameter < min(par_range)
                ):
                    message = "Continuation terminated as parameter left target range"
                    break
                if solutions[-1].period > max_period:
                    message = "Continuation terminated as period exceeded maximum value"
                    break
            if message is None:
                message = "Continuation terminated as maximum number of iterations were reached"
        except KeyboardInterrupt:
            # Can exit the continuation steps with CTRL-C
            message = "Continuation terminated though keyboard interrupt"
        return solutions, message

    def _update_solution_to_new_discretisor(self, solution):
        """
        TODO docstring
        """
        ts = np.linspace(0, solution.period, N_UPDATE_POINTS)
        ys = solution.control_target(ts)
        signal = np.vstack((ts, ys))
        new_discretisation = self.discretisor.discretise(signal, solution.period)
        return self.build_vector(solution.parameter, solution.period, new_discretisation)

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

    def _construct_continuation_result(self, accepted_solution):
        discretisor = copy.deepcopy(self.discretisor)
        discretisation = self.get_discretisation(accepted_solution)
        period = self.get_period(accepted_solution)
        return ContinuationSolution(
            accepted_solution,
            self.get_parameter(accepted_solution),
            period,
            discretisation,
            discretisor.undiscretise(discretisation, period),
            discretisor,
        )

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
    def build_vector(self, parameter, period, discretisation):
        """
        Given a set of data, turn them into an appropriately formatted
        continuation vector. Returns a continuation vector
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


class CBC(Continuation):
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
        return self.continuation_target(parameter, control_target, np.abs(period))


class NonautonymousCBC(CBC):
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

    def __init__(self, continuation_target, discretisor):
        super().__init__(continuation_target, discretisor, False)

    @abc.abstractmethod
    def get_period(self, continuation_vec):
        """ Implemented by the user """
        pass

    def build_vector(self, parameter, period, discretisation):
        return np.hstack((parameter, discretisation))

    def get_parameter(self, continuation_vec):
        return continuation_vec[0]

    def get_discretisation(self, continuation_vec):
        return continuation_vec[1:]


class AutonymousCBC(CBC):
    ##### TODO remove references to default_phase_condition

    def __init__(self, continuation_target, discretisor):
        super().__init__(continuation_target, discretisor, self.phase_condition)

    def phase_condition(self, current_vec, last_vec):
        current_model = self.discretisor.undiscretise(
            self.get_discretisation(current_vec), 1
        )
        reference_gradient = self.discretisor.derivative_undiscretise(
            self.get_discretisation(last_vec), 1
        )
        return scipy.integrate.quad(
            lambda t: np.inner(current_model(t), reference_gradient(t)),
            0,
            1,
            limit=250,
        )[0]

    def build_vector(self, parameter, period, discretisation):
        return np.hstack((parameter, period, discretisation))

    def get_parameter(self, continuation_vec):
        return continuation_vec[0]

    def get_period(self, continuation_vec):
        return continuation_vec[1]

    def get_discretisation(self, continuation_vec):
        return continuation_vec[2:]
