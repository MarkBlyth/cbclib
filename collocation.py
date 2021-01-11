import numpy as np
import abc
import scipy.integrate
import continuation
import splines

# Error tolerance for floating-point equality
FTOL = 1e-6

# TODO adapt to nonautonymous systems?


class CollocationMesh(abc.ABC):
    """
    Allows computation on collocation meshs. Contains a list of
    subintervals, with collocation basis functions being defined
    across each subinterval. Implementations of the CollocationMesh
    abstract class define these basis functions.

        mesh : 1d array
            List of evenly spaced timepoints that separate up the unit
            interval into a mesh.

        collocation_points : 1d array
            List of timepoints corresponding to the roots of the
            appropriately ordered Legendre polynomial, scaled to within
            each mesh subinterval. These are the points at which the
            collocation equations are to be evaluated.

        sub_intervals : list
            List of CollocationSubinterval objects, representing each
            subinterval of the collocation mesh.

        size : int
            Size of the collocation mesh.

        shape : tuple
            Shape of the collocation mesh.

        # TODO other class attributes
    """

    class CollocationSubinterval:
        """
        Holds useful information about each subinterval of a
        collocation mesh.

            boundary_lo : float
                Left-hand endpoint of the subinterval.

            boundary_hi : float
                Right-hand endpoint of the subinterval.

            collocation_points : 1d array
                Legendre-root collocation points within the
                subinterval.
        """

        def __init__(self, boundary_lo, boundary_hi, roots):
            self.boundary_lo = boundary_lo
            self.boundary_hi = boundary_hi
            self.collocation_points = (boundary_hi - boundary_lo) * (
                roots + 1
            ) / 2 + boundary_lo
            self.mesh = np.linspace(boundary_lo, boundary_hi, roots.size + 1)

        def in_subinterval(self, t):
            """
            For ndarray t, returns an ndarray where each element is
            True if the associated t element is within the interval,
            and False otherwise.
            """
            return np.logical_and(
                t >= self.boundary_lo - FTOL, t <= self.boundary_hi + FTOL
            )

    def __init__(self, mesh_size, order):
        """
        Mesh size = number of subintervals.
        """
        self.size = mesh_size + 1
        self.shape = (mesh_size + 1,)
        self.n_subintervals = mesh_size
        self.order = order
        self.mesh = np.linspace(0, 1, mesh_size + 1)
        self.full_mesh = np.zeros(mesh_size * (order + 1))

        legendre_coefficients = np.zeros(order + 1)
        legendre_coefficients[-1] = 1
        legendre_roots = np.polynomial.legendre.legroots(legendre_coefficients)
        self.sub_intervals = []
        self.collocation_points = np.zeros(mesh_size * order)
        for i in range(mesh_size):
            subinterval = self.CollocationSubinterval(
                self.mesh[i], self.mesh[i + 1], legendre_roots
            )
            self.sub_intervals.append(subinterval)
            self.collocation_points[
                i * order : (i + 1) * order
            ] = subinterval.collocation_points
            self.full_mesh[i * (order + 1) : (i + 1) * (order + 1)] = subinterval.mesh

    def eval(self, coefficients, ts=None):
        """
        Evaluate the user-defined collocation function. If timepoints
        are None, automatically use the collocation points.

            coefficients : size (n, m, N) array
                For n collocation subintervals, the [i, :, :]'th
                element specifies the (m, N) array of interpolating
                basis function coefficients within the i'th
                collocation subinterval. The [i, j, :]'th elements
                specify the coefficients for each dimension, at the
                j'th subinterval-meshpoint.

            ts : 1d array, None
                Timepoints at which to evaluate the basis function. If
                None, the collocation points are used.
        """
        if ts is None:
            ts = self.collocation_points
        try:
            out_size = (ts.size, coefficients[0, 0, :].size)
        except AttributeError:
            out_size = (1, coefficients[0, 0, :].size)
            ts = np.array([ts])
        output = np.zeros(out_size)
        for subinterval, coeffs in zip(self.sub_intervals, coefficients):
            output[subinterval.in_subinterval(ts)] = self._eval(
                subinterval.mesh, coeffs, ts[subinterval.in_subinterval(ts)]
            )
        return output

    def derivative(self, coefficients, ts=None):
        """
        Evaluate the user-defined collocation function
        time-derivative. If timepoints are None, automatically use the
        collocation points.

            coefficients : size (n, m, N) array
                For n collocation subintervals, the [i, :, :]'th
                element specifies the (m, N) array of interpolating
                basis function coefficients within the i'th
                collocation subinterval. The [i, j, :]'th elements
                specify the coefficients for each dimension, at the
                j'th subinterval-meshpoint.

            ts : 1d array, None
                Timepoints at which to evaluate the basis function. If
                None, the collocation points are used.
        """
        if ts is None:
            ts = self.collocation_points
        try:
            out_size = (ts.size, coefficients[0, 0, :].size)
        except AttributeError:
            out_size = (1, coefficients[0, 0, :].size)
            ts = np.array([ts])
        output = np.zeros(out_size)
        for subinterval, coeffs in zip(self.sub_intervals, coefficients):
            output[subinterval.in_subinterval(ts)] = self._derivative(
                subinterval.mesh, coeffs, ts[subinterval.in_subinterval(ts)]
            )
        return output

    @abc.abstractmethod
    def remesh(self, mesh_size, order, coefficients):
        """
        TODO Come up with a new mesh object and list of coefficients,
        appropriate for the new mesh parameters. Useful for adaptive
        methods. The output coefficients should be corrected using a
        solver, and hence the re-meshing is only an approximate
        operation.
        """
        pass

    @abc.abstractmethod
    def _eval(self, submesh, coefficients, ts):
        """
        User-implemented. Evaluate the user-defined collocation
        function, for a set of basis funciton coefficients, at a set
        of timepoints. If timepoints are None, automatically use the
        collocation points. Evaluates within a single interval, ie. no
        need to split ts, coeffs up into multiple subintervals.

            submesh : 1d array
                Mesh on the current subinterval

            coefficients : size (m, n) array
                Array of coefficients, of form [[coeffs1], [coeffs2],
                ...], where [coeffs_i] specifies the N-dimensional
                coefficients at the i'th submesh point.

            ts : 1d array
                Timepoints at which to evaluate the collocation
                function.
        """
        pass

    @abc.abstractmethod
    def _derivative(self, submesh, coefficients, ts):
        """
        User-implemented. Evaluate the time-derivative of the
        user-defined collocation function, for a set of basis funciton
        coefficients, at a set of timepoints. If timepoints are None,
        automatically use the collocation points. Evaluates within a
        single interval, ie. no need to split ts, coeffs up into
        multiple subintervals.

            submesh : m-element array
                Mesh on the current subinterval, containing m meshpoints

            coefficients : size (m, n) array
                Array of coefficients, of form [[coeffs1], [coeffs2],
                ...], where [coeffs_i] specifies the N-dimensional
                coefficients at the i'th submesh point.

            ts : 1d array
                Timepoints at which to evaluate the collocation
                function time derivative.
        """
        pass


class KroghMesh(CollocationMesh):
    def __init__(self, mesh_size, order):
        super().__init__(mesh_size, order)
        # TODO precompute interpolants at init, then select
        # appropriate sets of basis functions at runtime, to avoid
        # recomputing interpolants every single time the mesh is
        # evaluated

    def _eval(self, submesh, coefficients, ts):
        return scipy.interpolate.krogh_interpolate(submesh, coefficients, ts)

    def _derivative(self, submesh, coefficients, ts):
        return scipy.interpolate.krogh_interpolate(submesh, coefficients, ts, der=1)

    def remesh(self, mesh_size, order, coefficients):
        raise NotImplementedError


class BSplineMesh(CollocationMesh):
    def __init__(self, mesh_size, order=3):
        """
        BSpline collocation doesn't divide the data into subintervals.
        Instead, it maintains a single subinterval over the entire BVP
        domain [0,1]. The subinterval mesh is taken as a knot list for
        the spline functions. mesh_size therefore specifies the number
        of BSpline basis functions, and order specifies the order of
        the splines, similarly to with polynomial collocation. Note
        that BSpline collocation is only implemented for third-order
        (cubic) BSplines, so choosing an order!=3 will raise a
        NotImplementedError. Also, for third-order periodic splines,
        we require at least 4 BSpline functions, so choosing
        mesh_size<4 will raise a ValueError.
        """
        if mesh_size < 4:
            raise ValueError("Must have at least 4 cubic BSpline functions")
        if order != 3:
            raise NotImplementedError("BSpline collocation is only implemented for third-order BSplines")
        super().__init__(1, mesh_size)
        self.splines = splines.PeriodicSpline(self.full_mesh[1:-1])

    def _eval(self, submesh, coefficients, ts):
        # TODO make sure this works for multidimensional coefficients
        # TODO make sure we get passed only a 2d coefficient matrix
        return self.splines.eval(ts, coefficients, 1)

    def _derivative(self, submesh, coefficients, ts):
        # TODO make sure this works for multidimensional coefficients
        # TODO make sure we get passed only a 2d coefficient matrix
        return self.splines.derivative(ts, coefficients, 1)

    def remesh(self, mesh_size, order, coefficients):
        raise NotImplementedError


class NumericalContinuation(continuation.Continuation):
    """
    Implements a Continuation class using orthogonal collocation for
    the numerical continuation of periodic orbits in autonymous
    systems.
    """

    def __init__(self, ode_RHS, collocation_mesh):
        # TODO modify this for consistency with the Continuation superclass
        self.ode_RHS = ode_RHS
        self.col_func = collocation_mesh

    def get_parameter(self, continuation_vec):
        return continuation_vec[0]

    def get_period(self, continuation_vec):
        return continuation_vec[1]

    def get_discretisation(self, continuation_vec):
        return np.array(continuation_vec[2:]).reshape(
            (self.col_func.n_subintervals, self.col_func.order + 1, -1)
        )

    def _continuation_system(
        self, continuation_vector, last_solution, prediction, secant
    ):
        """
        TODO
        """
        last_coefficients = self.get_discretisation(last_solution)

        def v_dot(t):
            ####### TODO THIS RELIES ON A FIXED MESH
            ####### TODO CHANGE TO ALLOW FOR ADAPTIVE MESH
            return self.col_func.derivative(last_coefficients, t)

        collocations = self.collocation_system(continuation_vector, v_dot)
        pseudo_arclength_condition = np.dot(continuation_vector - prediction, secant)
        return np.hstack((collocations, pseudo_arclength_condition))

    def collocation_system(self, continuation_vector, v_dot):
        """
        Set up a collocation system, without the pseudo-arclength
        constraint. Given a phase reference solution and some initial
        estimate of the current continuation vector, this system can
        be solved to locate a limit cycle. Returns an array of zeros
        when the collocation and phase equations are satisfied,
        indicating that a LC has been found.

            continuation_vector : 1d array
                Vector encoding the current limit cycle solution, or
                estimate thereof.

            v_dot : func
                Time-derivative of the reference signal used by the
                phase condition.
        """
        period = self.get_period(continuation_vector)
        parameter = self.get_parameter(continuation_vector)
        coefficients = self.get_discretisation(continuation_vector)

        target_LHS = self.col_func.derivative(coefficients)
        target_RHS = period * self.ode_RHS(self.col_func.eval(coefficients), parameter,)
        collocations = (target_LHS - target_RHS).reshape(-1)
        continuity = self._continuity_error(coefficients).reshape(-1)
        phase_condition = self._phase_condition(coefficients, v_dot)
        return np.hstack((collocations, continuity, phase_condition))

    def _continuity_error(self, coefficients):
        """
        Evaluate continuity constraint, including continuity between
        periods (ie. periodicity constraint). Mandates that the
        section of basis function within each subinterval meets the
        section of basis function within adjacent subintervals at the
        subinterval boundary.

            coefficients : TODO

        Returns a TODO dimensional array representing the difference
        between each adjacent polynomial section at the boundaries.
        """
        out_size = (self.col_func.size - 1, coefficients[0, 0, :].size)
        errors = np.zeros(out_size)
        for i in range(self.col_func.n_subintervals):
            # Evaluate last polynomial at last meshpoint, not first
            RHS_meshpoint = 1 if i == 0 else self.col_func.mesh[i]
            RHS = self.col_func._eval(
                self.col_func.sub_intervals[i - 1].mesh,
                coefficients[i - 1],
                RHS_meshpoint,
            )
            LHS = self.col_func._eval(
                self.col_func.sub_intervals[i].mesh,
                coefficients[i],
                self.col_func.mesh[i],
            )
            errors[i] = LHS - RHS
        return errors

    def _phase_condition(self, coefficients, v_dot):
        """
        Integral phase condition. Selects the periodic orbit that
        minimises the distance from some reference periodic orbit
        v(t).

            coefficients: ndarray
                Coefficients of the current collocation solution.

            v_dot : func
                Time-derivative of the reference signal.
        """

        def objective(t):
            return np.inner(self.col_func.eval(coefficients, t), v_dot(t))

        return scipy.integrate.quad(objective, 0, 1, limit=100)[0]

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
        period = self.get_period(continuation_vector)
        parameter = self.get_parameter(continuation_vector)
        discretisation = self.get_discretisation(continuation_vector)
        initial_cond = discretisation[0, 0, :]

        def ode(t, y):
            return self.ode_RHS(y, parameter)

        soln = scipy.integrate.solve_ivp(
            ode, [0, period], initial_cond, atol=1e-9, rtol=1e-9
        )
        return [soln.t, soln.y]


class BSplineContinuation(NumericalContinuation):
    """
    The same as NumericalContinuation, only we've dropped the
    continuity / periodicity requirement, since it's automatically
    satisfied by our periodic BSplines.
    """

    def get_discretisation(self, continuation_vec):
        return np.array(continuation_vec[2:]).reshape(
            (self.col_func.n_subintervals, self.col_func.order, -1)
        )

    def collocation_system(self, continuation_vector, v_dot):
        """
        Set up a collocation system, without the pseudo-arclength
        constraint. Given a phase reference solution and some initial
        estimate of the current continuation vector, this system can
        be solved to locate a limit cycle. Returns an array of zeros
        when the collocation and phase equations are satisfied,
        indicating that a LC has been found.

            continuation_vector : 1d array
                Vector encoding the current limit cycle solution, or
                estimate thereof.

            v_dot : func
                Time-derivative of the reference signal used by the
                phase condition.
        """
        period = self.get_period(continuation_vector)
        parameter = self.get_parameter(continuation_vector)
        coefficients = self.get_discretisation(continuation_vector)

        target_LHS = self.col_func.derivative(coefficients)
        target_RHS = period * self.ode_RHS(self.col_func.eval(coefficients), parameter,)
        collocations = (target_LHS - target_RHS).reshape(-1)
        phase_condition = self._phase_condition(coefficients, v_dot)
        return np.hstack((collocations, phase_condition))
