import abc
import numpy as np
import scipy.interpolate
import splines

"""
TODOs:
    * README
    * Robustness / error checking
    * Test the adaptive discretisor; it'll probably have issues with knots producing invalid splines curves, when there aren't enough data to fit from; this is an issue with splines.py
    * Unit tests
"""


class _Discretisor(abc.ABC):
    """
    Abstract class definition for a discretisor. A discretisor must
    implement the discretise and undiscretise methods.
    """

    def __init__(self, dsize):
        """
        Initialiser for all discretisors.

            dsize : int > 0
                Some measure of the discretisation size, eg. number of
                harmonics, or number of basis functions. Exact
                interpretation depends on the chosen discretisation
                method.
        """
        self.dsize = dsize

    @abc.abstractmethod
    def discretise(self, signal, period):
        """
        Take a recorded signal, with evenly-spaced samples. Compute a
        discretisation of the signal, using some implementing
        discretisation method.

            signal : 2-by-n float array
                The signal to discretise. Array must be of form
                [[signal ts], [signal ys]].

        Returns discretisation, period. Period is the time taken for
        the signal to complete one cycle, discretisation is a
        unit-period representation that can be used either with the
        undiscretise method to produce a control target, or for
        continuation.
        """
        pass

    @abc.abstractmethod
    def undiscretise(self, discretisation, period):
        """
        Take a discretisation of a signal, and the period of the
        original signal. Produce an evaluable model to represent the
        signal described in the discretisation.

            discretisation : 1-by-n float array
                Some discretisation, eg. that returned by
                self.discretise, or as computed by a Newton update in
                a continuation routine.

            period : float > 0
                The desired period of the undiscretised signal.

        Returns a function of signature model(ts), which gives the
        signal value at times ts.
        """
        pass


class _AdaptiveDiscretisor(_Discretisor, abc.ABC):
    """
    Abstract class definition for an adaptive discretisor. The
    abstract discretisor must implement the discretise and
    undiscretise methods of a _Discretisor, as well as a method for
    constructing the initial discretisation procedure, and updating
    that discretisation procedure for new data.
    """

    def __init__(self, signal, period, dsize, **kwargs):
        """
        Given some signal, and the period of the signal, initialize a
        discretisation scheme from the signal, eg. produce a set of
        basis functions or a mesh etc., based on the data.

            signal : 2-by-n float array
                Signal to initialise the discretisation scheme from.
                signal[0] is the independent (time-like) variable.
                signal[1] is the dependent (voltage / position /
                etc.)-like variable.

            period : float > 0
                Time taken for the signal to undergo a single
                oscillation cycle.

            dsize : int > 0
                Some measure of the discretisation size, eg. number of
                harmonics, or number of basis functions. Exact
                interpretation depends on the chosen discretisation
                method.

            **kwargs :
                Any extra kwargs necessary for the specific
                implementation.
        """
        super().__init__(dsize)
        self.mesh = self._initialise_discretisation_scheme(signal, period, **kwargs)

    @abc.abstractmethod
    def _initialise_discretisation_scheme(self, signal, period, **kwargs):
        """
        Given some signal, and the period of the signal, produce the
        appropriate discretisation data for the signal, eg. a mesh or
        set of knots, based on the passed signal.

            signal : 2-by-n float array
                Signal to initialise the discretisation scheme from.
                signal[0] is the independent (time-like) variable.
                signal[1] is the dependent (voltage / position /
                etc.)-like variable.

            period : float > 0
                Time taken for the signal to undergo a single
                oscillation cycle.

            **kwargs :
                Any extra kwargs necessary for the specific
                implementation.
        """
        pass

    @abc.abstractmethod
    def update_discretisation_scheme(self, signal, period, **kwargs):
        """
        Given a new signal, and the period of the signal, update the
        existing discretisation scheme for the new signal, eg. produce
        an updated set of basis functions or meshpoints, based on the
        new data.

            signal : 2-by-n float array
                Signal to initialise the new discretisation scheme
                from. signal[0] is the independent (time-like)
                variable. signal[1] is the dependent (voltage /
                position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to undergo a single
                oscillation cycle.

            **kwargs :
                Any extra kwargs necessary for the specific
                implementation.
        """
        pass


class FourierDiscretisor(_Discretisor):
    """
    Implement a _Discretisor using a Fourier discretisation. This
    represents a periodic signal by the coefficients of its trucated
    Fourier series.

        dsize : int > 0
            Number of Fourier harmonics. Total discretisation size is
            2*dsize+1.
    """

    def discretise(self, signal, period):
        """
        Given a set of datapoints, fit a truncated Fourier series of given
        period.

            signal : 2-by-n float array
                signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                time taken for the signal to complete a single
                oscillation

        Returns a 1-by-(2k+1) array [a0, ai, bi], for the DC offset,
        cosine coefficients, and sine coefficients of the fitted
        truncated Fourier series.
        """
        ones_mat = np.ones((self.dsize, signal.shape[1]))
        # Transpose of trig args in eval function
        # Matrix M[r,c] = (c+1) * ts[r]; cols give harmonics, rows give ts
        trig_args = (
            2
            * np.pi
            / period
            * (np.arange(1, self.dsize + 1).reshape((-1, 1)) * ones_mat).T
            * signal[0].reshape((-1, 1))
        )
        one = np.ones((signal.shape[1], 1))
        design_mat = np.hstack((one, np.cos(trig_args), np.sin(trig_args)))
        lsq_solution = np.linalg.lstsq(design_mat, signal[1].reshape((-1, 1)), rcond=None)
        lsqfit = lsq_solution[0].reshape((-1,))
        a0, ai, bi = lsqfit[0], lsqfit[1:self.dsize + 1], lsqfit[self.dsize + 1:]
        return np.hstack((a0, ai, bi))

    def undiscretise(self, discretisation, period):
        """
        Produce a model to evaluate a truncated Fourier series.

            discretisation : 1-by-(2k+1) float array
                Float array of form [a0, ai, bi], for
                    a0 : DC offset
                    ai : Cosine harmonics coefficients
                    bi : Sine harmonics coefficients

            period : float > 0
                Time taken for the signal to complete a single oscillation

        Returns a function that evaluates the truncated Fourier series at
        a 1-by-n array of time points.
        """
        a0, ai, bi = (
            discretisation[0],
            discretisation[1:self.dsize+1],
            discretisation[self.dsize + 1:],
        )

        def deparameterized(ts):
            ts = np.array(ts, dtype=float).reshape((-1,))
            # Transpose of fitting trig args
            ones_mat = np.ones((len(ts), self.dsize))
            trig_args = (
                2
                * np.pi
                / period
                * (ts.reshape((-1, 1)) * ones_mat).T
                * np.arange(1, self.dsize + 1).reshape((-1, 1))
            )
            coss = np.matmul(ai, np.cos(trig_args))
            sins = np.matmul(bi, np.sin(trig_args))
            return np.squeeze(a0 + sins + coss)

        return deparameterized


class SplinesDiscretisor(_Discretisor):
    """
    Implement a _Discretisor using a periodic BSpline discretisation,
    with knots evenly spaced across the data period, and held constant
    throughout. This represents a periodic signal by the coefficients
    of its BSpline curve.

        dsize : int > 0
            Number of interior knots
    """

    def __init__(self, dsize):
        """
        Initialiser for the new splines discretisor. Knots are kept
        constant throughout, so BSplines are computed at
        initialisation.

            dsize : int > 0
                Some measure of the discretisation size, eg. number of
                harmonics, or number of basis functions. Exact
                interpretation depends on the chosen discretisation
                method.
        """
        self.dsize = dsize
        interior_knots = np.linspace(0, 1, dsize + 2)[1:-1]
        self._spline = splines.PeriodicSpline(interior_knots)

    def discretise(self, signal, period):
        """
        Given some data, find the BSpline coefficients that discretise
        the data. Knots are uniformly spaced across the interior of
        the signal period.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Returns a 1-by-k float array of the BSpline coefficients that
        discretise the data.
        """
        return self._spline.fit(signal[0], signal[1], period)

    def undiscretise(self, discretisation, period):
        """
        Given a discretisation, construct a cubic periodic splines
        model, and return a function that evaluates it.

            discretisation : 1-by-n float array
                Spline discretisation, as computed by
                get_spline_discretisation_from_knots.

            period : float>0
                Desired period of the signal.

        Returns a function that evaluates the fitted splines model.
        """
        return lambda ts: self._spline(ts, discretisation, period)


class OldSplinesDiscretisor(_Discretisor):
    """
    Implement a _Discretisor using a periodic BSpline discretisation,
    with knots evenly spaced across the data period, and held constant
    throughout. This represents a periodic signal by the coefficients
    of its BSpline curve.

        dsize : int > 0
            Number of interior knots
    """

    def discretise(self, signal, period):
        """
        Given some data, find the BSpline coefficients that discretise
        the data. Knots are uniformly spaced across the interior of
        the signal period.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Returns a 1-by-k float array of the BSpline coefficients that
        discretise the data.
        """
        stacked_ts, stacked_ys = self._stack_periods(signal[0], signal[1], period)
        interior_knots = np.linspace(0, 1, self.dsize + 2)[1:-1]
        _, betas, _ = scipy.interpolate.splrep(
            stacked_ts, stacked_ys, t=interior_knots, per=True, xb=0.0, xe=1.0
        )
        return betas[:-4]

    def undiscretise(self, discretisation, period):
        """
        Given a discretisation, construct a cubic periodic splines
        model, and return a function that evaluates it.

            discretisation : 1-by-n float array
                Spline discretisation, as computed by
                get_spline_discretisation_from_knots.

            period : float>0
                Desired period of the signal.

        Returns a function that evaluates the fitted splines model.
        """
        interior_knots = np.linspace(0, 1, self.dsize + 2)[1:-1]
        full_knots = _get_full_knots(interior_knots)
        full_discretisation = np.hstack((discretisation, np.zeros((4,))))
        spline = (full_knots, full_discretisation, 3)

        def model(x):
            return scipy.interpolate.splev(x/period, spline)

        return lambda x: model(np.mod(x, period))

    def _stack_periods(self, data_t, data_y, period):
        """
        Private helper function. Rescale signal time-samples to a
        phase variable on the unit interval, to make fitting a splines
        model easier.

            data_t : 1-by-n float array
                Time-like variable for the signal; rescaled to a single
                period.

            data_y : 1-by-n float array
                Dependent variable for the signal

            period : float
                Time taken for the signal to complete a single full
                oscillation

        Returns the signal with times rescaled to unit-phases, and sorted
        in increasing t.
        """
        ts = np.mod(data_t / period, 1)
        sort_indices = np.argsort(ts)
        return ts[sort_indices], data_y[sort_indices]


class AdaptiveSplinesDiscretisor(_AdaptiveDiscretisor):
    def discretise(self, signal, period):
        """
        Given some data, find the BSpline coefficients that discretise
        the data. Knots are uniformly spaced across the interior of
        the signal period.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Returns a 1-by-k float array of the BSpline coefficients that
        discretise the data.
        """
        return self._spline.fit(signal[0], signal[1], period)

    def undiscretise(self, discretisation, period):
        """
        Given a discretisation, construct a cubic periodic splines
        model, and return a function that evaluates it.

            discretisation : 1-by-n float array
                Spline discretisation, as computed by
                get_spline_discretisation_from_knots.

            period : float>0
                Desired period of the signal.

        Returns a function that evaluates the fitted splines model.
        """
        return lambda ts: self._spline(ts, discretisation, period)

    def _initialise_discretisation_scheme(self, signal, period, n_tries=50):
        """
        Given some periodic data, find the set of interior knots that
        provide a best-possible periodic splines model to the data (in
        the least-squares sense). This is done by starting with a
        randomly distributed set of knots, then attempting a numerical
        optimization on the knot set, to maximise goodness-of-fit. To
        avoid local minima, this procedure is repeated numerous times,
        with the best knots being recorded.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

            n_tries : int > 0
                Number of times to restart the optimisation, to avoid
                local minima. More is better, but slower.

        Returns an interior knot vector that produces the
        lowest-residual splines fit to the provided data.
        """
        def loss(interior_knots):
            # Prediction error under current knots
            spline = splines.PeriodicSpline(np.sort(interior_knots))
            residuals = signal[1] - spline(signal[0])
            return np.linalg.norm(residuals) ** 2

        # Uniform distribution for initial knots
        initial_knots = scipy.stats.uniform().rvs((n_tries, self.dsize))

        # Keep re-optimizing; store best result
        best_loss = np.inf
        best_knots = None
        for k in initial_knots:
            opti = scipy.optimize.minimize(loss, k, tol=1e-3)
            if opti.fun < best_loss:
                best_loss = opti.fun
                best_knots = opti.x
        self._spline = splines.PeriodicSpline(np.sort(best_knots))
        return np.sort(best_knots)

    def update_discretisation_scheme(self, signal, period):
        """
        Given an existing set of interior knots, re-optimize the knots
        to the new optimal value, for the given signal.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Stores the result of an optimization procedure over the knot
        sets, to minimize fitting error, starting from the current
        knots.
        """
        def loss(interior_knots):
            # Prediction error under current knots
            spline = splines.PeriodicSpline(np.sort(interior_knots))
            residuals = signal[1] - spline(signal[0])
            return np.linalg.norm(residuals) ** 2

        opti = scipy.optimize.minimize(loss, self.mesh)
        self.mesh = np.sort(opti.x)
        self._spline = splines.PeriodicSpline(self.mesh)
        return opti.x


class OldAdaptiveSplinesDiscretisor(_AdaptiveDiscretisor):
    """
    TODO docstring
    """

    def discretise(self, signal, period):
        """
        Given some data and a set of pre-computed interior knots, find
        the BSpline coefficients that discretise the data.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Returns a 1-by-k float array of the BSpline coefficients that
        discretise the data.
        """
        stacked_ts, stacked_ys = self._stack_periods(signal[0], signal[1], period)
        _, betas, _ = scipy.interpolate.splrep(
            stacked_ts, stacked_ys, t=self.mesh, per=True, xb=0.0, xe=1.0
        )
        # Last 4 betas are always zero, so drop them
        return betas[:-4]

    def undiscretise(self, discretisation, period):
        """
        Given a discretisation, fit a cubic periodic splines model,
        and return a function that evaluates it.

            discretisation : 1-by-n float array
                Spline discretisation, as computed by
                get_spline_discretisation_from_knots.

            period : float>0
                Desired period of the signal.

        Returns a function that evaluates the fitted splines model.
        """
        full_discretisation = np.hstack((discretisation, np.zeros((4,))))
        full_knots = _get_full_knots(self.mesh)
        spline = (full_knots, full_discretisation, 3)

        def model(x):
            return scipy.interpolate.splev(x/period, spline)

        return lambda x: model(np.mod(x, period))

    def _initialise_discretisation_scheme(self, signal, period, n_tries=50):
        """
        Given some periodic data, find the set of interior knots that
        provide a best-possible periodic splines model to the data (in
        the least-squares sense). This is done by starting with a
        randomly distributed set of knots, then attempting a numerical
        optimization on the knot set, to maximise goodness-of-fit. To
        avoid local minima, this procedure is repeated numerous times,
        with the best knots being recorded.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

            n_tries : int > 0
                Number of times to restart the optimisation, to avoid
                local minima. More is better, but slower.

        Returns an interior knot vector that produces the
        lowest-residual splines fit to the provided data.
        """
        stacked_ts, stacked_ys = self._stack_periods(signal[0], signal[1], period)

        def loss(knotvec):
            # Prediction error under current knots
            try:
                full_spline = scipy.interpolate.splrep(
                    stacked_ts, stacked_ys, t=np.sort(knotvec), per=True, xb=0.0, xe=1.0
                )
                residuals = stacked_ys - scipy.interpolate.splev(stacked_ts, full_spline)
                return np.linalg.norm(residuals) ** 2
            except ValueError:
                # If knots are invalid, return infinite error
                return np.inf

        # Uniform distribution for initial knots
        initial_knots = scipy.stats.uniform().rvs((n_tries, self.dsize))

        # Keep re-optimizing; store best result
        best_loss = np.inf
        best_knots = None
        for k in initial_knots:
            opti = scipy.optimize.minimize(loss, k, tol=1e-3)
            if opti.fun < best_loss:
                best_loss = opti.fun
                best_knots = opti.x
        return best_knots

    def update_discretisation_scheme(self, signal, period):
        """
        Given an existing set of interior knots, re-optimize the knots
        to the new optimal value, for the given signal.

            signal : 2-by-n float array
                Signal to discretise. signal[0] is the independent
                (time-like) variable. signal[1] is the dependent
                (voltage / position / etc.)-like variable.

            period : float > 0
                Time taken for the signal to complete a single
                oscillation.

        Stores the result of an optimization procedure over the knot
        sets, to minimize fitting error, starting from the current
        knots.
        """
        stacked_ts, stacked_ys = self._stack_periods(signal[0], signal[1], period)

        def loss(knotvec):
            try:
                full_spline = scipy.interpolate.splrep(
                    stacked_ts, stacked_ys, t=np.sort(knotvec), per=True, xb=0.0, xe=1.0
                )
                residuals = stacked_ys - scipy.interpolate.splev(stacked_ts, full_spline)
                return np.linalg.norm(residuals) ** 2
            except ValueError:
                return np.inf

        opti = scipy.optimize.minimize(loss, self.mesh)
        self.mesh = opti.x
        return opti.x

    def _stack_periods(self, data_t, data_y, period):
        """
        Private helper function. Rescale signal time-samples to a
        phase variable on the unit interval, to make fitting a splines
        model easier.

            data_t : 1-by-n float array
                Time-like variable for the signal; rescaled to a single
                period.

            data_y : 1-by-n float array
                Dependent variable for the signal

            period : float
                Time taken for the signal to complete a single full
                oscillation

        Returns the signal with times rescaled to unit-phases, and sorted
        in increasing t.
        """
        ts = np.mod(data_t / period, 1)
        sort_indices = np.argsort(ts)
        return ts[sort_indices], data_y[sort_indices]


def _get_full_knots(interior_knots):
    """
    Given a set of interior knots, calculate the exterior knots
    required to form a periodic BSpline curve on the unit interval.

        interior_knots : ndarray
            Interior knots for the spline curve, distributed within
            the unit interval.

    Returns an augmented knot set containing both interior and
    periodic exterior knots.
    """
    first_exterior_knots = np.hstack((interior_knots[-3:], [1])) - 1
    last_exterior_knots = np.hstack(([0], interior_knots[:3])) + 1
    return np.hstack((first_exterior_knots, interior_knots, last_exterior_knots))
