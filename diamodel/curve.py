from dataclasses import dataclass, astuple
import numpy as np
from scipy.stats import gengamma

# Each curve is uniquely identified by its start tick and amount
CurveKey = tuple[int, float]  # (start, amount)


@dataclass
class Curve:
    """Represents probabilistic response curve."""

    start: int
    duration: int
    peak: float | np.ndarray
    tail: float | np.ndarray
    amount: float
    corr: float | np.ndarray
    sens: float | np.ndarray

    @property
    def key(self) -> CurveKey:
        return (self.start, self.amount)

    def to_tuple(self) -> tuple:
        return astuple(self)

    def at(self, tick: int, gscale: float, nsamples: int) -> np.ndarray | float:
        """Calculates probabilistic curve value at `tick`"""
        # check if `tick` falls within the curve range: [self.start, self.start + self.duration)
        if tick < self.start:
            return np.zeros((nsamples,), dtype=float) if nsamples > 1 else 0.0

        if tick >= (self.start + self.duration):
            return np.zeros((nsamples,), dtype=float) if nsamples > 1 else 0.0

        # convert `tick` to gamma domain
        x = (tick - self.start + 1) * gscale

        g = gengamma.pdf(x, self.peak, self.tail, scale=1)

        # "amplitude"
        a = gscale * (self.amount + self.corr) * self.sens

        # curve values at tick
        y = a * g

        if nsamples > 1:
            if np.isscalar(y):
                y = np.full((nsamples,), y, dtype=float)
            assert y.shape == (nsamples,)

        return y
