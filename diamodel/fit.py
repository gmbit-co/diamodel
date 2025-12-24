from dataclasses import dataclass, asdict
import pickle
from datetime import datetime
from functools import partial
from typing import Optional, Any

import numpy as np
import pandas as pd
from .curve import Curve, CurveKey
from .config import Config

__all__ = ["Fit"]


@dataclass
class Fit:
    """Represents joint posterior distribution of model params."""

    # Params order has to be in sync with stan code
    GLOBAL_PARAMS = ["isens", "ipeak", "sigma", "rho", "csens", "cpeak"]

    # insulin params
    isens: np.ndarray  # (nsamples,)
    ipeak: np.ndarray  # (nsamples,)

    rho: np.ndarray  # (nsamples,)
    sigma: np.ndarray  # (nsamples,)

    # carbs params
    csens: Optional[np.ndarray] = None  # (nsamples,)
    cpeak: Optional[np.ndarray] = None  # (nsamples,)

    # bolus
    carbs: Optional[pd.Series] = None  # (nc,)
    insulin: Optional[pd.Series] = None  # (ni,)

    # carb curves params
    cpeaki: Optional[np.ndarray] = None  # (nsamples,nc)
    ccorri: Optional[np.ndarray] = None  # (nsamples,nc)

    chunks: Optional[pd.Series] = None  # (nchunks,)
    bg0i: Optional[np.ndarray] = None  # (nsamples,nchunks)

    lp__: Optional[np.ndarray] = None  # (nsamples,), log-likelihood
    fit_at: Optional[datetime] = None

    @staticmethod
    def default(
        cfg: Config,
        std_perc: float = 0.1,
        overrides: Optional[dict[str, float]] = None,
        bolus_df: Optional[pd.DataFrame] = None,
    ) -> "Fit":
        """Default prior"""
        # mean prior values
        fit: dict[str, Any] = {
            "isens": 5.0,  # mmol/L / unit
            "ipeak": 90.0 / cfg.dt * cfg.gscale,
            "sigma": 1.0,  # mmol/L
            "rho": 0.9,
            "csens": 0.3,  # mmol/L / gram
            "cpeak": 120.0 / cfg.dt * cfg.gscale,
        }

        if overrides is not None:
            fit.update(overrides)

        n = cfg.nsamples
        for name in Fit.GLOBAL_PARAMS:
            mean = fit[name]
            fit[name] = clipped_normal(mean, mean * std_perc, n)

        if bolus_df is not None:
            # assign bolus
            fit["carbs"] = bolus_df["carbs"][bolus_df["carbs"] > 0.0]
            fit["insulin"] = bolus_df["insulin"][bolus_df["insulin"] > 0.0]

            # assign carbs curve params

            nc = len(fit["carbs"])
            cpeak = np.mean(fit["cpeak"])
            fit["cpeaki"] = np.zeros((n, nc), dtype=float)
            fit["ccorri"] = np.zeros((n, nc), dtype=float)
            for i in range(nc):
                j = i % 3  # selects [fast, medium, slow] carbs
                _cpeak = cpeak + [-1.0, 0.0, 1.0][j]
                fit["cpeaki"][:, i] = clipped_normal(_cpeak, _cpeak * std_perc, n)

        return Fit(**fit)

    @staticmethod
    def from_blob(blob: bytes) -> "Fit":
        return pickle.loads(blob)

    def __post_init__(self):
        n = self.sigma.shape[0]  # nsamples

        self.carbs = pd.Series([], dtype=float) if self.carbs is None else self.carbs
        self.insulin = pd.Series([], dtype=float) if self.insulin is None else self.insulin

        arrays = [getattr(self, name) for name in self.GLOBAL_PARAMS]
        arrays = [arr for arr in arrays if arr is not None]
        assert all(map(lambda arr: isinstance(arr, np.ndarray), arrays))
        assert all(arr.shape == (n,) for arr in arrays)

        if self.chunks is not None:
            assert self.bg0i is not None and self.bg0i.shape == (n, len(self.chunks))

        # check consistency of carbs params
        nc = len(self.carbs)
        if nc > 0:
            assert self.cpeaki is not None and self.cpeaki.shape == (n, nc)
            assert self.ccorri is not None and self.ccorri.shape == (n, nc)

    def to_blob(self):
        return pickle.dumps(self, protocol=5)

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return Fit(**asdict(self))

    def to_df(self):
        is_vector = lambda v: isinstance(v, np.ndarray) and len(v.shape) == 1
        d = self.to_dict()
        d = {k: v for k, v in d.items() if is_vector(v)}
        return pd.DataFrame(d)

    def to_params_dict(self) -> dict:
        assert self.carbs is not None
        assert self.cpeaki is not None
        assert self.ccorri is not None

        params = {name: getattr(self, name) for name in self.GLOBAL_PARAMS}

        for i, start in enumerate(self.carbs.index):
            params[f"cpeak.{start}"] = self.cpeaki[:, i]
            params[f"ccorr.{start}"] = self.ccorri[:, i]
        return params

    def __str__(self):
        args = asdict(self)
        for k, v in args.items():
            if v is None:
                continue
            if isinstance(v, pd.Series):
                args[k] = len(v)
            if isinstance(v, np.ndarray):
                args[k] = round(v.mean(), 3)
            if isinstance(v, datetime):
                args[k] = v.isoformat(timespec="minutes")

        argstr = " ".join(f"{k}={v}" for k, v in args.items())
        return f"Fit({argstr})"

    def __repr__(self):
        return str(self)

    @property
    def icr(self):
        """Insulin-To-Carbs Ratio"""
        return self.isens / self.csens

    def chunks_at(self, tick):
        assert self.chunks is not None
        chunks = self.chunks
        before = chunks.index <= tick
        after = tick < chunks.index + chunks
        chunks = chunks[before & after]
        return chunks

    def carbs_curve(self, key: CurveKey, cfg: Config, default=True) -> Curve:
        """Returns carbs curve. If curve does not exist and default=True,
        returns a curve with default params."""
        assert self.carbs is not None
        assert self.csens is not None
        assert self.cpeak is not None
        assert self.cpeaki is not None
        assert self.ccorri is not None

        start, amount = key
        if key in self.carbs.items():
            i = list(self.carbs.items()).index(key)
            curve = Curve(
                start=start,
                duration=cfg.maxact,
                peak=self.cpeaki[:, i],
                tail=1.0,
                amount=amount,
                corr=self.ccorri[:, i],
                sens=self.csens,
            )
        elif default:
            # no requested curve in the fit, return default
            curve = Curve(
                start=start,
                duration=cfg.maxact,
                peak=self.cpeak,
                tail=1.0,
                amount=amount,
                corr=0.0,
                sens=self.csens,
            )
        else:
            raise KeyError(key)

        # set default values for gscale and nsamples
        curve.at = partial(curve.at, gscale=cfg.gscale, nsamples=cfg.nsamples)
        return curve

    def insulin_curve(self, key: CurveKey, cfg: Config, default=True) -> Curve:
        """Returns insulin curve. If curve does not exist and default=True,
        returns a curve with default params."""
        assert self.insulin is not None

        if key not in self.insulin.items() and not default:
            raise KeyError(key)

        start, amount = key
        curve = Curve(
            start=start,
            duration=cfg.maxact,
            peak=self.ipeak,
            tail=1.0,
            amount=amount,
            corr=0.0,
            sens=self.isens,
        )

        # set default values for gscale and nsamples
        curve.at = partial(curve.at, gscale=cfg.gscale, nsamples=cfg.nsamples)
        return curve

    def predict_car(
        self, ticks: np.ndarray, cfg: Config, override_carbs: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Predicts carbs absorption rate (nsamples, nticks)"""
        assert self.carbs is not None
        carbs = self.carbs if override_carbs is None else override_carbs

        carbs = carbs[carbs >= 0.0]  # allow zero amount carbs to override existing
        ccurves = [self.carbs_curve(key, cfg) for key in carbs.items()]  # type: ignore[arg-type]

        n = cfg.nsamples
        assert self.sigma.shape == (n,)

        car = np.zeros((n, len(ticks)), dtype=float)
        for i, t in enumerate(ticks):
            car[:, i] = sum(c.at(t) for c in ccurves)  # type: ignore[arg-type]

        return car

    def predict_iar(
        self, ticks: np.ndarray, cfg: Config, override_insulin: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Predicts insulin absorption rate (nsamples, nticks)"""
        assert self.insulin is not None
        insulin = self.insulin if override_insulin is None else override_insulin
        insulin = insulin[insulin >= 0.0]  # allow zero amount insulin to override existing
        icurves = [self.insulin_curve(key, cfg) for key in insulin.items()]  # type: ignore[arg-type]

        n = cfg.nsamples
        assert self.sigma.shape == (n,)

        iar = np.zeros((n, len(ticks)), dtype=float)
        for i, t in enumerate(ticks):
            iar[:, i] = sum(c.at(t) for c in icurves)  # type: ignore[arg-type]

        return iar

    def predict_bg(
        self,
        ticks: np.ndarray,
        cfg: Config,
        bg0: Optional[np.ndarray | float] = None,
        override_bolus_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        carbs, insulin = None, None
        if override_bolus_df is not None:
            carbs, insulin = override_bolus_df["carbs"], override_bolus_df["insulin"]
        car = self.predict_car(ticks, cfg, carbs)  # (nticks, nsamples)
        iar = self.predict_iar(ticks, cfg, insulin)  # (nticks, nsamples)

        dar = car - iar  # expected bg delta for the next tick, (nsamples, nticks)
        dar = np.roll(dar, shift=1, axis=1)  # align ticks by shifting right by 1

        if bg0 is None:
            assert self.chunks is not None
            assert self.bg0i is not None
            t0 = ticks[0]
            assert t0 in self.chunks.index
            bg0 = self.bg0i[:, self.chunks.index.get_loc(t0)]
        dar[:, 0] = bg0  # start with bg0

        bg = np.cumsum(dar, axis=1)  # (nsamples, nticks)
        return bg


def clipped_normal(loc, scale, size, a_min=1e-6, a_max=np.inf):
    arr = np.random.normal(loc, scale, size)
    return np.clip(arr, a_min=a_min, a_max=a_max, out=arr)
