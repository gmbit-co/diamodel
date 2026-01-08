from pathlib import Path
import time
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
from typing import Any

from .fit import Fit
from .window import window_overlaps
from .config import Config
from .utils import Logger

__all__ = ["DiaModel"]

log = Logger(__name__)


class DiaModel:

    GLOBAL_PARAMS = ["isens", "ipeak", "sigma", "rho", "csens", "cpeak"]

    def __init__(self, cfg: Config):
        # check params order
        assert all(p1 == p2 for p1, p2 in zip(self.GLOBAL_PARAMS, Fit.GLOBAL_PARAMS))
        self.cfg = cfg
        self.stan_file = Path(__file__).parent / "stan/model_full.stan"

    def select_chunks(
        self, data_df: pd.DataFrame, between_hours: tuple[int, int] | None = None
    ) -> pd.Series:
        """Selects chunks (prediction intervals) for training."""
        cfg = self.cfg
        bolus = data_df[(data_df["carbs"] > 0) | (data_df["insulin"] > 0)]

        if len(data_df) == 0 or len(bolus) == 0:
            return pd.Series([])

        # select chunks where we have active carbs and insulin curves
        last = data_df.iloc[-1].tick  # last tick
        activity = np.zeros((last,), dtype=bool)  # bolus activity mask
        for start in bolus["tick"]:
            activity[start : start + cfg.maxact] = True

        # pad to catch True at the beginning and end
        activity = np.concatenate(([False], activity, [False]))
        ranges = np.diff(activity).nonzero()[0]
        ranges = ranges.reshape(-1, 2)
        anchors = [np.arange(ta, tb, cfg.maxpred) for ta, tb in ranges]
        sizes = [end - aa for aa, (_, end) in zip(anchors, ranges)]
        sizes = [np.minimum(ss, cfg.maxpred) for ss in sizes]

        anchors = [v for sub in anchors for v in sub]
        sizes = [v for sub in sizes for v in sub]
        chunks = pd.Series(sizes, index=anchors)

        cgm = data_df["cgm"]
        if between_hours is not None:
            start_hour, end_hour = between_hours
            hour = data_df["timestamp"].dt.hour  # type: ignore[arg-type]
            mask = (hour >= start_hour) & (hour < end_hour)
            cgm = cgm.copy()
            cgm.loc[~mask] = float("nan")

        # truncate chunks at first NaN in cgm
        new_sizes = []
        for a, s in chunks.items():
            chunk_cgm = cgm.loc[a : a + s]
            if chunk_cgm.hasnans:
                new_sizes.append(chunk_cgm.isna().argmax() - 1)
            else:
                new_sizes.append(s)
        chunks = pd.Series(new_sizes, index=chunks.index)

        # filter out small chunks
        chunks = chunks[chunks > 12 // cfg.dt]

        # check all chunks have bolus
        assert all(not bolus.loc[a - cfg.maxact + 1 : a + s - 1].empty for a, s in chunks.items())  # type: ignore[arg-type]
        return chunks

    def stan_data(self, data_df: pd.DataFrame, chunks: pd.Series, prior_fit: Fit):
        """Prepares stan data dictionary"""
        cfg = self.cfg

        carbs = data_df["carbs"][data_df["carbs"] > 0]
        insulin = data_df["insulin"][data_df["insulin"] > 0]

        xg = np.arange(1, cfg.maxact + 1) * cfg.gscale  # gamma domain grid

        # insulin curves
        ixg = [window_overlaps(ta, cfg.maxpred, xg, insulin.index) for ta in chunks.index]
        iid = [x.sum(axis=1).nonzero()[0] for x in ixg]
        islice = [(ii[0] if len(ii) > 0 else 0, len(ii)) for ii in iid]  # [(pos, num),...]
        # check that all slices are sequential
        assert all(sl[0] + sl[1] - 1 == ii[-1] for sl, ii in zip(islice, iid) if len(ii) > 0)
        ixg = [x[ii] for ii, x in zip(iid, ixg)]

        # carb curves
        cxg = [window_overlaps(ta, cfg.maxpred, xg, carbs.index) for ta in chunks.index]
        cid = [x.sum(axis=1).nonzero()[0] for x in cxg]
        cslice = [(ci[0] if len(ci) > 0 else 0, len(ci)) for ci in cid]  # [(pos, num),...]
        # check that all slices are sequential
        assert all(sl[0] + sl[1] - 1 == ci[-1] for sl, ci in zip(cslice, cid) if len(ci) > 0)
        cxg = [x[ii] for ii, x in zip(cid, cxg)]

        # max number of curves per each chunk
        islice = np.array(islice)
        cslice = np.array(cslice)
        maxc = max(max(islice[:, 1], default=0), max(cslice[:, 1], default=0))

        # convert to fixed size arrays
        ixg = [to_fixed_np(xs, maxc) for xs in ixg]
        ixg = np.array(ixg)
        cxg = [to_fixed_np(xs, maxc) for xs in cxg]
        cxg = np.array(cxg)

        assert all(islice[:, 1] + cslice[:, 1] > 0)  # all chunks must have at least one curve

        cgms = [data_df["cgm"].loc[a : a + s] for a, s in chunks.items()]  # (na, chunksize + 1)
        assert all(not cgm.hasnans for cgm in cgms)
        cgms = np.array([to_fixed_np(cgm, cfg.maxpred + 1) for cgm in cgms])  # (na, maxpred)
        assert cgms.shape == (len(chunks), cfg.maxpred + 1)

        log.debug(
            "Stan data",
            chunks=len(chunks),
            cgm=cgms.size,
            carbs=len(carbs),
            insulin=len(insulin),
            maxc=maxc,
        )

        # pack priors into 2d arrays
        nz = len(self.GLOBAL_PARAMS)  # number of parameters
        x = [getattr(prior_fit, key) for key in self.GLOBAL_PARAMS]
        x = np.array(x, dtype=float)  # (nz, nsamples)
        assert x.shape[0] == nz

        z = x2z(x)
        mu_z = z.mean(axis=1)
        cov = np.cov(z)
        cov = np.diag(np.diag(cov))  # make priors independent

        L_z = np.linalg.cholesky(cov)
        assert mu_z.shape[0] == nz
        assert L_z.shape == (nz, nz)

        islice[:, 0] += 1  # 1-based indexing
        cslice[:, 0] += 1

        data = {
            "maxpred": cfg.maxpred,
            "maxc": maxc,
            "gscale": cfg.gscale,
            # chunks
            "na": len(chunks),
            "chunk": chunks,
            # insulin
            "ni": len(insulin),
            "insulin": insulin,
            "islice": islice,
            "ixg": ixg,
            # carbs
            "nc": len(carbs),
            "carbs": carbs,
            "cslice": cslice,
            "cxg": cxg,
            # cgm
            "cgm": cgms,
            # priors
            "nz": nz,
            "mu_z": mu_z,
            "L_z": L_z,
            # extra data
            "xg": xg,
        }

        return data

    def create_fit(
        self, chunks: pd.Series, carbs: pd.Series, insulin: pd.Series, samples_df: pd.DataFrame
    ) -> Fit:
        """Creates Fit from MCMC samples"""
        fit: dict[str, Any] = {"carbs": carbs, "insulin": insulin, "chunks": chunks}

        assert self.GLOBAL_PARAMS == Fit.GLOBAL_PARAMS
        for i, name in enumerate(self.GLOBAL_PARAMS):
            fit[name] = samples_df[f"x[{i+1}]"].to_numpy()

        # local params
        fit["cpeaki"] = samples_df.filter(like="cpeaki[").to_numpy()
        fit["ccorri"] = samples_df.filter(like="ccorri[").to_numpy()
        fit["bg0i"] = samples_df.filter(like="bg0i[").to_numpy()
        fit["lp__"] = samples_df["lp__"].to_numpy()

        return Fit(**fit)

    def fit(
        self,
        data_df: pd.DataFrame,
        prior_fit: Fit,
        chunks=None,
        num_chains: int = 4,
        num_samples: int = 2000,
    ) -> Fit:
        t0 = time.time()

        chunks = self.select_chunks(data_df) if chunks is None else chunks
        if len(chunks) == 0:
            log.warning("No chunks selected")

        sdata = self.stan_data(data_df, chunks, prior_fit)

        # Compile model
        model = CmdStanModel(stan_file=self.stan_file)
        # Perform MCMC
        sfit = model.sample(
            data=sdata, chains=num_chains, iter_sampling=num_samples, seed=42, show_progress=False
        )
        samples_df = sfit.draws_pd()
        divergent_mask = samples_df["divergent__"] > 0
        samples_df = samples_df[~divergent_mask]  # drop diverged samples
        ndiverged = divergent_mask.sum()

        # # Perform Variational inference
        # sfit = model.variational(
        #     data=sdata,
        #     seed=42,
        #     algorithm="fullrank",
        #     iter=num_samples,
        #     tol_rel_obj=0.02,
        #     draws=self.cfg.nsamples,
        #     # show_console=True,
        # )
        # ndiverged = 0.0
        # samples_df = sfit.variational_sample_pd

        # keep random subset of samples
        if self.cfg.nsamples < num_samples:
            samples_df = samples_df.sample(n=self.cfg.nsamples, random_state=42)

        fit = self.create_fit(sdata["chunk"], sdata["carbs"], sdata["insulin"], samples_df)

        elapsed = time.time() - t0
        log.info("Fit model", elapsed=round(elapsed, 1), ndiverged=ndiverged)

        return fit


def x2z(x):
    """Converts positively constrained x to unconstrained z."""
    return np.log(x)


def z2x(z):
    """Converts unconstrained z to positively constrained x."""
    return np.exp(z)


def to_fixed_np(arr, n):
    arr = np.array(arr)
    assert n >= arr.shape[0]
    shape = (n,) + arr.shape[1:]
    fixed = np.zeros(shape, dtype=arr.dtype)
    fixed[: len(arr)] = arr
    return fixed
