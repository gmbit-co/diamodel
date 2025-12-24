from pathlib import Path
import time
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel

from .fit import Fit
from .window import window_overlaps
from .config import Config
from .utils import Logger

__all__ = ["InsulinModel"]

log = Logger(__name__)


class InsulinModel:

    GLOBAL_PARAMS = ["isens", "ipeak", "sigma", "rho"]

    def __init__(self, cfg: Config):
        # check params order
        assert all(p1 == p2 for p1, p2 in zip(self.GLOBAL_PARAMS, Fit.GLOBAL_PARAMS))
        self.cfg = cfg
        self.stan_file = Path(__file__).parent / "stan/model_insulin.stan"

    def select_chunks(self, data_df: pd.DataFrame) -> pd.Series:
        """Selects chunks (prediction intervals) for training."""
        cfg = self.cfg
        insulin = data_df[data_df["insulin"] > 0]

        # keep only morning insulin. Other times we might have missing carbs data
        mask = insulin["timestamp"].dt.hour < 9  # type: ignore[arg-type]
        iticks = insulin[mask]["tick"]

        if iticks.empty:
            return pd.Series([])

        cticks = data_df[data_df["carbs"] > 0]["tick"].copy()
        # set virtual "boundary" carbs so all insulin ticks have matching pre- and post-carbs
        ticka, tickb = data_df.iloc[0].tick, data_df.iloc[-1].tick
        cticks.loc[ticka] = ticka
        cticks.loc[tickb] = tickb
        cticks.sort_values(inplace=True)

        def pre_carb(tick):
            """Returns closest carbs tick before given tick"""
            pre_ticks = cticks.loc[:tick]  # inclusive of `tick`
            assert not pre_ticks.empty
            return pre_ticks.iloc[-1]

        def post_carb(tick):
            """Returns closest carb tick after given tick"""
            post_ticks = cticks.loc[tick:]
            assert not post_ticks.empty
            return post_ticks.iloc[0]

        pre_cticks = iticks.map(pre_carb).values  # closest carbs before insulin
        deltas = iticks - pre_cticks  # ticks between closest pre-carb and insulin
        # only keep insulin without carbs
        mask = deltas > cfg.maxact
        pre_cticks = pre_cticks[mask]
        iticks = iticks[mask]
        deltas = deltas[mask]

        # when multiple insulin exist, select the closest one to given pre-carb
        deltas = deltas.groupby(pre_cticks).min()  # (cticks, deltas)
        anchors = deltas.index + deltas  # iticks = cticks + deltas
        post_cticks = anchors.map(post_carb).values
        sizes = post_cticks - anchors
        sizes = np.array([min(cs, cfg.maxpred) for cs in sizes])
        sizes -= 1  # so we always have tick+1 cgm value

        # select large enough chunks
        mask = sizes > 30 // cfg.dt  # > 30mins
        sizes = sizes[mask]
        anchors = anchors[mask]
        # filter out chunks with cgm nan's
        cgm = data_df["cgm"]
        mask = [not cgm.loc[a : a + s].hasnans for a, s in zip(anchors, sizes)]
        sizes = sizes[mask]
        anchors = anchors[mask]

        data_df = data_df.loc[anchors]
        assert np.all(data_df["insulin"] > 0)
        assert np.all(data_df["carbs"] == 0.0)

        return pd.Series(sizes, index=anchors)

    def stan_data(self, data_df: pd.DataFrame, chunks: pd.Series, prior_fit: Fit):
        """Prepares stan data dictionary"""
        cfg = self.cfg
        t0 = data_df.iloc[0].tick
        # select relevants subsets of data_df
        data_df2 = [data_df.loc[max(a - cfg.maxact, t0) : a + s - 1] for a, s in chunks.items()]  # type: ignore[arg-type]
        data_df2 = pd.concat(data_df2)
        assert not any(data_df2["carbs"] > 0)  # expect no carbs

        insulin = data_df2["insulin"][data_df2["insulin"] > 0]
        insulin = insulin[~insulin.index.duplicated(keep="first")]  # deduplicate

        na = len(chunks)
        ni = len(insulin)

        xg = np.arange(1, cfg.maxact + 1) * cfg.gscale  # gamma domain grid

        # ixg = (na, ni, maxpred)
        ixg = [window_overlaps(ta, cfg.maxpred, xg, insulin.index) for ta in chunks.index]
        iid = [xgs.sum(axis=1).nonzero()[0] for xgs in ixg]
        assert all(len(ii) > 0 for ii in iid)  # all chunks must have insulin
        islice = [(ii[0], len(ii)) for ii in iid]  # [(pos, num),...]
        islice = np.array(islice)
        # check that all slices are sequential
        assert all(sl[0] + sl[1] - 1 == ii[-1] for sl, ii in zip(islice, iid))

        maxc = max(islice[:, 1], default=0)
        ixg = [xg[ii] for ii, xg in zip(iid, ixg)]
        ixg = [to_fixed_np(xs, maxc) for xs in ixg]
        ixg = np.array(ixg)

        cgms = [data_df["cgm"].loc[a : a + s] for a, s in chunks.items()]  # (na, chunksize + 1)
        assert all(not cgm.hasnans for cgm in cgms)
        cgms = np.array([to_fixed_np(cgm, cfg.maxpred + 1) for cgm in cgms])  # (na, maxpred)
        assert cgms.shape == (na, cfg.maxpred + 1)

        log.debug(
            "Stan data",
            chunks=len(chunks),
            cgm=cgms.size,
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
        nz = len(mu_z)
        assert L_z.shape == (nz, nz)

        islice[:, 0] += 1  # 1-based indexing

        data = {
            "maxpred": cfg.maxpred,
            "maxc": maxc,
            "gscale": cfg.gscale,
            # chunks
            "na": na,
            "chunk": chunks,
            # insulin
            "ni": ni,
            "insulin": insulin,
            "islice": islice,
            "ixg": ixg,
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

    def create_fit(self, chunks: pd.Series, insulin: pd.Series, samples_df: pd.DataFrame) -> Fit:
        """Creates Fit from MCMC samples"""
        fit = {"carbs": None, "insulin": insulin, "chunks": chunks}

        for i, name in enumerate(self.GLOBAL_PARAMS):
            fit[name] = samples_df[f"x[{i+1}]"].to_numpy()

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

        # keep random subset of samples
        if self.cfg.nsamples < num_samples:
            samples_df = samples_df.sample(n=self.cfg.nsamples, random_state=42)

        fit = self.create_fit(sdata["chunk"], sdata["insulin"], samples_df)
        # copy carbs priors that this model does not estimate
        fit.csens = prior_fit.csens
        fit.cpeak = prior_fit.cpeak

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
