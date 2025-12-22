import pytest
from functools import partial
import numpy as np
from scipy.stats import gengamma

from .config import Config
from .curve import Curve
from . import window as ww


@pytest.fixture
def cfg():
    return Config.default_config()


@pytest.mark.parametrize("start", [-500, 10, 15, 20])
@pytest.mark.parametrize("peak", [2.0, 3.0, 4.0])
def test_curve(start, peak, cfg):
    amount = 10
    tail = 1.1
    crv = Curve(
        start=start,
        duration=cfg.maxact,
        peak=peak,
        tail=tail,
        amount=amount,
        corr=0.0,
        sens=1.0,
    )

    crv.at = partial(crv.at, gscale=cfg.gscale, nsamples=1)

    tw = 10  # window start position
    nw = 20  # window size
    xg = np.arange(1, cfg.maxact + 1) * cfg.gscale
    xg = ww.window_overlap(tw=tw, nw=nw, x=xg, tx=start)

    y0 = amount * gengamma.pdf(xg, peak, tail, scale=1)
    y0 *= cfg.gscale  # gamma.sum() ~= 1

    ticks = list(range(tw, tw + nw))
    y = np.array([crv.at(t) for t in ticks])

    assert np.allclose(y, y0)


def test_gscale(cfg):
    amount = 5
    crv = Curve(
        start=0,
        duration=cfg.maxact,
        peak=3.0,
        tail=1.0,
        amount=amount,
        corr=0.0,
        sens=1.0,
    )

    crv.at = partial(crv.at, gscale=cfg.gscale, nsamples=1)
    ticks = np.arange(0, cfg.maxact)
    y = np.array([crv.at(t) for t in ticks])

    assert np.allclose(y.sum(), amount, rtol=1e-3)
