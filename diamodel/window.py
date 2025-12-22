from typing import Iterable
import numpy as np

__all__ = ["window_overlap", "window_overlaps"]


def window_overlap(tw: int, nw: int, x: np.ndarray, tx: int, zero=0.0) -> np.ndarray:
    """Given a window of size `nw` positioned at `tw` returns overlap
    with `x` positioned at `tx`.

    Example 1:
        -123456789--------  global ticks
        -xxxxxxxx---------  array x at tx = 1
        ------______------  window at tw = 6
        ------xxx000------  result xs = [xxx000], nw = 6

    Example 2:
        -123456789--------  global ticks
        ------xxxxxxxx----- array x at tx = 6
        --______----------  window at tw = 2
        --0000xx----------  result xs = [0000xx], nw = 6
    """

    xs = np.full((nw,), zero, dtype=x.dtype)
    nx = len(x)
    if nx == 0:
        return xs

    if tx <= tw:
        # example 1
        ix = min(tw - tx, nx)
        iw = 0
        n = min(nx - ix, nw)
    else:
        # example 2
        ix = 0
        iw = min(tx - tw, nw)
        n = min(nw - iw, nx)
    xs[iw : iw + n] = x[ix : ix + n]
    return xs


def window_overlaps(tw: int, nw: int, x: np.ndarray, txs: np.ndarray | Iterable) -> np.ndarray:
    """Returns all overlaps of window with `x` positioned at `txs`."""
    xss = [window_overlap(tw=tw, nw=nw, x=x, tx=tx) for tx in txs]
    if len(xss) > 0:
        return np.array(xss)
    else:
        return np.zeros((0, nw), dtype=x.dtype)
