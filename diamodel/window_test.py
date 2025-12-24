import numpy as np
from . import window as ww


def test_window_overlap():
    x = np.arange(1, 9)

    # -123456789--------  global ticks
    # -12345678---------  array x at tx = 1
    # ------______------  window at tw = 6
    # ------678000------  result xs = [678000], nw = 6
    assert np.allclose(ww.window_overlap(tw=6, nw=6, x=x, tx=1), np.array([6, 7, 8, 0, 0, 0]))

    # -123456789--------  global ticks
    # ------12345678----  array x at tx = 6
    # --______----------  window at tw = 2
    # --000012----------  result xs = [000012], nw = 6
    assert np.allclose(ww.window_overlap(tw=2, nw=6, x=x, tx=6), np.array([0, 0, 0, 0, 1, 2]))

    # -123456789--------  global ticks
    # --12345678--------  array x at tx = 2
    # ---_____----------  window at tw = 3
    # ---23456----------  result xs = [23456], nw = 5
    assert np.allclose(ww.window_overlap(tw=3, nw=5, x=x, tx=2), np.array([2, 3, 4, 5, 6]))


def test_window_overlaps():
    x = np.full((8,), True)
    xss = ww.window_overlaps(tw=2, nw=6, x=x, txs=[-1, 6, 8])  # only txs=[-1, 6] overlap
    assert np.allclose(xss.sum(axis=1), [5, 2, 0])

    xss = ww.window_overlaps(tw=2, nw=6, x=x, txs=[])  # no overlaps
    assert np.allclose(xss.sum(axis=1), [])
