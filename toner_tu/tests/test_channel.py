
import numpy as np

from toner_tu.fields import Grid
from toner_tu.channel import Channel


def test_channel_shape():
    grid = Grid(nx=128, ny=48, lx=40.0, ly=12.0)
    channel = Channel(grid)

    assert channel.shape == (48, 128)


def test_channel_coordinates():
    grid = Grid(nx=128, ny=48, lx=40.0, ly=12.0)
    channel = Channel(grid)

    x, y = channel.coordinates()

    assert x.shape == (48, 128)
    assert y.shape == (48, 128)

    assert np.isclose(x[0, -1], 39.6875)
    assert np.isclose(y[-1, 0], 11.75)


def test_inside():
    channel = Channel(Grid(nx=128, ny=48, lx=40.0, ly=12.0))

    assert channel.is_inside(10.0, 5.0)
    assert channel.is_inside(0.0, 0.0)
    assert not channel.is_inside(-1.0, 5.0)
    assert not channel.is_inside(10.0, 13.0)
