"""
Test channel object
"""
import astropy.units as u
import numpy as np
import pytest

from mocksipipeline.detector.response import (Channel,
                                              get_all_dispersed_channels,
                                              get_all_filtergram_channels)

ALL_CHANNELS = get_all_dispersed_channels() + get_all_filtergram_channels()


@pytest.mark.parametrize(
        ('name', 'order'),
        [(f'filtergram_{i}', 0) for i in range(1, 5)]+[('spectrogram_1', i) for i in range(-4, 5, 1)])
def test_channel_creation(name, order):
    chan = Channel(name, order=order)
    assert isinstance(chan, Channel)


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_channel(channel):
    # Use repr as simple smoke test
    assert isinstance(str(channel), str)


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_effective_area(channel):
    assert isinstance(channel.effective_area, u.Quantity)
    assert channel.effective_area.to('cm2')


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_wavelength_response(channel):
    assert isinstance(channel.wavelength_response, u.Quantity)
    assert channel.wavelength_response.to('cm2 ct / ph')


@pytest.mark.parametrize('channel', get_all_filtergram_channels())
def test_filtergrams_grating_efficiency_one(channel):
    assert np.all(channel.grating_efficiency == 1)
