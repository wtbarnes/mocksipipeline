"""
Test channel object
"""
import astropy.units as u
import numpy as np
import pytest

from mocksipipeline.instrument.configuration import moxsi_short
from mocksipipeline.instrument.optics.response import Channel

ALL_CHANNELS = moxsi_short.channel_list

@pytest.fixture
def filters():
    return moxsi_short.channel_list[0].filters


@pytest.fixture
def design():
    return moxsi_short.channel_list[0].design


@pytest.fixture
def aperture():
    return moxsi_short.channel_list[0].aperture


@pytest.mark.parametrize(
        ('name', 'order'),
        [(f'filtergram_{i}', 0) for i in range(1, 5)]+[('spectrogram_1', i) for i in range(-4, 5, 1)])
def test_channel_creation(name, order, filters, design, aperture):
    chan = Channel(name, filters, order, design, aperture, (0,0))
    assert isinstance(chan, Channel)


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_channel(channel):
    # Use repr as simple smoke test
    assert isinstance(str(channel), str)


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_effective_area(channel):
    assert isinstance(channel.effective_area, u.Quantity)
    assert channel.effective_area.shape == channel.wavelength.shape
    assert channel.effective_area.unit.is_equivalent('cm2')


@pytest.mark.parametrize('channel', ALL_CHANNELS)
def test_wavelength_response(channel):
    assert isinstance(channel.wavelength_response, u.Quantity)
    assert channel.wavelength_response.shape == channel.wavelength.shape
    assert channel.wavelength_response.unit.is_equivalent('cm2 ct / ph')


@pytest.mark.parametrize('channel', moxsi_short.channel_list[:4])
def test_filtergrams_grating_efficiency_one(channel):
    assert np.all(channel.grating_efficiency == 1)
