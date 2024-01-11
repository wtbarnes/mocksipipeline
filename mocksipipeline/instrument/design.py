"""
Class for holding full instrument design
"""
import dataclasses

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
    channel_list: list[Channel]
