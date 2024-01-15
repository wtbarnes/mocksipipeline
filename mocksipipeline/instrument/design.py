"""
Class for holding full instrument design
"""
import dataclasses

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
    name: str
    channel_list: list[Channel]

    def __add__(self, instrument):
        if not isinstance(instrument, InstrumentDesign):
            raise TypeError(f'Addition is not supported with types {type(instrument)}')
        combined_name = f'{self.name}+{instrument.name}'
        return InstrumentDesign(combined_name, self.channel_list+instrument.channel_list)

    @property
    def optical_design(self):
        # All channels will have the same optical design
        return self.channel_list[0].design

    def __repr__(self):
        channel_reprs = '\n'.join(str(c) for c in self.channel_list)
        return f"""MOXSI Instrument Configuration {self.name}
                   ------------------------------------------
{self.optical_design}

{channel_reprs}
"""
