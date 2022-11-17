"""
Module for modeling physics portion of the pipeline
"""


__all__ = ['PhysicsModel']


class PhysicsModel:

    def __init__(self, dem_model, spectral_model):
        self.dem_model = dem_model
        self.spectral_model = spectral_model

    def run(self):
        dem_cube = self.dem_model.run()
        spec_cube = self.spectral_model.run(dem_cube)
        return spec_cube
