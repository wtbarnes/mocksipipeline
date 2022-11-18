# mocksipipeline

Pipeline for simulating data products from the Multi-Order X-ray Spectral Imager (MOXSI) onboard
the CubIXSS cubesat.

A quick mock-up of what this could look like:

* `MockSiPipeline`
  * `PhysicsPipeline`
    * `DEMModule`
    * `SpectralModule`
  * `DetectorPipeline`
    * `WaveResponseModule`
    * `ProjectionModule`
  * `OperationsPipeline`
    * `StrayLightModule`
    * `ThresholdingModule`
    * `...`

## Top-Level Pipeline

The top-level simulation pipeline is responsible for coordinating each piece.
The pipeline ingests three instances of the individual component pipelines and returns a detector image (or images).

```python
class MockSiPipeline:

    def __init__(self, physics, detector, operations):
        self.physics = physics
        self.detector = detector
        self.operations = operations

    def run(self):
        spec_cube = self.physics.run()
        detector_image = self.detector.run(spec_cube)
        real_detector_image = self.operations.run(detector_image)
        return real_detector_image
```

## Component Pipelines

Each component pipeline is responsible for coordinating each of the modules.
Each component pipeline ingests an instance for each module that comprises the pipeline.

### Physics

```python
class PhysicsPipeline:

    def __init__(self, dem, spectra):
        self.dem = dem
        self. spectra = spectra

    def run(self):
        dem_cube = self.dem.run()
        spec_cube = self.spectra.run(dem_cube)
       return spec_cube
```

#### DEM Module

The only requirement of the DEM module is that it produces a DEM cube.
Very generally, this would look like the following:

```python
class DemModule:

    def __init__(self, *args, **kwargs):
        ...

    def run(self):
        # Do some computation to return a  DEM cube of dimensions (nT, nX, nY)
        return dem_cube
```

In the case where we compute an inverted DEM from AIA and XRT, the subclass would look like the following

```python
class InvertedDem(DemModule):

    def __init__(self, date, dem_model_class):
        self.date = date
        self.dem_model_class = dem_model_class

    def fetch_data(self):
        # Use Fido to query the data
        return list_of_maps

    def build_collection(maps):
        # Reproject all maps to same coordinate frame
        # Place in a collection 
        return map_collection

    def get_responses(map_collection):
        # For each key in the map, get a wavelength response
        # and put it in a dictionary
        return response_dict

    def compute_dem(map_collection, response_dict):
        dem_model = self.dem_model_class(map_collection, response_dict)
        return dem_model.fit()

    def run(self):
        maps = self.fetch_data()
        map_collection = self.build_collection(maps)
        responses = self.get_responses(map_collection)
        dem_cube = self.compute_dem(map_collection, responses)
        return dem_cube
```

#### Spectral Module

### Detector

```python
class DetectorPipeline:
    ...
```

Whether the detector is for the pinhole or the dispersed image should depend on what is substituted for the wavelength response and projection modules.

#### Wavelength Response Module

This module will convert spectral cube from physical units (e.g. photon or erg) to detector units

#### Projection Module

Reproject the spectral cube into the detector frame

### Operations

```python
class OperationsPipeline:
    ...
```

## Notes

* The component pipelines of the overall pipeline may change.
* Similarly, the number or kind of modules in each component pipeline may change.
* However, we should define the interfaces between each component to change little such that components can be easily swapped out, added, or removed without altering the flow of the whole pipeline.
  * As an example, we're fairly certain our `PhysicsPipeline` will always produce a spectral cube, but they way in which we produce that may change
  * e.g. maybe instead of starting with a DEM, we'll run a model and project the volume emissivity along a selected line of sight.
  * In this case, `DEMModule` and `SpectralModule` would be replaced by a more complicated physical model, but the interface with the other components remains the same.
* Should consider implementing metaclasses for each component and module to specify what methods are required on the subclasses for each of these.