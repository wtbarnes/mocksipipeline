"""
Top level pipeline
"""
class MockSiPipeline:

    def __init__(self, physics, detector, operations):
        self.physics = physics
        self.detector = detector
        self.operations = operations

    def run(self):
        spec_cube = self.physics.run()
        det_image = self.detector.run(spec_cube)
        moxsi_image = self.operations.run(det_image)
        return moxsi_image