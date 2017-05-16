import numpy

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.decorators import WavefrontDecorator

class WiseWavefront(WavefrontDecorator):

    def __init__(self,
                 wavelength=1e-10,
                 positions=numpy.zeros(100),
                 electric_fields=numpy.zeros(100),
                 residuals=numpy.zeros(0)):
        self._wavelength = wavelength
        self._positions = positions
        self._electric_fields = electric_fields
        self._residuals = residuals


    def toGenericWavefront(self):
        return GenericWavefront1D.initialize_wavefront_from_arrays(x_array=self._positions, y_array=self._electric_fields, wavelength=self._wavelength)

    @classmethod
    def fromGenericWavefront(cls, wavefront):
        wavelength = wavefront.get_wavelength()
        position = wavefront.get_abscissas()
        electric_fields = wavefront.get_complex_amplitude()

        return WiseWavefront(wavelength=wavelength,
                             position=position,
                             electric_fields=electric_fields)