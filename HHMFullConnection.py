from scipy import reshape, dot, outer

from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer
import numpy


class HHMFullConnection(Connection, ParameterContainer):
    """Connection which fully connects every element from the first module's
    output buffer to the second module's input buffer in a matrix multiplicative
    manner."""

    def __init__(self, learningRate, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
        ParameterContainer.__init__(self, self.indim * self.outdim)
        self.learningRate = learningRate

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += dot(reshape(self.params, (self.outdim, self.indim)), inbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += dot(reshape(self.params, (self.outdim, self.indim)).T, outerr)
        ds = self.derivs
        ds += outer(inbuf, outerr).T.flatten()
        
        
    def _updateInputBuffer(self):
        outbuf = dot(reshape(self.params, (self.outdim, self.indim)), self.inmod.nodeValues)
        return numpy.array([outbuf])
        
    def whichBuffers(self, paramIndex):
        """Return the index of the input module's output buffer and
        the output module's input buffer for the given weight."""
        return paramIndex % self.inmod.outdim, paramIndex / self.inmod.outdim