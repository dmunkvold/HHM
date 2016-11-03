from HHMNeuronLayer import HHMNeuronLayer
import numpy
from pybrain.tools.functions import sigmoid
 
 
class HHMLinearLayer(HHMNeuronLayer):
    """ The simplest kind of module, not doing any transformation. """
    
    def _forwardImplementation(self, inbuf, outbuf):
        self.nodeValues = inbuf
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
        
    def _computeProbabilities(self):
            outbuf = numpy.zeros(len(self.inputbuffer[0]))
            outbuf[:] = sigmoid(self.inputbuffer)
            return outbuf