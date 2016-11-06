from HHMNeuronLayer import HHMNeuronLayer
from pybrain.structure.modules.module import Module
 
 
class HHMBiasUnit(HHMNeuronLayer):
    """A simple bias unit with a single constant output."""
        
        
    def _forwardImplementation(self, inbuf, outbuf):
        self.nodeValues = [1]
        outbuf[:] = 1    