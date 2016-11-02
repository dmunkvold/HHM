from neuronlayer import NeuronLayer
 
 
class HHMLinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. """
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr