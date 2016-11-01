import pybrain
import numpy
from pybrain.structure.modules.neuronlayer import NeuronLayer
from HHMNeuronLayer import HHMNeuronLayer
from pybrain.tools.functions import sigmoid
 
#currently testing. should be hhmneruonlayer


class HHMSigmoidLayer(HHMNeuronLayer):
    """Layer specifically for Helmholtz Machines implementing the sigmoid squashing function."""
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)
        nodeValues = numpy.zeros(self.dim)
        for i in range(0, len(outbuf)-1):
            neuronActivated = numpy.random.choice(numpy.arange(0, 2), p=[1 - outbuf[i], outbuf[i]])
            nodeValues[i] = neuronActivated
            outbuf[i] = neuronActivated*outbuf[i]
        self.nodeValues = nodeValues
        return outbuf
    
    def _computeProbabilities(self):
        outbuf = numpy.zeros(len(self.inputbuffer))
        outbuf[:] = sigmoid(self.inputbuffer)
        return outbuf

    