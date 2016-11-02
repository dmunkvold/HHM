from pybrain.structure.networks.network import Network
import numpy
 
class HelmholtzNetworkComponent(object):
    
    def __init__(self, name=None, **args):
        pass
 
    def activate(self, inpt):
        """Do one transformation of an input and return the result."""
        self.reset()
        return super(HelmholtzNetworkComponent, self).activate(inpt)
        
    def _forwardImplementation(self, inbuf, outbuf):
        assert self.sorted, ".sortModules() has not been called"
        index = 0
        offset = self.offset
        for m in self.inmodules:
            m.inputbuffer[offset] = inbuf[index:index + m.indim]
            index += m.indim
        
        for m in self.modulesSorted:
            m.forward()
            for c in self.connections[m]:
                c.forward()
                
        index = 0
        for m in self.outmodules:
            outbuf[index:index + m.outdim] = m.outputbuffer[offset]
            index += m.outdim

    def _adjustWeights(self):

        #This function adjusts the parameters of generative training

        assert self.sorted, ".sortModules() has not been called"
        
        for m in self.modulesSorted:
            for c in self.connections[m]:
                computedProbs = c.outmod._computeProbabilities()
                for p in range(0, len(c.params)-1):
                    buffers = c.whichBuffers(p)
                    c.params[p] += (c.learningRate*(c.outmod.nodeValues[buffers[1]] - computedProbs[buffers[1]]))*(c.inmod.nodeValues[buffers[0]])



    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        assert self.sorted, ".sortModules() has not been called"
        index = 0
        offset = self.offset
        for m in self.outmodules:
            m.outputerror[offset] = outerr[index:index + m.outdim]
            index += m.outdim
        
        for m in reversed(self.modulesSorted):
            for c in self.connections[m]:
                c.backward()
            m.backward()
                
        index = 0
        for m in self.inmodules:
            inerr[index:index + m.indim] = m.inputerror[offset]
            index += m.indim
            
            
class HelmholtzNetwork(HelmholtzNetworkComponent, Network):
    """FeedForwardNetworks are networks that do not work for sequential data. 
    Every input is treated as independent of any previous or following inputs.
    """
    
    def __init__(self, *args, **kwargs):
        Network.__init__(self, *args, **kwargs)        
        HelmholtzNetworkComponent.__init__(self, *args, **kwargs)