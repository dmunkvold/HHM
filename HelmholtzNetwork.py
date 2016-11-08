from pybrain.structure.networks.network import Network
from HHMBiasUnit import HHMBiasUnit
import numpy
import random
 
class HelmholtzNetworkComponent(object):
    
    def __init__(self, name=None, **args):
        pass
 
    def activate(self, inpt):
        """Do one transformation of an input and return the result."""
        self.reset()
        return super(HelmholtzNetworkComponent, self).activate(inpt)
        
    def _forwardImplementation(self, inbuf, outbuf):
        #print "forward"
        assert self.sorted, ".sortModules() has not been called"
        index = 0
        offset = self.offset
        for m in self.inmodules:
            m.inputbuffer[offset] = inbuf[index:index + m.indim]
            index += m.indim
        
        for m in self.modulesSorted:
            
            #print m
            m.forward()
            #print "forward inbuf",m.inputbuffer
            #print m.nodeValues
            for c in self.connections[m]:
                #print c
                c.forward()
                #print c.params
                
        index = 0
        for m in self.outmodules:
            outbuf[index:index + m.outdim] = m.outputbuffer[offset]
            index += m.outdim

    def _adjustWeights(self):
        #print "adjusting"
        #This function adjusts the parameters of generative training
        #testing
        assert self.sorted, ".sortModules() has not been called"

        index = 0
        offset = self.offset
        for m in self.inmodules:
            if isinstance(m, HHMBiasUnit):
                m.nodeValues = [1]            
            m.inputbuffer[offset] = m.nodeValues
            
        
        for m in self.modulesSorted:
            
            #print m
            
            if isinstance(m, HHMBiasUnit):
                m.nodeValues = [1]
            #print m.inputbuffer
            
            for c in self.connections[m]:
                inputbuff = c._updateInputBuffer()
                #print "adjusting outmod inbuf", inputbuff
                computedProbs = c.outmod._computeProbabilities(inputbuff)
                for p in range(0, len(c.params)):
                    buffers = c.whichBuffers(p)
                    #print c
                    #print "buffers", buffers
                    #print "outmod node values", c.outmod.nodeValues
                    #print "computedprobs", computedProbs
                    #print "inmod nodevalues", c.inmod.nodeValues, c.inmod
                    #print "before",c.params
                    #print (c.learningRate*(c.outmod.nodeValues[buffers[1]] - computedProbs[buffers[1]]))*(c.inmod.nodeValues[buffers[0]])
                    c.params[p] += (c.learningRate*(c.outmod.nodeValues[buffers[1]] - computedProbs[buffers[1]]))*(c.inmod.nodeValues[buffers[0]])
                    
                    #print "after:", c.params
                   



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