import pybrain
import scipy
import HHMSigmoidLayer
from scipy import random, array, empty
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit, SoftmaxLayer
from pybrain import *

class HelmHoltzMachine():
    
    #the helmholtz machine class is a container for the recognition and generation modules
    
    def __init__(self, indim, Layers, LayerDims):
        
        
        self.recNet = FeedForwardNetwork()
        self.genNet = FeedForwardNetwork()
        self.indim = indim
        self.layerDims = LayerDims
        self.layers = numpy.zeros(Layers)
        self.biasUnits = numpy.zeros(Layers + 1)
        
        #declaring and adding input layer
        layers[0] = HHMSigmoidLayer(indim, "HHM Sigmoid Input/Output Layer")
        self.recNet.addInputModule(self.layers[0])
        self.genNet.addOutputModule(self.layers[0])
        
        #declaring and adding all other layers and connections (full for now)
        for i in range(1, len(self.layers)):
            self.layers[i] = HHMSigmoidLayer(self.layerDims[i], "HHM Sigmoid Layer " + str(i))
            self.recNet.addModule(self.layers[i])
            self.genNet.addModule(self.layers[i])
            self.recNet.addConnection(FullConnection(self.layers[j-1], self.layers[j]))
            self.genNet.addConnection(FullConnection(self.layers[j], self.layers[j-1]))            
        
        #declaring and adding bias units and connections
        for j in range(0, len(self.biasUnits)):
            self.biasUnits[j] = BiasUnit("BiasUnit " + str(j))
            if j==0:
                self.recNet.addModule(self.biasUnits[j])
                self.recNet.addConnection(FullConnection(self.biasUnits[j], self.layers[j+1]))
                continue
            if j == (len(self.biasUnits)-2):
                self.genNet.addModule(self.biasUnits[j])
                self.genNet.addConnection(FullConnection(self.biasUnits[j], self.layers[j-1]))
                continue
            if j == (len(self.biasUnits)-1):
                self.genNet.addModule(self.biasUnits[j])
                self.genNet.addConnection(FullConnection(self.biasUnits[j], self.layers[j-1]))
                break            
            else:
                self.recNet.addModule(self.biasUnits[j])
                self.genNet.addModule(self.biasUnits[j])
                self.recNet.addConnection(FullConnection(self.biasUnits[j], self.layers[j+1]))
                self.genNet.addConnection(FullConnection(self.biasUnits[j], self.layers[j-1]))
        
        
    
    
    
        
        