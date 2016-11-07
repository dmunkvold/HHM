import pybrain
import scipy
from HHMSigmoidLayer import HHMSigmoidLayer
from scipy import random, array, empty
import numpy
from HelmholtzNetwork import HelmholtzNetwork
from HHMFullConnection import HHMFullConnection
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from HHMLinearLayer import HHMLinearLayer
from HHMBiasUnit import HHMBiasUnit
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit, SoftmaxLayer
from pybrain import *

class HelmholtzMachine():
    
    #the helmholtz machine class is a container for the recognition and generation modules
    
    def __init__(self, indim, hLayers, LayerDims):
        
        
        self.recNet = HelmholtzNetwork()
        self.genNet = HelmholtzNetwork()
        self.indim = indim
        self.layerDims = LayerDims
        self.layers = numpy.empty(hLayers + 1, dtype=HHMSigmoidLayer)
        self.biasUnits = numpy.empty(hLayers + 1, dtype = BiasUnit)
        print(self.layers)
        self.layers[0] = HHMLinearLayer(indim, name = "HHM Linear Input/Output Layer")
        self.recNet.addInputModule(self.layers[0])
        self.genNet.addOutputModule(self.layers[0])

        
        #declaring and adding all other layers and connections (full for now)
        for i in range(1, len(self.layers)):
            self.layers[i] = HHMSigmoidLayer(self.layerDims[i-1],  "HHM Sigmoid Layer " + str(i))
            self.recNet.addModule(self.layers[i])
            self.genNet.addModule(self.layers[i])
            #.1 worked earlier
            self.recNet.addConnection(HHMFullConnection((.1), self.layers[i-1], self.layers[i]))
            #.001 worked earlier
            self.genNet.addConnection(HHMFullConnection((.1), self.layers[i], self.layers[i-1]))
        
        
        genBias = HHMBiasUnit(1, "Generative Input Bias")
        self.genNet.addInputModule(genBias)
        #.01 worked for biases
        self.genNet.addConnection(HHMFullConnection((.01), genBias, self.layers[len(self.layers)-1]))
        """
        #declaring and adding bias units and connections
        for j in range(0, len(self.biasUnits)):
            self.biasUnits[j] = HHMBiasUnit(1, "BiasUnit " + str(j))
            if j == 0:
                self.recNet.addModule(self.biasUnits[j])
                self.recNet.addConnection(HHMFullConnection(.01, self.biasUnits[j], self.layers[j+1]))
                continue
            if j == (len(self.biasUnits)-1):
                self.genNet.addModule(self.biasUnits[j])
                self.genNet.addConnection(HHMFullConnection(.01, self.biasUnits[j], self.layers[j-1]))
                continue
            else:
                self.recNet.addModule(self.biasUnits[j])
                self.genNet.addModule(self.biasUnits[j])
                self.recNet.addConnection(HHMFullConnection(.01, self.biasUnits[j], self.layers[j+1]))
                self.genNet.addConnection(HHMFullConnection(.01, self.biasUnits[j], self.layers[j-1]))
        """
        self.recNet.sortModules()
        self.genNet.sortModules()

        for p in range(0, len(self.recNet.params)):
            self.recNet.params[p] = 0.
        for q in range(0, len(self.genNet.params)):
            self.genNet.params[q] = 0.

    
    def printMachine(self):
        print("recognition network:")
        print(self.recNet)
        print("generation network:")
        print(self.genNet)
    
        
        