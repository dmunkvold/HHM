import pybrain

from scipy import random, array, empty
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit, SoftmaxLayer
from pybrain import *

"""
input hiddenLayers is number of hidden layers, the next input, hiddenLayerSizes, is an array of 
hidden layer sizes, in order of "bottom" to "top." Therefore length of hiddenLayerSizes = 
hiddenLayers Then there's an input dimension, inputDim, and
output dimension, which may be unnecessary, outputDim
"""
def buildHHM(hiddenLayers, hiddenLayerSizes, inputDim, outputDim):
    
    #declaring layers
    inputLayer = LinearLayer(inputDim, name = "inputLayer")
    outputLayer = LinearLayer(outputDim, name = "outputLayer")
    hLayers = [0]*hiddenLayers
    for h in range(0, hiddenLayers):
        name = "sigmoid" + str(h)
        newLayer = SigmoidLayer(hiddenLayerSizes[h], name = name)
        hLayers[h] = newLayer
    
    #declaring bias units
    biasUnits = [0] * (hiddenLayers + 2)
    for u in range(0, len(biasUnits)):
        name = "bias"+ str(u)
        biasUnits[u] = BiasUnit(name = name)
    
    #declaring connections
    input_to_hidden = FullConnection(inputLayer, hLayers[0])
    hidden_to_output = FullConnection(hLayers[len(hLayers)-1], outputLayer)
    recBiasIn = FullConnection(biasUnits[0], hLayers[0])
    recBiasOut = FullConnection(biasUnits[len(biasUnits)-2], outputLayer)
    recWeights = [0] * (len(hLayers) -1)
    recBias = [0] * (len(hLayers)+1)
    
    #for generative network 
    output_to_hidden = FullConnection(outputLayer, hLayers[len(hLayers)-1])
    hidden_to_input = FullConnection(hLayers[0], inputLayer)
    genBiasIn = FullConnection(biasUnits[len(biasUnits)-1], hLayers[len(hLayers)-1])
    genBiasOut = FullConnection(biasUnits[1], inputLayer)
    genWeights = [0] * (len(hLayers) -1)
    genBias = [0] * (len(hLayers)+1) 
    
    #declaring weights
    for j in range(0, hiddenLayers-1):
        recWeights[j] = FullConnection(hLayers[j], hLayers[j+1])
        genWeights[j] = FullConnection(hLayers[hiddenLayers - 1 - j], hLayers[hiddenLayers - 2 - j])
        recBias[j] = FullConnection(biasUnits[j+1], hLayers[j+1])
        genBias[j] = FullConnection(biasUnits[len(biasUnits) - 1 - j], hLayers[hiddenLayers - 2 - j])
    
     
    
    #building recognition network, assigning in and out
    rec = FeedForwardNetwork()
    rec.addInputModule(inputLayer)
    rec.addOutputModule(outputLayer)
    
    #building generative network assigning in and out
    gen = FeedForwardNetwork()
    gen.addInputModule(outputLayer)
    gen.addOutputModule(inputLayer)
    
    #adding hidden layers to both network
    for r in range (0, len(hLayers)):
        gen.addModule(hLayers[r])
        rec.addModule(hLayers[r])
    
    for k in range(0, len(biasUnits)):
        rec.addModule(biasUnits[k])
        gen.addModule(biasUnits[k])
    
    #adding connections
    rec.addConnection(input_to_hidden)
    rec.addConnection(hidden_to_output)
    rec.addConnection(recBiasIn)
    rec.addConnection(recBiasOut)
    gen.addConnection(output_to_hidden)
    gen.addConnection(hidden_to_input)
    gen.addConnection(genBiasIn)
    gen.addConnection(genBiasOut)

    for b in range(0, len(recWeights)):
        gen.addConnection(genWeights[b])
        gen.addConnection(genBias[b])
        rec.addConnection(recWeights[b])
        rec.addConnection(recBias[b])
        
    
    rec.sortModules()
    gen.sortModules()

    print rec.activate([1,0,0,1])
    recMods = list(rec.modules)
    print recMods[4].dim
    return rec, gen
    
buildHHM(2, [3, 4], 4, 3)
#at this point, I think i have built it, need to examine structure and build trainer