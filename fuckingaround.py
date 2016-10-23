import pybrain
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit, SoftmaxLayer

#setting up layers for both networks

d = LinearLayer(4)
y = SigmoidLayer(3)
x = LinearLayer(2)
b = LinearLayer(1)
#getting error with this, alternatively just going to use x as input module and feed in 1
#b = BiasUnit(name='genBias')

# building recognition network ////////////////////////////////////////

rec = FeedForwardNetwork()

# adding layers
rec.addInputModule(d)
rec.addModule(y)
rec.addOutputModule(x)
# adding connections
vR = FullConnection(d, y)
wR = FullConnection(y, x)
rec.addConnection(vR)
rec.addConnection(wR)

#sorting
rec.sortModules()

#initializing params at 0
lengthOfRecParams = len(rec.params)
recInitialParams = [0] * lengthOfRecParams
rec._setParameters(recInitialParams)


# building generative network /////////////////////////////////////////

gen = FeedForwardNetwork()

# adding layers
gen.addInputModule(b)
gen.addModule(x)
gen.addModule(y)
gen.addOutputModule(d)

# ading connections
bG = FullConnection(b, x)
wG = FullConnection(x, y)
vG = FullConnection(y, d)
#gen.addConnection(bG)
gen.addConnection(wG)
gen.addConnection(vG)

# sorting
gen.sortModules()

#initializing params at 0
lengthOfGenParams = len(gen.params)
genInitialParams = [0] * lengthOfGenParams
gen._setParameters(genInitialParams)

rec.activate((0,1, 0, 1))

#def trainTheBrain(rec, gen, dataPiece):
    








