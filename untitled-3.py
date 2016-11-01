import pybrain
import numpy
from HHMTrainer import HHMTrainer
from HelmholtzMachine import HelmholtzMachine

elements = numpy.array([1, 2, 3])
probs = numpy.array([.7, .2, .1])
distribution = {'[0, 1, 1]': .7, '[0, 0, 1]': .2, '[1, 1, 1]': .1}
data = []

for i in range(0, 999):
    event = numpy.random.choice(elements, 1, list(probs))
    if event[0] == 1:
        data.append([0, 1, 1])
    if event[0] == 2:
        data.append([0, 0, 1])
    if event[0] == 3:
        data.append([1, 1, 1])



mac = HelmholtzMachine(3, 3, [3, 4, 2])
trainer = HHMTrainer(mac, data, distribution)

trainer.train(1000, .01)
"""
mac.recNet.activate([0,0,1])
for m in mac.recNet.modulesSorted:
            #print m
            for c in mac.recNet.connections[m]:
                print c.params
                print c.whichBuffers(0)


"""

