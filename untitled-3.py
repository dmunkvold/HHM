import pybrain
import numpy
import scipy
from HHMTrainer import HHMTrainer
from HelmholtzMachine import HelmholtzMachine

elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
probs = [.1111, .1111,.1111,.1111,.1111,.1111, .0556, .0556, .0556, .0556, .0556, .0556]
distribution = [[numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1]), numpy.array([0, 1, 0, 0, 1, 0, 0, 1, 0]), numpy.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
                 numpy.array([0, 1, 1, 0, 1, 1, 0, 1, 1]), numpy.array([1, 1, 0, 1, 1, 0, 1, 1, 0]), numpy.array([1, 0, 1, 1, 0, 1, 1, 0, 1]),
                 numpy.array([0, 0, 0, 1, 1, 1, 1, 1, 1]), numpy.array([1, 1, 1, 0, 0, 0, 1, 1, 1]), numpy.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
                 numpy.array([1, 1, 1, 1, 1, 1, 0, 0, 0]), numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1]), numpy.array([1, 1, 1, 0, 0, 0, 0, 0, 0])], [.1111, .1111, .1111, .1111, .1111, .1111, .0556, .0556, .0556, .0556, .0556, .0556]]
data = []
stuff = [0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0, 59999):
    event = numpy.random.choice(elements, 1, list(probs))
    if event[0] == 1:
        data.append([0, 0, 1, 0, 0, 1, 0, 0, 1])
        stuff[0] += 1
    if event[0] == 2:
        data.append([0, 1, 0, 0, 1, 0, 0, 1, 0])
        stuff[1] += 1
    if event[0] == 3:
        data.append([1, 0, 0, 1, 0, 0, 1, 0, 0])
        stuff[2] += 1
    if event[0] == 4:
        data.append([0, 1, 1, 0, 1, 1, 0, 1, 1])
        stuff[3] += 1
    if event[0] == 5:
        data.append([1, 1, 0, 1, 1, 0, 1, 1, 0])
        stuff[4] += 1
    if event[0] == 6:
        data.append([1, 0, 1, 1, 0, 1, 1, 0, 1])
        stuff[5] += 1
    if event[0] == 7:
        data.append([0, 0, 0, 1, 1, 1, 1, 1, 1])
        stuff[6] += 1
    if event[0] == 8:
        data.append([1, 1, 1, 0, 0, 0, 1, 1, 1])
        stuff[7] += 1
    if event[0] == 9:
        data.append([0, 0, 0, 1, 1, 1, 0, 0, 0])
        stuff[8] += 1
    if event[0] == 10:
        data.append([1, 1, 1, 1, 1, 1, 0, 0, 0])
        stuff[9] += 1
    if event[0] == 11:
        data.append([0, 0, 0, 0, 0, 0, 1, 1, 1])
        stuff[10] += 1
    if event[0] == 12:
        data.append([1, 1, 1, 0, 0, 0, 0, 0, 0])
        stuff[11] += 1

print stuff
#.9, .1, and 01 on 18, 9, 3
#.9, .01, .001 
#.9, .001, .001 was better
#.99 worked worse
#mac = HelmholtzMachine(9, 2, [18, 4, 1], [.9, .01, .001]) 0.580822946929
#mac = HelmholtzMachine(9, 3, [18, 4, 1], [.9, .01, .001]) 0.597912031125
# .1 for second was worse, .798
# 2 node second hidden worked aight, .700
#7 wasnt better than four
#two wasnt better than 4
#5 wasnt better
mac = HelmholtzMachine(9, 2, [36, 4], [.75, 01])
trainer = HHMTrainer(mac, data, distribution)

#trainer.train(60000, .01)
"""
mac.recNet.activate([0,0,1])
for m in mac.recNet.modulesSorted:
            #print m
            for c in mac.recNet.connections[m]:
                print c.params
                print c.whichBuffers(0)


"""

