import numpy

class HHMEnvironment():
    
    def __init__(self, dim, eventProbabilities):
        
        self.dim = dim
        self.eventProbs = eventProbabilities
        self.events = []
        self.probabilities = []
        self.dummyList = []
        self.parseEventProbs()
        
        
    def parseEventProbs(self):
        for i in self.eventProbs:
            self.events.append(i[0])
            self.probabilities.append(i[1])
            self.dummyList.append(self.eventProbs.index(i))
            
    def sample(self, sampleSize):
        print self.dummyList
        samples = []
        for j in range(0, sampleSize):
            sample = numpy.random.choice(self.dummyList, p=self.probabilities)
            print sample
            samples.append(self.events[sample])
        print samples
        return samples
            

env1=HHMEnvironment(2, [([1,1], .9),([0,0], .1)])

env1.sample(10)