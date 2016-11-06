import numpy

class HHMEnvironment():
    
    def __init__(self, dim, eventProbabilities):
        
        self.dim = dim
        self.eventProbs = eventProbabilities
        self.events = []
        self.probabilities = []
        self.dummyList = []
        self.parseEventProbs()
        self.generated = [0] * len(self.eventProbs)
        
        
    def parseEventProbs(self):
        for i in self.eventProbs:
            self.events.append(i[0])
            self.probabilities.append(i[1])
            self.dummyList.append(self.eventProbs.index(i))
            
    def sample(self, sampleSize):
        samples = []
        for j in range(0, sampleSize):
            sample = numpy.random.choice(self.dummyList, p=self.probabilities)
            self.generated[sample] += 1
            samples.append(self.events[sample])
        return samples
            