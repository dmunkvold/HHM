import pybrain

from scipy import random, array, empty
from HelmholtzNetwork import HelmholtzNetwork
from random import shuffle
import math
import numpy

class HHMTrainer():
    
    def __init__(self, HelmholtzMachine, dataset, distribution):
        self.dist = distribution
        self.genDist = [[], []]
        self.hhm = HelmholtzMachine
        self.recNet = self.hhm.recNet
        self.genNet = self.hhm.genNet
        self.setData(dataset)
        self.setDist()
        self.samplesGenerated = len(self.dist[0])
        #self.samplesMade = []
        
    def setData(self, dataset):
        self.ds = dataset

    def setDist(self):
        print self.genDist[0]
        print self.dist[0]
        for k in range(0, len(self.dist[0])):
            print self.genDist[0]
            self.genDist[0].append(self.dist[0][k])
            self.genDist[1].append(1)
            print self.genDist


    def wakePhase(self, datapoint):
        self.recNet.activate(datapoint)
        self.genNet._adjustWeights()

    def sleepPhase(self):
        sample = self.genNet.activate([1])
        for s in range(0, len(sample)):
            sample[s] = numpy.random.choice(numpy.arange(0, 2), p=[1 - sample[s], sample[s]])
        #self.samplesMade.append(sample)
        #print self.samplesMade
        #print sample
        self.updateGenerativeDistribution(sample)
        self.recNet._adjustWeights()
            
        
    def train(self, iterations, desiredDivergence):
        for i in range(0, iterations-1):
            self.wakePhase(self.ds[i%len(self.ds)])
            self.sleepPhase()
            kldiv = self.calcKLDivergence()
            if i%1000 == 0:
                print kldiv
            #print kldiv
            #if kldiv <= desiredDivergence:
                #return kldiv
        #print self.samplesMade
        return kldiv


    def calcKLDivergence(self):
        divergence = 0
        #print self.genDist
        #print self.dist
        for j in range(0, len(self.dist[0])):
            divergence += self.dist[1][j]*(math.log(self.dist[1][j]/(self.genDist[1][j]/float(self.samplesGenerated))))
        return divergence

    def updateGenerativeDistribution(self, sample):

        self.samplesGenerated += 1
        for h in range(0, len(self.genDist[0])):
            #print sample, numpy.array(self.genDist[0][h])
            if numpy.array_equal(sample, numpy.array(self.genDist[0][h])):
                #print "match"
                self.genDist[1][h] = 1 + self.genDist[1][h]
                break




"""
this is an outline, don't have internet rn

def HHMTrainer:

#fields

unsupervised dataset
helmholtz machine

#methods

wakePhase(iterations):
    for iterations:
        make a 2 dimensional nodeValues array that stores node values for each layers' nodes
        make array compProb that accounts for each generative weight to keep track of probs
        for layers in HHM:
            "forward implement" sample through recognition network, saving node values
        for layers in HHM:
            pass back down through generation network, using same node values to calculate compProb for each set of weights, save to list
        for all generative weights:
            adjust by values saved in compProb on gradient descent

sleepPhase(iterations):
    for iterations:
        make 2d array nodeValues that stores node values for each layers' nodes
        make 2d array compProb to save computed probabilities for each weight
        for layers in HHM:
            pass generated sample down, starting at sampling the bias unit, saving nodeValues.
            eventually it generates a "sample"
        for layers in HHM:
            pass sample back up through recognition network using node values, saving computed probs
        for recognition weights:
            adjust by computed probs on gradient descent
            


train(WSiterations, sizeOfWSiteration):
    for WSiterations
        wakePhase(sizeOfWSiterations)
        sleepPhase(sizeOfWSiterations)
        
trainUntilConvergence(sizeOfWSiterations):
    


"""