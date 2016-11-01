import pybrain

from scipy import random, array, empty
from HelmholtzNetwork import HelmholtzNetwork
from random import shuffle
import math

def HHMTrainer():
    
    def __init__(self, HelmholtzMachine, dataset, distribution):
        self.dist = distribution
        self.genDist = {}
        self.hhm = HelmholtzMachine
        self.recNet = self.hhm.recNet
        self.genNet = self.hhm.genNet
        self.setData(dataset)
        self.setDist()
        self.samplesGenerated = len(self.dist.keys())
        
    def setData(self, dataset):
        self.ds = dataset

    def setDist(self):
        for k in self.dist.keys():
            self.genDist[k] = 1

    def wakePhase(self, datapoint):
        self.recNet.activate(datapoint)
        self.genNet._adjustWeights()

    def sleepPhase(self):
        self.genNet.activate([1])
        self.recNet._adjustWeights()
            
        
    def train(self, iterations, desiredDivergence):
        for i in iterations:
            wakePhase(self.ds[i%len(self.ds)])
            sleepPhase()
            kldiv = calcKLDivergence()
            if kldiv <= desiredDivergence:
                return kldiv
        return kldiv


    def calcKLDivergence(self):
        divergence = 0
        for j in self.dist.keys():
            divergence += self.dist[j]*(math.log(self.dist[j])/(self.genDist[j])/self.samplesGenerated)
        return divergence

    def updateGenerativeDistribution(self, sample):
        self.samplesGenerated += 1
        if sample in self.genDist.keys():
            self.genDist[sample] += 1




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