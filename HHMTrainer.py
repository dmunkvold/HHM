import pybrain

from scipy import random, array, empty

from random import shuffle

def HHMTrainer():
    
    def __init__(self, recModule, genModule, dataset, learningrate):    
         
        self.setData(dataset)
        self.rec = recModule
        self.gen = genModule
        self.lrate = learningrate
        
        
    def setData(dataset):
        self.ds = dataset
        
    def wakePhase(iterations):
        for i in iterations:
            nodeValues = [[]] * len(self.rec.modules//2)
            
        
    def train(self):
        
        


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