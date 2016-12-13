from HHMEnvironment import HHMEnvironment
from HHMTrainer import HHMTrainer
from HelmholtzMachine import HelmholtzMachine
import random
import math
import numpy

verticalEnv = HHMEnvironment(9, [([0, 0, 1, 0, 0, 1, 0, 0, 1], .1111), ([0, 1, 0, 0, 1, 0, 0, 1, 0], .1111), ([1, 0, 0, 1, 0, 0, 1, 0, 0], .1111), ([0, 1, 1, 0, 1, 1, 0, 1, 1], .1111), ([1, 1, 0, 1, 1, 0, 1, 1, 0], .1111), ([1, 0, 1, 1, 0, 1, 1, 0, 1], .1111), ([0, 0, 0, 1, 1, 1, 1, 1, 1], .0556), ([1, 1, 1, 0, 0, 0, 1, 1, 1], .0556), ([0, 0, 0, 1, 1, 1, 0, 0, 0], .0556), ([1, 1, 1, 1, 1, 1, 0, 0, 0], .0556), ([0, 0, 0, 0, 0, 0, 1, 1, 1], .0555), ([1, 1, 1, 0, 0, 0, 0, 0, 0], .0555)])
horizontalEnv = HHMEnvironment(9, [([0, 0, 1, 0, 0, 1, 0, 0, 1], .0556), ([0, 1, 0, 0, 1, 0, 0, 1, 0], .0556), ([1, 0, 0, 1, 0, 0, 1, 0, 0], .0556), ([0, 1, 1, 0, 1, 1, 0, 1, 1], .0556), ([1, 1, 0, 1, 1, 0, 1, 1, 0], .0555), ([1, 0, 1, 1, 0, 1, 1, 0, 1], .0555), ([0, 0, 0, 1, 1, 1, 1, 1, 1], .1111), ([1, 1, 1, 0, 0, 0, 1, 1, 1], .1111), ([0, 0, 0, 1, 1, 1, 0, 0, 0], .1111), ([1, 1, 1, 1, 1, 1, 0, 0, 0], .1111), ([0, 0, 0, 0, 0, 0, 1, 1, 1], .1111), ([1, 1, 1, 0, 0, 0, 0, 0, 0], .1111)])
cornerEnv = HHMEnvironment(9, [([0, 0, 1, 0, 0, 1, 0, 0, 1], .05), ([0, 1, 0, 0, 1, 0, 0, 1, 0], .15), ([1, 0, 0, 1, 0, 0, 1, 0, 0], .05), ([0, 1, 1, 0, 1, 1, 0, 1, 1], .05), ([1, 1, 0, 1, 1, 0, 1, 1, 0], .05), ([1, 0, 1, 1, 0, 1, 1, 0, 1], .15), ([0, 0, 0, 1, 1, 1, 1, 1, 1], .05), ([1, 1, 1, 0, 0, 0, 1, 1, 1], .15), ([0, 0, 0, 1, 1, 1, 0, 0, 0], .15), ([1, 1, 1, 1, 1, 1, 0, 0, 0], .05), ([0, 0, 0, 0, 0, 0, 1, 1, 1], .05), ([1, 1, 1, 0, 0, 0, 0, 0, 0], .05)])

environments = [verticalEnv, horizontalEnv, cornerEnv]

#this is dumb! the first one is so unnecessary!? methinks i fixed
def runExperiment(envs, iterations):
    agent = []
    for i in range(0, iterations):
        env = random.choice(envs)
        found = False
        for u in agent:
            data = countOccurances(env.sample(1000))
            generated = u[1].generateSamples(1000)
            if compareSamples(data, generated):
                results = u[1].generateSamples(100000)
                analyzeResults(env, results)
                found = True
                break
        if found:
            continue
        else:
            print "first time learning this!"
            data = env.sample(80000)
            hhm = HelmholtzMachine(9, 2, [6, 11], [.98, .025])
            trainer = HHMTrainer(hhm, data, [env.events, env.probabilities])
            agent.append((hhm, trainer))
            results = trainer.train(80000, .01)
            analyzeResults(env, results)






def analyzeResults(env, results):
    analysis = [[], []]
    for e in environments:
        if e == env:
            analysis[0] = env.eventProbs
    generations = []
    sortedVals =sorted(results.values())
    sortedVals.reverse()
    for r in range(0, 20):
        generations.append((results.keys()[results.values().index(sortedVals[r])], sortedVals[r]))
    analysis[1] = generations
    print analysis

def countOccurances(samples):
    dataprobs = {}
    for s in samples:
        for i in range(0, len(s)):
            s[i] = float(s[i])
        if str(numpy.array(s)) in dataprobs:
            dataprobs[str(numpy.array(s))] += 1
        else:
            dataprobs[str(numpy.array(s))] = 1

    for q in dataprobs:
        dataprobs[q] = float(dataprobs[q])/float(len(samples))
    return dataprobs

def compareSamples(d1, d2):
    d1maxes = sorted(d1.values())[-6:]
    for i in range(0, len(d1maxes)):
        d1maxes[i] = d1.keys()[d1.values().index(d1maxes[i])]
    d2maxes = sorted(d2.values())[-3:]

    for j in range(0, len(d2maxes)):
        d2maxes[j] = d2.keys()[d2.values().index(d2maxes[j])]
    print d2maxes
    if set(d2maxes).issubset(set(d1maxes)):
        print "match"
        return True
    else:
        return False


"""
def calcKLDivergence(data, generated):
    divergence = 0
    dataprobs = {}
    genprobs = {}
    print generated
    for d in data:
        for i in range(0, len(d)):
            d[i] = float(d[i])
        if str(numpy.array(d)) in dataprobs:
            dataprobs[str(numpy.array(d))] += 1
        else:
            dataprobs[str(numpy.array(d))] = 1

    for q in dataprobs:
        dataprobs[q] = float(dataprobs[q])/float(len(data))
    print "dataprobs", dataprobs

    for w in generated:
        generated[w] = float(generated[w])/float(len(generated))
    print "genprobs", generated
    #print dataprobs
    #print genprobs
    for j in range(0, len(dataprobs)):
        if dataprobs.keys()[j] in generated.keys():
            divergence += dataprobs[dataprobs.keys()[j]] * (
            math.log(dataprobs[dataprobs.keys()[j]] / generated[dataprobs.keys()[j]]))

        else:
            divergence += dataprobs[dataprobs.keys()[j]] * (
            math.log(dataprobs[dataprobs.keys()[j]] / .0001))



    return divergence
"""
runExperiment(environments, 100)