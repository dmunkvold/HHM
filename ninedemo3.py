from HHMEnvironment import HHMEnvironment
from HHMTrainer import HHMTrainer
from HelmholtzMachine import HelmholtzMachine

env = HHMEnvironment(9, [([0, 0, 1, 0, 0, 1, 0, 0, 1], .05), ([0, 1, 0, 0, 1, 0, 0, 1, 0], .15), ([1, 0, 0, 1, 0, 0, 1, 0, 0], .05), ([0, 1, 1, 0, 1, 1, 0, 1, 1], .05), ([1, 1, 0, 1, 1, 0, 1, 1, 0], .05), ([1, 0, 1, 1, 0, 1, 1, 0, 1], .15), ([0, 0, 0, 1, 1, 1, 1, 1, 1], .05), ([1, 1, 1, 0, 0, 0, 1, 1, 1], .15), ([0, 0, 0, 1, 1, 1, 0, 0, 0], .15), ([1, 1, 1, 1, 1, 1, 0, 0, 0], .05), ([0, 0, 0, 0, 0, 0, 1, 1, 1], .05), ([1, 1, 1, 0, 0, 0, 0, 0, 0], .05)])


#hhm = HelmholtzMachine(9, 2, [21, 7], [.95, .01]) got to 0.208668853232!
#hhm = HelmholtzMachine(9, 2, [21, 7], [.95, .1]) got to 0.0793812953608!
#HelmholtzMachine(9, 2, [21, 7], [.975, .01]) got o\to 0.064424758814
# HelmholtzMachine(9, 2, [21, 7], [.975, .1]) got to 0.0355046686714!
# HelmholtzMachine(9, 2, [21, 7], [.98, .1]) got to 0.032075310275!

#.15 got to .11
#so in this one, patterns with all four corners the same are most frequent. hhm = HelmholtzMachine(9, 2, [6, 11], [.98, .025]) works enough!
data = env.sample(80000)
print env.generated
hhm = HelmholtzMachine(9, 2, [6, 11], [.98, .025])
#print hhm.recNet, hhm.genNet

    
#for m in range(0, len(hhm.recNet.modulesSorted)):
#    print hhm.recNet.modulesSorted[m]
#    for c in hhm.recNet.connections[hhm.recNet.modulesSorted[m]]:
#        print c
#        print c.learningRate
        
#for m in range(0, len(hhm.genNet.modulesSorted)):
#    print hhm.genNet.modulesSorted[m]
#    for c in hhm.genNet.connections[hhm.genNet.modulesSorted[m]]:
#        print c
#        print c.learningRate
        
    

trainer = HHMTrainer(hhm, data, [env.events, env.probabilities])
print trainer.samplesGenerated
trainer.train(80000, .01)

