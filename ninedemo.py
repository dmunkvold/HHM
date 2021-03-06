from HHMEnvironment import HHMEnvironment
from HHMTrainer import HHMTrainer
from HelmholtzMachine import HelmholtzMachine

env = HHMEnvironment(9, [([0, 0, 1, 0, 0, 1, 0, 0, 1], .1111), ([0, 1, 0, 0, 1, 0, 0, 1, 0], .1111), ([1, 0, 0, 1, 0, 0, 1, 0, 0], .1111), ([0, 1, 1, 0, 1, 1, 0, 1, 1], .1111), ([1, 1, 0, 1, 1, 0, 1, 1, 0], .1111), ([1, 0, 1, 1, 0, 1, 1, 0, 1], .1111), ([0, 0, 0, 1, 1, 1, 1, 1, 1], .0556), ([1, 1, 1, 0, 0, 0, 1, 1, 1], .0556), ([0, 0, 0, 1, 1, 1, 0, 0, 0], .0556), ([1, 1, 1, 1, 1, 1, 0, 0, 0], .0556), ([0, 0, 0, 0, 0, 0, 1, 1, 1], .0555), ([1, 1, 1, 0, 0, 0, 0, 0, 0], .0555)])


#hhm = HelmholtzMachine(9, 2, [21, 7], [.95, .01]) got to 0.208668853232!
#hhm = HelmholtzMachine(9, 2, [21, 7], [.95, .1]) got to 0.0793812953608!
#HelmholtzMachine(9, 2, [21, 7], [.975, .01]) got o\to 0.064424758814
# HelmholtzMachine(9, 2, [21, 7], [.975, .1]) got to 0.0355046686714!
# HelmholtzMachine(9, 2, [21, 7], [.98, .1]) got to 0.032075310275!
# 100, 7 did the same as the above eve with lower rates: .15, .01
#.15 got to .11
#hhm = HelmholtzMachine(9, 2, [6, 7], [.9, .01]) worked best realy ao far
#9 in second hidden worked okay but focused too much on most frequent patterns
#10 worked fairly well for second hiddem
#hhm = HelmholtzMachine(9, 2, [6, 11], [.98, .0275]) was best so far

#so in this one, all vertical lines happen more frequently. HelmholtzMachine(9, 2, [6, 11], [.98, .025]) is currently standard.
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
