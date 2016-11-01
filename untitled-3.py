import pybrain


from HelmholtzMachine import HelmholtzMachine

mac = HelmholtzMachine(3, 3, [3, 4, 2])
mac.recNet.activate([0,0,1])
for m in mac.recNet.modulesSorted:
            #print m
            for c in mac.recNet.connections[m]:
                print c.params
                print c.whichBuffers(0)




