import numpy as np

class Data:
    def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=5e5, dCore=30e-9, dHyd=40e-9,
                    temp=300, alpha=1, kAnis=3e3, visc=1e-3, filedAmpl=20e-3,
                    nPeriod=3, fieldFreq=25e3, rsol=20, nParticle=1000, minParticleDist=25):
        self.kB = kB
        self.gamGyro = gamGyro
        self.Ms = Ms
        self.dCore = dCore
        self.dHyd = dHyd
        self.temp = temp
        self.alpha = alpha
        self.kAnis = kAnis
        self.visc = visc
        self.filedAmpl = filedAmpl
        self.nPeriod = nPeriod
        self.fieldFreq = fieldFreq
        self.rsol = rsol # number of samples per period -> for signal resolution
        self.nParticle = nParticle
        self.minDist = minParticleDist

    def genCordinates(self, numParticles, mindist):
        particles_per_dim = int(np.ceil(numParticles**(1/3)))
        cordinates = []
        count = 0
        for i in range(particles_per_dim):
            for j in range(particles_per_dim):
                for k in range(particles_per_dim):
                    if count < numParticles:
                        x = i * mindist
                        y = j * mindist
                        z = k * mindist
                        cordinates.append([x, y, z])
                        count += 1

        return np.array(cordinates) * 1e-9
