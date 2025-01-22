import numpy as np

class Data:
  def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=4.8e5, meanCore=29.53*1e-9, stdCore= .09, 
                 coatingSize = 16*1e-9, temp=300, alpha=1, kAnis=3e3, visc=1e-3, fieldAmpl=20e-3,
                 nPeriod=3, fieldFreq=20e3, rsol=20, nParticle=10000):

    self.kB = kB
    self.gamGyro = gamGyro
    self.Ms = Ms
    self.nParticle = nParticle
    self.meanCore = meanCore
    self.stdCore = stdCore
    self.coatingSize = coatingSize
    self.temp = temp
    self.alpha = alpha
    self.kAnis = kAnis
    self.visc = visc
    self.fieldAmpl = fieldAmpl
    self.nPeriod = nPeriod
    self.fieldFreq = fieldFreq
    self.rsol = rsol # number of samples per period -> for signal resolution

class Params:
  def __init__(self, data):
    al = data.alpha
    gam = data.gamGyro
    visc = data.visc
    cycs = data.nPeriod
    num = data.nParticle
    kT = data.kB*data.temp
    B = data.fieldAmpl
    f = data.fieldFreq
    Ms = data.Ms
    ka = data.kAnis

    # size dependent parameters
    self.dco = np.random.lognormal(mean=np.log(data.meanCore), sigma=data.stdCore, size=(num,1))
    self.dhy = self.dco + data.coatingSize
    self.Vc = 1 / 6 * np.pi * self.dco ** 3
    self.Vh = 1 / 6 * np.pi * self.dhy ** 3
    self.mu = Ms * self.Vc
    self.sig = ka * self.Vc / kT
    self.t0 = self.mu / (2 * gam * kT) * (1 + al ** 2) / al
    self.tB = 3 * visc * self.Vh / kT
    self.xi0 = self.mu * B / (2*kT)
    self.fs = data.rsol*2*f 

    # sampling
    self.dt = 1/self.fs
    self.tf = cycs*(1/f)
    self.lent = int(np.ceil(self.tf/self.dt))
    wrf = 1e-3 # this is used to reduce the Wiener noise
    self.ut = wrf*self.dt/self.tB
    self.vt = wrf*self.dt/self.t0