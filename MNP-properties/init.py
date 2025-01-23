import numpy as np

class Data:
  def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=3e5, dCore=30e-9, dHyd=40e-9,
                 temp=300, alpha=1, kAnis=3e3, visc=1e-3, fieldAmpl=20e-3,
                 nPeriod=3, fieldFreq=25e3, rsol=20, nParticle=10000):

    self.kB = kB
    self.gamGyro = gamGyro
    self.Ms = Ms
    self.dCore = dCore
    self.dHyd = dHyd
    self.temp = temp
    self.alpha = alpha
    self.kAnis = kAnis
    self.visc = visc
    self.fieldAmpl = fieldAmpl
    self.nPeriod = nPeriod
    self.fieldFreq = fieldFreq
    self.rsol = rsol # number of samples per period -> for signal resolution
    self.nParticle = nParticle

class Params:
  def __init__(self, data):
    al = data.alpha
    gam = data.gamGyro
    visc = data.visc
    kT = data.kB*data.temp
    B = data.fieldAmpl
    f = data.fieldFreq
    Ms = data.Ms
    ka = data.kAnis
    cycs = data.nPeriod

    # size dependent parameters
    dco = data.dCore
    dhyd = data.dHyd
    self.Vc = np.pi / 6 * dco ** 3
    self.Vh = np.pi / 6 * dhyd ** 3
    self.mu = Ms * self.Vc
    self.sig = ka * self.Vc / kT
    self.t0 = self.mu / (2 * gam * kT) * (1 + al ** 2) / al
    self.tB = 3 * visc * self.Vh / kT
    self.xi0 = self.mu * B / (2*kT)

    # Simulation time grid (this is going to be simulated counterpart of the Analog signal)
    # self.ds = min(f * self.t0, f * self.tB)  # unitless time step
    # self.teval = np.arange(0, 1, self.ds)  # time grid for one period
    # self.tPts = int(2 ** np.ceil(np.log2(len(self.teval))))  # number of grids in one period
    # self.tu = np.linspace(0, cycs, self.tPts * cycs)  # unitless time grids 10 times larger
    self.tu = np.linspace(0, cycs, 15000*cycs)
    self.lent = len(self.tu)
    self.dt = self.tu[1]
    self.ut = self.dt / f / self.tB
    self.vt = self.dt / f / self.t0
    


