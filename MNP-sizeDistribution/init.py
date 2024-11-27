import numpy as np

class Data:
  def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=4.8e5, meanCore=29.53*1e-9, stdCore= .09, 
                 coatingSize = 16*1e-9, temp=300, alpha=1, kAnis=3e3, visc=1e-3, fieldAmpl=20e-3,
                 nPeriod=3, fieldFreq=25e3, rsol=20, nParticle=10000):

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