import numpy as np

class Params:
    def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=3e5, dCoreMean=29.53*1e-9, dCoreStd = .09,
                 coatingSize = 16*1e-9, temp=300, alpha=1, kAnis=3e3, visc=1e-3, filedStrength=20e-3,
                 nPeriod=3, fieldFreq=25e3, nParticle=1000):
        self.kB = kB
        self.gamGyro = gamGyro
        self.Ms = Ms
        self.dCoreMean = dCoreMean
        self.dCoreStd = dCoreStd
        self.coatingSize = coatingSize
        self.temp = temp
        self.alpha = alpha
        self.kAnis = kAnis
        self.visc = visc
        self.fieldB = filedStrength
        self.nPeriod = nPeriod
        self.f_excitation = fieldFreq
        self.nParticle = nParticle

        self.calculate_params()

    def calculate_params(self):
        self.dCore = np.random.lognormal(mean=np.log(self.dCoreMean), sigma=self.dCoreStd, size=(self.nParticle,1))
        self.vCore = np.pi / 6 * self.dCore ** 3
        self.dHyd = self.dCore + self.coatingSize
        self.vHyd = np.pi / 6 * self.dHyd ** 3
        self.mu = self.Ms * self.vCore
        self.unitlessAnis = self.kAnis * self.vCore / (self.kB * self.temp)
        self.t0 = self.mu / (2 * self.gamGyro * self.kB*self.temp) * (1 + self.alpha ** 2) / self.alpha  # Neel event time
        self.tB = 3 * self.visc * self.vHyd / (self.kB * self.temp)  # Brown relaxation time
        self.x0 = self.mu * self.fieldB / (self.kB * self.temp)  # unitless energy
        self.zeros = np.zeros((self.nParticle, 2))
        self.xi0 = np.column_stack((self.zeros, self.x0))
        self.tu = np.linspace(0, self.nPeriod, 30000 * self.nPeriod)
        self.dt = self.tu[1]
        self.ut = self.dt / self.f_excitation / self.tB
        self.vt = self.dt / self.f_excitation / self.t0

    def get_params(self):
        """Return the calculated values as a dictionary."""
        return {
            'unitless_energy': self.xi0,
            'normalized_brown_time_step': self.ut,
            'normalized_neel_time_step': self.vt,
            'neel_event_time': self.t0,
            'brown_relax_time': self.tB,
            'time_step': self.dt,
            'evaluation_time_length': len(self.tu),
            'unitless_anisotropy': self.unitlessAnis,
            'constant_damping': self.alpha,
            'number_of_particles': self.nParticle,
            'coatingSize': self.coatingSize,
        }

    def set_params(self, **kwargs):
        """
        Set new values for the parameters and recalculate the derived values.
        kwargs: Dictionary of parameter names and their new values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'Params' object has no attribute '{key}'")
        self.calculate_params()
