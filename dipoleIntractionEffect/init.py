import numpy as np

class Params:
    def __init__(self, kB=1.381e-23, gamGyro=1.76e11, Ms=3e5, dCore=25e-9, dHyd=35e-9,
                 temp=300, alpha=1, kAnis=7e3, visc=1e-3, filedStrength=20e-3,
                 nPeriod=4, fieldFreq=25e3, nParticle=1000, minParticleDist=25):
        self.kB = kB
        self.gamGyro = gamGyro
        self.Ms = Ms
        self.dCore = dCore
        self.dHyd = dHyd
        self.temp = temp
        self.alpha = alpha
        self.kAnis = kAnis
        self.visc = visc
        self.fieldB = filedStrength
        self.nPeriod = nPeriod
        self.f_excitation = fieldFreq
        self.nParticle = nParticle
        self.minDist = minParticleDist

        self.calculate_params()

    def calculate_params(self):
        self.vCore = np.pi / 6 * self.dCore ** 3
        self.vHyd = np.pi / 6 * self.dHyd ** 3
        self.mu = self.Ms * self.vCore
        self.unitlessAnis = self.kAnis * self.vCore / (self.kB * self.temp)
        self.t0 = self.mu / (2 * self.gamGyro * self.kB*self.temp) * (1 + self.alpha ** 2) / self.alpha  # Neel event time
        self.tB = 3 * self.visc * self.vHyd / (self.kB * self.temp)  # Brown relaxation time
        self.xi0 = self.mu * self.fieldB / (self.kB * self.temp)  # unitless energy
        self.dt_init = min(self.f_excitation * self.t0, self.f_excitation * self.tB)  # unitless time step
        self.tu = np.linspace(0, self.nPeriod, 30000 * self.nPeriod)
        #self.tu = np.linspace(0, self.nPeriod, 1000 * self.nPeriod)
        self.dt = self.tu[1]
        self.ut = self.dt / self.f_excitation / self.tB
        self.vt = self.dt / self.f_excitation / self.t0
        self.cordinates = self.genCordinates(self.nParticle, self.minDist)

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
            'magnetic_moment': self.mu,
            'Boltzman_energy': self.kB*self.temp,
            'particle_cordinates': self.cordinates
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