import numpy as np
from init import *
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser

Ht = lambda f, t: np.cos(2*np.pi*f*t)

def low_pass_filter(signal, order, fs, fpass):
  fnyq = 0.5 * fs        # Nyquist frequency
  cutoff = fpass / fnyq  # Normalized cutoff frequency
  b, a = butter(order, cutoff, btype='low')
  return filtfilt(b, a,  signal)

def fourier_derivative(signal, dx):
    """
    Compute the derivative of a signal using the Fourier transform.
    Parameters:
        signal (array): The input signal.
        dx (float): The spacing between points in the input signal.
    Returns:
        array: The derivative of the signal.
    """
    signal_length = len(signal)
    freq = fftfreq(signal_length, d=dx)  # Frequency domain
    spectrum = fft(signal)   # Fourier transform of the signal
    derivative_spectrum = 1j * 2 * np.pi * freq * spectrum  # Differentiate in frequency domain
    return np.real(ifft(derivative_spectrum)) 

# Neel relaxation time Fannin and Charless
def NeelRelaxation(sig, t0):
  if sig < 1:
    return t0 * (1 - 2 / 5 * sig + 48 / 875 * sig ** 2) ** (-1)
  else:
    return t0 * np.exp(sig) / 2 * np.sqrt(np.pi / sig ** 3)
  
def midPointVal(x):
  y = np.zeros(len(x)-1)
  for i in range(len(x)-1):
    y[i] = (x[i]+x[i+1])/2
  return y
  
def dxi_dt(xi0, sig, dt, f, lent, mlow, nlow):
  t = np.array([i*dt for i in range(lent)])
  He = Ht(f,t)
  dift = np.diff(t)
  dH = np.diff(He)
  dm = np.diff(mlow)
  dn = np.diff(nlow)
  newm = midPointVal(mlow)
  newn = midPointVal(nlow)
  dxidt = xi0*(dH/dift) + sig * ( (dm/dift) * newn + newm * (dn/dift) + newm * newn * ( dn/dift ) ) 
  return dxidt
  
def ftSignal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, f, lent, pz, num, mu, cycs):
  st = np.zeros(lent-1)
  sf = np.zeros(lent-1)
  winlen = (np.where(xiH == np.min(xiH))[0][0] - np.where(xiH == np.max(xiH))[0][0])
  dxidt = dxi_dt(xi0, sig, dt, f, lent, mlow, nlow)
  for i in range(cycs):
    _, leftpsf, _, rightpsf = psf_xiH(xiH, sigHlow, mlow, i+1)
    _, leftpsf, _, rightpsf = psf_xiH(xiH, sigHlow, mlow, i+1)
    j = 2*i # the first half of a period has leftpsf kernel and the other half has rightpsf kernel
    st[j*winlen:(j+1)*winlen-1] = -(pz*mu*num*num/lz)*(-leftpsf)*dxidt[j*winlen:(j+1)*winlen-1]
    st[(j+1)*winlen:(j+2)*winlen-1] = -(pz*mu*num*num/lz)*(-rightpsf)*dxidt[(j+1)*winlen:(j+2)*winlen-1]
  uk = np.fft.fft(st)
  sf = abs(np.fft.fftshift(uk))
  return st, sf
  
def MPI_Langevin_std_init(data):
    return Params(data)

def MPI_Langevin_std_core(data):
  params = MPI_Langevin_std_init(data)
  al = data.alpha
  num = data.nParticle
  sig = params.sig
  xi0 = params.xi0
  dt = params.dt
  lent = params.lent
  ut = params.ut
  vt = params.vt

  M = np.zeros((lent, 3))
  N = np.zeros((lent, 3))
  sigt = np.zeros((lent, 3))

  m = np.tile([1, 0, 0], (num, 1))
  n = m.copy()
  xI= np.tile([0, 0, xi0], (num, 1))               
  sign = np.tile([0, 0, sig], (num, 1))     

  for j in range(lent):
      M[j, :] = np.mean(m, axis=0)
      N[j, :] = np.mean(n, axis=0)
      sigt[j, :] = np.mean(sign, axis=0)
      a = np.sum(m * n, axis=1)
      dn = sig*a[:, np.newaxis] * (m - a[:, np.newaxis] * n) * ut + np.cross(np.random.randn(num, 3), n) * np.sqrt(ut)
      n = n + dn
      n = n / np.linalg.norm(n, axis=1, keepdims=True)
      xi = xI * np.cos(2 * np.pi * j * dt) + sig*a[:, np.newaxis]*n
      sign = sig * a[:, np.newaxis] * n
      h = np.random.randn(num, 3)
      f1 = np.cross(xi / al + np.cross(m, xi), m) 
      g1 = np.cross(h / al + np.cross(m, h), m)
      mb = m + f1 * vt + g1 * np.sqrt(vt)
      a2 = np.sum(mb * n, axis=1)
      xb = xI* np.cos(2 * np.pi * j * dt) + sig * a2[:, np.newaxis] * n
      h2 = np.random.randn(num, 3)
      f2 = np.cross(xb / al + np.cross(mb, xb), mb) 
      g2 = np.cross(h2 / al + np.cross(mb, h2), mb)
      m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
      m = m / np.linalg.norm(m, axis=1, keepdims=True)
      print('\r', 'time step in samples: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))
      
  return M[:, -1], N[:, -1], sigt[:, -1]

def psf_xiH(xiH, sigH, m, period):
  """ This function extract the psf and its related xiH values for a period 
  inputs:
    xiH: u*B/kT
    m: magnetization unite vecotr in direction of the applied field
    period: the desired period, T=1, T=2, T=3
  outputs:
    psf: dm/dxi
    leftxiH
    rightxiH """
  winlen = (np.where(xiH == np.min(xiH))[0][0] - np.where(xiH == np.max(xiH))[0][0])
  leftsigH = sigH[(2*period-2)*winlen : (2*period-1)*winlen] 
  rightsigH = sigH[(2*period-1)*winlen : (2*period)*winlen]
  leftxiH = xiH[(2*period-2)*winlen : (2*period-1)*winlen] # the second period is used [2*winlen : 4*winlen]
  rightxiH = xiH[(2*period-1)*winlen : (2*period)*winlen]
  leftxi = leftxiH + leftsigH
  rightxi = rightxiH + rightsigH
  leftm = m[(2*period-2)*winlen : (2*period-1)*winlen]
  rightm = m[(2*period-1)*winlen : (2*period)*winlen]
  leftpsf = savgol_filter(kaiser(winlen-1,14)*np.diff(leftm)/np.diff(leftxi), 500, 1, mode='nearest')
  rightpsf = savgol_filter(kaiser(winlen-1,14)*np.diff(rightm)/np.diff(rightxi), 500, 1, mode='nearest')
  return leftxiH, leftpsf, rightxiH, rightpsf

def fwhm_and_psf_peaks(xiH, psf):
  """ This function find peaks coordinates and fwhm data for each lobe 
  input:
    xiH: dimensionless magnetic field for one lobe
    psf: psf for one lobe
  output:
    a dictionary of peaks coordinates and fwhm of one lobe """
  peakpsf = max(psf)
  half_max = max(psf) / 2
  indices = np.where(psf >= half_max)[0]
  left_index = indices[0]
  right_index = indices[-1]
  v1 = xiH[left_index] 
  v2 = xiH[right_index] 
  fwhm = abs(v1 - v2)
  res = {
        'peakpsf': peakpsf,
        'peakxiH': xiH[np.where(psf==peakpsf)],
        'fwhm_left': v1,
        'fwhm_right': v2,
        'fwhm': fwhm,
    }
  return res

def fwhm(xiH, sigH, m, period):
  leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH, sigH, m, period)
  resl = fwhm_and_psf_peaks(leftxiH, leftpsf)
  resr = fwhm_and_psf_peaks(rightxiH, rightpsf)
  return max(resl['fwhm'], resr['fwhm'])
