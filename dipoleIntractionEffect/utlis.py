import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import UnivariateSpline
from numba import njit, prange

Ht = lambda t, f: np.cos(2*np.pi*f*t)

def low_pass_filter(signal, order, fs, fpass):
  fnyq = 0.5 * fs        # Nyquist frequency
  cutoff = fpass / fnyq  # Normalized cutoff frequency
  b, a = butter(order, cutoff, btype='low')
  return filtfilt(b, a,  signal)

@njit
def dipole_field(particlei, particlej, mj, mu, kT):
    r = particlei - particlej
    lenr = np.linalg.norm(r)
    rhat = r / lenr  # Normalizing r
    B = (1e-7/lenr**3)*(3 * np.sum(mj*rhat) * rhat - mj)
    return (mu*mu/kT)*B

def contact_matrix(A, num, threshold):
    SerialNum = np.zeros((num, num), dtype=int)
    ThreadNum = np.zeros(num, dtype=int)  # Save the thread
    for i in range(num - 1):
        particlei = np.array([A[i, 0], A[i, 1], A[i, 2]])
        for j in range(i + 1, num):
            particlej = np.array([A[j, 0], A[j, 1], A[j, 2]])
            r = particlei - particlej
            dis_r = np.linalg.norm(r)
            diff = dis_r - threshold
            if diff <= 0:
                ThreadNum[i] += 1
                ThreadNum[j] += 1
                num_i = ThreadNum[i]
                num_j = ThreadNum[j]
                SerialNum[i, num_i - 1] = j + 1
                SerialNum[j, num_j - 1] = i + 1
            else:
                continue

    return SerialNum, ThreadNum

@njit(parallel=True)
def sum_dipole_field(A, num, m, ThreadNum, SerialNum, mu, kT):
    Hdd = np.zeros((num, 3))
    for i in range(num):
        particlei = np.array([A[i, 0], A[i, 1], A[i, 2]])
        ConPar_num = ThreadNum[i]
        H = np.zeros((3))
        for j in range(ConPar_num):
            k = SerialNum[i, j] - 1  # Adjusting for 1-based indexing
            particlej = np.array([A[k, 0], A[k, 1], A[k, 2]])  # Corrected index j -> k
            mj = m[k, :]
            B = dipole_field(particlei, particlej, mj, mu, kT)
            H += B

        Hdd[i, :] = H

    return Hdd

def CombinedNeelBrown(init_data):
    xi0 = init_data['unitless_energy']
    ut = init_data['normalized_brown_time_step']
    vt = init_data['normalized_neel_time_step']
    t0 = init_data['neel_event_time']
    tB = init_data['brown_relax_time']
    dt = init_data['time_step']
    lent = init_data['evaluation_time_length']
    sig = init_data['unitless_anisotropy']
    al = init_data['constant_damping']
    num = init_data['number_of_particles']
    A = init_data['particle_cordinates']
    mu = init_data['magnetic_moment']
    kT = init_data['Boltzman_energy']

    M = np.zeros((lent, 3))
    N = np.zeros((lent, 3))
    Hdipole = np.zeros((lent, 3))

    m = np.tile([1, 0, 0], (num, 1))
    n = m.copy()
    xI=np.tile([0, 0, xi0], (num, 1))

    serialNum, threadNum =contact_matrix(A, num, 500e-9)

    for j in range(lent):
        M[j, :] = np.mean(m, axis=0)
        N[j, :] = np.mean(n, axis=0)

        a = np.sum(m * n, axis=1)

        dn = sig*a[:, np.newaxis] * (m - a[:, np.newaxis]) * n * ut + np.cross(np.random.randn(num, 3), n) * np.sqrt(ut)
        n = n + dn
        n = n / np.linalg.norm(n, axis=1, keepdims=True)

        Hdd = sum_dipole_field(A, num, m, threadNum, serialNum, mu, kT)

        Hdipole[j, :] = np.mean(Hdd, axis=0)

        xi = xI * np.cos(2 * np.pi * j * dt) + 2*sig * a[:, np.newaxis] * n + Hdd  # total field over time

        h = np.random.randn(num, 3)
        f1 = np.cross(xi / al + np.cross(m, xi), m) / 2
        g1 = np.cross(h / al + np.cross(m, h), m)
        mb = m + f1 * vt + g1 * np.sqrt(vt)

        a2 = np.sum(mb * n, axis=1)

        xb = xI * np.cos(2 * np.pi * (j + 1) * dt) + 2*sig * a2[:, np.newaxis] * n + Hdd

        h2 = np.random.randn(num, 3)
        f2 = np.cross(xb / al + np.cross(mb, xb), mb) / 2
        g2 = np.cross(h2 / al + np.cross(mb, h2), mb)

        m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
        m = m / np.linalg.norm(m, axis=1, keepdims=True)

        print('\r', 'time step in samples: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))

    return M[:, -1], Hdipole

def peaksInit(He, dMk, dH, cycs, H_range=(-18e-3, 18e-3)):
    l = dMk.shape[0]
    khalf = int(l / (2 * cycs))
    Hl = He[-2*khalf: -khalf]
    dmdhl = dMk[-2*khalf: -khalf] / dH[-2*khalf: -khalf]
    Hr = He[-khalf:]
    dmdhr = dMk[-khalf:] / dH[-khalf:]

    maskl = np.where((Hl >= H_range[0]) & (Hl <= H_range[1]))[0]
    maskr = np.where((Hr >= H_range[0]) & (Hr <= H_range[1]))[0]

    Hlmask = Hl[maskl]
    dMdHlmask = dmdhl[maskl]
    Hrmask = Hr[maskr]
    dMdHrmask = dmdhr[maskr]

    return Hlmask, dMdHlmask , maskl, Hrmask, dMdHrmask, maskr

def peaks_analysis(HeMask, dmdhMask, mask):
    peaksIdx, _ = find_peaks(dmdhMask)
    srt = np.sort(dmdhMask[peaksIdx])
    max_peak = srt[-1]
    idxMax = np.where(dmdhMask[peaksIdx] == max_peak)[0]
    spline = UnivariateSpline(mask, dmdhMask - max_peak / 2, s=0)
    roots = spline.roots()
    idxr1 = np.where(mask == int(np.ceil(roots[0])))[0]
    idxr2 = np.where(mask == int(roots[0]))[0]
    idxr3 = np.where(mask == int(np.ceil(roots[1])))[0]
    idxr4 = np.where(mask == int(roots[1]))[0]
    v1 = (HeMask[idxr1] + HeMask[idxr2])[0] / 2
    v2 = (HeMask[idxr3] + HeMask[idxr4])[0] / 2
    fwhm = abs(v2 - v1)
    res = {
        'dmdH_peak': max_peak,
        'He_peak': HeMask[peaksIdx[idxMax]][0],
        'fwhm_left': v1,
        'fwhm_right': v2
    }
    return res