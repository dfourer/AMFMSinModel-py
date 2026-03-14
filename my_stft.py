#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
Module:      AM/FM Signal Processing - Local Parameter Estimation
Author:      Dominique Fourer
Email:       dominique@fourer.fr
Reference:   Dominique Fourer, François Auger, Geoffroy Peeters,
             "Local AM/FM parameters estimation: application
             to sinusoidal modeling and blind audio source separation,"
             IEEE Signal Processing Letters, Vol. 25, Issue 10,
             pp. 1600-1604, Oct. 2018, DOI: 10.1109/LSP.2018.2867799
===========================================================

Description:
------------
This Python module implements tools for local AM/FM parameter estimation,
including functions to generate AM/FM signals, compute STFT, and perform
sine parameter reassignment. The implementation is inspired by the
methodology described in the referenced IEEE Signal Processing Letters paper.

Dependencies:
-------------
- numpy
- scipy
- matplotlib
===========================================================
"""

import numpy as np
from scipy.signal import resample
import reassignment as rs

import matplotlib.pyplot as plt


# --- ajout de bruit (SNR) ---
def sigmerge(sig, noise, snr_db):
    """
    Mélange signal + bruit pour obtenir SNR en dB
    """
    sig_power = np.mean(np.abs(sig)**2)
    noise_power = np.mean(np.abs(noise)**2)
    k = np.sqrt(sig_power / (10**(snr_db/10) * noise_power))
    return sig + k*noise

# ------------------------------------------------------------
# Peak detection (simple version)
# ------------------------------------------------------------

def peak_detect(x, T):
    """
    Detect peaks above threshold T
    """
    x = np.asarray(x)
    peaks = []

    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > T:
            peaks.append(i)

    return np.array(peaks, dtype=int)

# ------------------------------------------------------------
# Time axis
# ------------------------------------------------------------

def reduced_time_axis(N):
    if N % 2 == 1:          # N impair
        H = (N - 1) // 2
        rt = np.arange(-H, H+1)
    else:                   # N pair
        H = N//2 - 1
        rt = np.arange(-H-1, H+1)

    return rt

def time_axis(N, Fs):
    return reduced_time_axis(N) / Fs

# ------------------------------------------------------------
# Modulopi
# ------------------------------------------------------------


def modulo2pi(M, v=2*np.pi):

    M = np.atleast_1d(np.array(M, dtype=float))

    I1 = np.where(M < -v/2)[0]
    I2 = np.where(M >  v/2)[0]

    while I1.size > 0 or I2.size > 0:

        M[I1] += v
        M[I2] -= v

        I1 = np.where(M < -v/2)[0]
        I2 = np.where(M >  v/2)[0]

    return M


# ------------------------------------------------------------
# Hann window (periodic version)
# ------------------------------------------------------------

def hann_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2*np.pi*n/N))


# ------------------------------------------------------------
# STFT + modulation estimation
# ------------------------------------------------------------
#
# Input :
# s: signal
# w: analysis windows (Hann by default N=2048)
# rec : overlap ratio (default: 2 <=> 50%)
# Fs: sampling frequency in Hz
# q_method
#
def my_stft(s, w=None, rec=2,
        Fs=22100, q_method=0, a_method=1,
        T=1e-5, p_method=1, k=2):

    """
    STFT with modulation estimation using my_reassignment
    """

    if w is None:
        w = hann_window(2048)

    w = np.asarray(w)
    s = np.asarray(s)

    N = len(w)
    sig_len = len(s)

    step = round(N / rec)
    nb_trame = int(np.floor(sig_len / step))

    Sw = np.zeros((N, nb_trame), dtype=complex)
    mod_tfr = np.zeros((N, nb_trame), dtype=complex)
    mod_tfr_ref = np.zeros((N, nb_trame), dtype=complex)

    wref = w.copy()
    sum_w = np.sum(w)

    for i_t in range(nb_trame):

        i0 = i_t * step
        i1 = min(sig_len, i0 + N)

        trame = s[i0:i1]
        N_tmp = len(trame)

        if N_tmp < N:
            w = resample(wref, N_tmp)
            sum_w = np.sum(w)
        else:
            w = wref

        #Sw[0:N_tmp, i_t] = np.fft.fft(trame * w)
        Sw[0:N, i_t] = np.fft.fft(trame * w,N)

        if q_method > 0:

            spectrum = np.abs(Sw[0:N_tmp, i_t])

            if p_method == 1:
                m = peak_detect(spectrum, T)
            else:
                m = np.where((spectrum / sum_w) > T)[0]

            if len(m) > 1:
                (a_hat, mu_hat, phi_hat, omega_hat, psi_hat, delta_t_hat, delta_amp,
                     m_idx, Xw, q) = rs.my_reassignment(trame, Fs, k, q_method, a_method, np.arange(N_tmp) )
                
                #else:
                #    raise NotImplementedError("Phase vocoder version not implemented")

                mod_tfr_ref[m, i_t] = (
                    mu_hat[m] + 1j * psi_hat[m]
                )

                mod_tfr[m, i_t] = (
                    np.real(mu_hat[m] / np.log(np.finfo(float).eps + 2 * a_hat[m]))
                    + 1j * psi_hat[m] / (np.finfo(float).eps + omega_hat[m])
                )

    return Sw, mod_tfr, mod_tfr_ref

def SNR(x_ref, x_est):
    p_signal = np.mean(np.abs(x_ref)**2)
    p_error  = np.mean(np.abs(x_ref - x_est)**2)
    return 10*np.log10(p_signal / p_error)


# --- reconstruction STFT simple ---
def my_inv_stft(Sx, w, rec):
    """
    Version simple reconstruction iSTFT
    """
    N, n_frames = Sx.shape
    step = len(w) // rec
    x_rec = np.zeros(N + (n_frames-1)*step)
    win_sum = np.zeros_like(x_rec)
    for i in range(n_frames):
        start = i*step
        frame = np.fft.ifft(Sx[:,i]).real * w
        x_rec[start:start+len(w)] += frame
        win_sum[start:start+len(w)] += w
    # éviter division par zéro
    nonzero = win_sum > 1e-12
    x_rec[nonzero] /= win_sum[nonzero]
    return x_rec


###############################################
# Relative Quadratic Error (RQF)
###############################################
def rqf(x, x_hat):
    eps = 1e-12
    return 20 * np.log10(
        (np.linalg.norm(x) + eps) /
        (np.linalg.norm(x - x_hat) + eps)
    )


def plot_spectrogram(Sw, Nh=None, t_axis=None, f_axis=None, cmap="magma"):
    """
    Display a spectrogram from a STFT matrix.

    Parameters
    ----------
    Sw : ndarray
        STFT matrix (freq x time)
    Nh : int, optional
        number of frequency bins to display (default = half spectrum)
    t_axis : ndarray, optional
        time axis
    f_axis : ndarray, optional
        frequency axis
    cmap : str
        matplotlib colormap
    """

    if Nh is None:
        Nh = Sw.shape[0] // 2

    S = 20*np.log10(np.abs(Sw[:Nh, :]) + 1e-12)

    plt.figure(figsize=(10,4))

    # axis extent if provided
    if t_axis is not None and f_axis is not None:
        extent = [t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]]
        plt.imshow(S, aspect='auto', origin='lower', extent=extent, cmap=cmap)
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")
    else:
        plt.imshow(S, aspect='auto', origin='lower', cmap=cmap)
        plt.xlabel("frame index")
        plt.ylabel("frequency bin")

    plt.title("Spectrogram")
    plt.colorbar(label="dB")
    plt.tight_layout()