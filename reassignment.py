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
import my_stft as st
from scipy.signal import find_peaks


def my_hann_window(N, order=0):
    """
    Hann window and derivatives approximation.
    order=0 : window
    order=1 : first derivative
    order=2 : second derivative
    """
    n = np.arange(N)
    w = 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))

    if order == 0:
        return w
    elif order == 1:
        return np.gradient(w)
    elif order == 2:
        return np.gradient(np.gradient(w))
    else:
        out = w.copy()
        for _ in range(order):
            out = np.gradient(out)
        return out


def zerophase_signal(x):
    """Equivalent of MATLAB zero-phase alignment."""
    N = len(x)
    return np.roll(x, -N//2)


def Gamma(N, Fs, Delta, mu, q, w):
    L = np.size(Delta)
    Mw = np.tile(w, (L,1))
    t = st.time_axis(N, Fs)

    result = np.sum(Mw * np.exp(
        mu[:,None]*t + 1j*Delta[:,None]*t + q[:,None]*(t**2)/2
    ), axis=1)

    idx = np.isnan(result) | np.isinf(result)
    result[idx] = 1

    return result


def my_reassignment(x, Fs=1, k=2, q_method=2, a_method=2, m=None):

    x = np.asarray(x).flatten()
    N = len(x)

    t = st.time_axis(N, Fs)

    w  = my_hann_window(N)
    wd = Fs * my_hann_window(N,1)
    tw = t * w

    if q_method == 0:
        wdd = Fs**2 * my_hann_window(N,2)
        twd = t * wd

        Xwdd = np.fft.fft(zerophase_signal(wdd*x))
        Xtwd = np.fft.fft(zerophase_signal(twd*x))

    elif q_method in [1,2]:

        wdk   = Fs**k     * my_hann_window(N,k)
        wdkm1 = Fs**(k-1) * my_hann_window(N,k-1)

        twdkm1 = t * wdkm1

        Xwdk    = np.fft.fft(zerophase_signal(wdk*x))
        Xwdkm1  = np.fft.fft(zerophase_signal(wdkm1*x))
        Xtwdkm1 = np.fft.fft(zerophase_signal(twdkm1*x))

    elif q_method == 3:

        tkm1wd = t**(k-1) * wd
        tkw    = t**k     * w
        tkm1w  = t**(k-1) * w
        tkm2w  = t**(k-2) * w

        Xtkm1wd = np.fft.fft(zerophase_signal(tkm1wd*x))
        Xtkw    = np.fft.fft(zerophase_signal(tkw*x))
        Xtkm1w  = np.fft.fft(zerophase_signal(tkm1w*x))
        Xtkm2w  = np.fft.fft(zerophase_signal(tkm2w*x))

    elif q_method == 4:

        t2w = t**2 * w
        t3w = t**3 * w
        twd = t * wd
        t2wd = t**2 * wd
        wd2 = Fs**2 * my_hann_window(N,2)

        Xt2w  = np.fft.fft(zerophase_signal(t2w*x))
        Xt3w  = np.fft.fft(zerophase_signal(t3w*x))
        Xtwd  = np.fft.fft(zerophase_signal(twd*x))
        Xt2wd = np.fft.fft(zerophase_signal(t2wd*x))
        Xwd2  = np.fft.fft(zerophase_signal(wd2*x))


    Xw  = np.fft.fft(zerophase_signal(w*x))
    Xwd = np.fft.fft(zerophase_signal(wd*x))
    Xtw = np.fft.fft(zerophase_signal(tw*x))


    if m is None:
        m = np.argmax(np.abs(Xw)**2)

    base_omega = (m) * 2*np.pi*Fs/N

    delta_omega_tilde = -Xwd[m] / Xw[m]
    omega_tilde = 1j*base_omega + delta_omega_tilde

    delta_t_tilde = Xtw[m] / Xw[m]
    delta_t = np.real(delta_t_tilde)


    if q_method == 0:

        q = 1j * (
            np.imag(Xwdd[m]/Xw[m]) - np.imag((Xwd[m]/Xw[m])**2)
        ) / (
            np.real((Xtw[m]*Xwd[m])/(Xw[m]**2)) - np.real(Xtwd[m]/Xw[m])
        )

    elif q_method == 1:

        q = 1j * (
            np.real(Xwdk[m]*np.conj(Xwdkm1[m])) /
            np.imag(Xtwdkm1[m]*np.conj(Xwdkm1[m]))
        )

    elif q_method == 2:

        q = (
            Xwdk[m]*Xw[m] - Xwdkm1[m]*Xwd[m]
        ) / (
            Xwdkm1[m]*Xtw[m] - Xtwdkm1[m]*Xw[m]
        )

    elif q_method == 3:

        q = (
            (Xtkm1wd[m] + (k-1)*Xtkm2w[m])*Xw[m] -
            Xtkm1w[m]*Xwd[m]
        ) / (
            Xtkm1w[m]*Xtw[m] - Xtkw[m]*Xw[m]
        )

    elif q_method == 4:

        A = np.array([
            [Xt2w[m], -Xtw[m], Xw[m]],
            [Xt2wd[m], -Xtwd[m], Xwd[m]],
            [Xt3w[m], -Xt2w[m], Xtw[m]]
        ])

        if abs(np.linalg.det(A)) > np.finfo(float).eps:

            Ainv = np.linalg.pinv(A)
            A2 = np.array([Xwd[m], Xwd2[m], Xtwd[m]+Xw[m]])

            rx = Ainv[0,:] @ A2
            q = Ainv[1,:] @ A2 - 2*rx*delta_t_tilde

        else:
            q = 0


    psi = np.imag(q)

    if a_method == 1:
        mu = -np.real(Xwd[m]/Xw[m])
    else:
        mu = np.real(omega_tilde - q*delta_t_tilde)


    omega = np.imag(omega_tilde - q*delta_t_tilde)

    delta_omega = omega - base_omega


    p = Gamma(
        N,
        Fs,
        np.array([delta_omega]),
        np.array([mu]),
        np.array([1j*np.imag(q)]),
        w
    )

    phi = np.angle(Xw[m]/p)
    a = np.abs(Xw[m]/p)

    delta_amp = -np.imag(Xtw[m]/Xw[m])

    return a, mu, phi, omega, psi, delta_t, delta_amp, m, Xw, q



def my_reassignment_multi(x, Fs=1, k=2, q_method=2, a_method=2, threshold=0.1):
    """
    Multi-peak reassignment: estimate AM-FM parameters at all spectral peaks.

    Parameters
    ----------
    x : ndarray
        Input signal (1D)
    Fs : float
        Sampling frequency
    k : int
        Modulation estimation order
    q_method : int
        Frequency modulation estimation method
    a_method : int
        Amplitude method
    threshold : float
        Minimum normalized amplitude to keep a peak

    Returns
    -------
    params : list of dicts
        Each dict contains parameters for one spectral peak:
        a, mu, phi, omega, psi, delta_t, delta_amp, m_idx
    """
    x = np.asarray(x).flatten()
    N = len(x)

    t = st.time_axis(N, Fs)

    w  = my_hann_window(N)
    wd = Fs * my_hann_window(N,1)
    tw = t * w

    if q_method == 0:
        wdd = Fs**2 * my_hann_window(N,2)
        twd = t * wd

        Xwdd = np.fft.fft(zerophase_signal(wdd*x))
        Xtwd = np.fft.fft(zerophase_signal(twd*x))

    elif q_method in [1,2]:

        wdk   = Fs**k     * my_hann_window(N,k)
        wdkm1 = Fs**(k-1) * my_hann_window(N,k-1)

        twdkm1 = t * wdkm1

        Xwdk    = np.fft.fft(zerophase_signal(wdk*x))
        Xwdkm1  = np.fft.fft(zerophase_signal(wdkm1*x))
        Xtwdkm1 = np.fft.fft(zerophase_signal(twdkm1*x))

    elif q_method == 3:

        tkm1wd = t**(k-1) * wd
        tkw    = t**k     * w
        tkm1w  = t**(k-1) * w
        tkm2w  = t**(k-2) * w

        Xtkm1wd = np.fft.fft(zerophase_signal(tkm1wd*x))
        Xtkw    = np.fft.fft(zerophase_signal(tkw*x))
        Xtkm1w  = np.fft.fft(zerophase_signal(tkm1w*x))
        Xtkm2w  = np.fft.fft(zerophase_signal(tkm2w*x))

    elif q_method == 4:

        t2w = t**2 * w
        t3w = t**3 * w
        twd = t * wd
        t2wd = t**2 * wd
        wd2 = Fs**2 * my_hann_window(N,2)

        Xt2w  = np.fft.fft(zerophase_signal(t2w*x))
        Xt3w  = np.fft.fft(zerophase_signal(t3w*x))
        Xtwd  = np.fft.fft(zerophase_signal(twd*x))
        Xt2wd = np.fft.fft(zerophase_signal(t2wd*x))
        Xwd2  = np.fft.fft(zerophase_signal(wd2*x))


    Xw  = np.fft.fft(zerophase_signal(w*x))
    Xwd = np.fft.fft(zerophase_signal(wd*x))
    Xtw = np.fft.fft(zerophase_signal(tw*x))
    
    
    
    # --- Detect peaks in the magnitude spectrum ---
    mag = np.abs(Xw)
    mag_norm = mag / np.max(mag)
    peaks, _ = find_peaks(mag_norm, height=threshold)

    params = []

    # import matplotlib.pyplot as plt 
    # plt.plot(mag_norm)
    # plt.plot(np.ones(len(mag_norm))*threshold, 'r-.')
    # plt.show()
    

    for m in peaks:
        
        base_omega = (m) * 2*np.pi*Fs/N

        delta_omega_tilde = -Xwd[m] / Xw[m]
        omega_tilde = 1j*base_omega + delta_omega_tilde

        delta_t_tilde = Xtw[m] / Xw[m]
        delta_t = np.real(delta_t_tilde)


        if q_method == 0:

            q = 1j * (
                np.imag(Xwdd[m]/Xw[m]) - np.imag((Xwd[m]/Xw[m])**2)
            ) / (
                np.real((Xtw[m]*Xwd[m])/(Xw[m]**2)) - np.real(Xtwd[m]/Xw[m])
            )

        elif q_method == 1:

            q = 1j * (
                np.real(Xwdk[m]*np.conj(Xwdkm1[m])) /
                np.imag(Xtwdkm1[m]*np.conj(Xwdkm1[m]))
            )

        elif q_method == 2:

            q = (
                Xwdk[m]*Xw[m] - Xwdkm1[m]*Xwd[m]
            ) / (
                Xwdkm1[m]*Xtw[m] - Xtwdkm1[m]*Xw[m]
            )

        elif q_method == 3:

            q = (
                (Xtkm1wd[m] + (k-1)*Xtkm2w[m])*Xw[m] -
                Xtkm1w[m]*Xwd[m]
            ) / (
                Xtkm1w[m]*Xtw[m] - Xtkw[m]*Xw[m]
            )

        elif q_method == 4:

            A = np.array([
                [Xt2w[m], -Xtw[m], Xw[m]],
                [Xt2wd[m], -Xtwd[m], Xwd[m]],
                [Xt3w[m], -Xt2w[m], Xtw[m]]
            ])

            if abs(np.linalg.det(A)) > np.finfo(float).eps:

                Ainv = np.linalg.pinv(A)
                A2 = np.array([Xwd[m], Xwd2[m], Xtwd[m]+Xw[m]])

                rx = Ainv[0,:] @ A2
                q = Ainv[1,:] @ A2 - 2*rx*delta_t_tilde

            else:
                q = 0


        psi = np.imag(q)

        if a_method == 1:
            mu = -np.real(Xwd[m]/Xw[m])
        else:
            mu = np.real(omega_tilde - q*delta_t_tilde)

        omega = np.imag(omega_tilde - q*delta_t_tilde)

        delta_omega = omega - base_omega


        p = Gamma(
            N,
            Fs,
            np.array([delta_omega]),
            np.array([mu]),
            np.array([1j*np.imag(q)]),
            w
        )

        phi = np.angle(Xw[m]/p)
        a = np.abs(Xw[m]/p)

        delta_amp = -np.imag(Xtw[m]/Xw[m])
        

        # Store parameters in dict
        params.append({
            "a": a,
            "mu": mu,
            "phi": phi,
            "omega": omega,
            "psi": psi,
            "delta_t": delta_t,
            "delta_amp": delta_amp,
            "m_idx": m
        })

    return params
