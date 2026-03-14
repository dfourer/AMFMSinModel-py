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

Signal parameter estimation and reconstruction example.

This script demonstrates the complete workflow for testing a sinusoidal
AM–FM parameter estimation algorithm.

Long signal AM-FM parameter estimation and frame-by-frame reconstruction

Steps:
1. Generate a synthetic signal with known parameters:
   - amplitude (a)
   - amplitude modulation rate (mu)
   - initial phase (phi)
   - angular frequency (omega)
   - frequency modulation rate (psi)

2. Add white Gaussian noise to the signal in order to obtain a desired
   input Signal-to-Noise Ratio (SNR).

3. Estimate the signal parameters from the noisy observation using
   a reassignment-based estimation method.

4. Reconstruct the signal from the estimated parameters.

5. Compare the reconstructed signal with the original one using
   error metric (RQF) and visualize the results.

The goal is to validate the accuracy of the parameter estimation
algorithm under noisy conditions.
"""

import matplotlib.pyplot as plt
import numpy as np

import reassignment as rs
import my_stft as st




###############################################
# Parameters
###############################################

Fs  = 22050
dur = 10
N   = 2048

L = int(dur * Fs)

t = np.arange(L) / Fs


## analysis parameters
rec  = 1
step = N // rec
nb_frames = (L - N) // step

###############################################
# True AM-FM parameters
###############################################

a     = 1.2
mu    = 0.02
psi   = 5000
phi   = st.modulo2pi(np.random.rand() * 2*np.pi)
omega = 2*np.pi*440


###############################################
# Generate signal
###############################################

x = 2*np.real(a * np.exp(mu*t + 1j*(phi + omega*t + psi*t**2/2)))

# #plt.plot(x)
# #plt.show()
# rec=4

# Sw,z,z = st.my_stft(x, rec=rec);
# f_axis  = np.arange(0,N//2)/N*Fs
# nb_frames = Sw.shape[1]  # nombre de colonnes de Sw
# step = N//rec
# t_axis = np.arange(nb_frames) * step / Fs

# st.plot_spectrogram(Sw, Nh=None, t_axis=t_axis, f_axis=f_axis, cmap="magma")
# plt.show()
#exit()



###############################################
# Add noise
###############################################

snr_in = 30
s = st.sigmerge(x, np.random.randn(L), snr_in)

x_hat = np.zeros(L)



q_method = 3
a_method = 2
k = 2


###############################################
# Frame-by-frame processing
###############################################

for i in range(nb_frames):

    i0 = i * step
    i1 = i0 + N

    frame = s[i0:i1]

    t_frame = st.time_axis(N, Fs)

    (a_hat, mu_hat, phi_hat, omega_hat, psi_hat,
     delta_t_hat, delta_amp, m_idx, Xw, q) = rs.my_reassignment(
        frame, Fs, k, q_method, a_method
    )


    # reconstruct frame
    frame_hat = 2*np.real(
        a_hat * np.exp(mu_hat*t_frame + 1j*(phi_hat + omega_hat*t_frame + psi_hat*t_frame**2/2))
    )


    # overlap-add reconstruction
    x_hat[i0:i1] += frame_hat # * st.hann_window(N)


###############################################
# Reconstruction quality
###############################################

rqf_val = st.rqf(x[:len(x_hat)], x_hat)

print("\n===== RQF =====")
print("RQF =", rqf_val)


###############################################
# Plot signals
###############################################

plt.figure(figsize=(10,4))

plt.plot(t[:4000], s[:4000], label="noisy signal")
plt.plot(t[:4000], x_hat[:4000], 'k--', label="reconstructed")

plt.xlabel("time (s)")
plt.title("Frame-by-frame reconstruction")
plt.legend()
plt.grid()


###############################################
# Spectrogram
###############################################

Sw,z,z = st.my_stft(s, st.hann_window(N), rec=8)

Nh = N//2

plt.figure(figsize=(10,4))

plt.imshow(
    20*np.log10(np.abs(Sw[:Nh,:])+1e-12),
    aspect='auto',
    origin='lower',
    cmap='magma'
)

plt.xlabel("frame")
plt.ylabel("frequency bin")
plt.title("Spectrogram of the analyzed signal")
plt.colorbar(label="dB")

plt.show()