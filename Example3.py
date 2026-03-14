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
# Generate AM-FM signal with known parameters
###############################################

Fs  = 22050
N   = 1000

t = st.time_axis(N, Fs)


# ----- True parameters -----

a     = 1.2
mu    = 10
psi   = 9000
phi   = st.modulo2pi(np.random.rand() * 2*np.pi)
omega = 2*np.pi*440

# signal
x = 2*np.real(a * np.exp(mu*t + 1j*(phi + omega*t + psi*t**2/2)))


#plt.plot(x)
#plt.show()


# add noise
snr_in = 30
s = st.sigmerge(x, np.random.randn(N), snr_in)

###############################################
# Estimation
###############################################

q_method = 3
a_method = 2
k = 2

(a_hat, mu_hat, phi_hat, omega_hat, psi_hat,
 delta_t_hat, delta_amp, m_idx, Xw, q) = rs.my_reassignment(
    s, Fs, k, q_method, a_method
)

print(m_idx)
print(a_hat)
print(mu_hat)
print(st.modulo2pi(phi_hat))
print(omega_hat)
print(psi_hat)
    
     
# exit()

     
###############################################
# Reconstructed signal
###############################################

x_hat = 2*np.real(a_hat * np.exp(mu_hat*t + 1j*(phi_hat + omega_hat*t + psi_hat*t**2/2)))


###############################################
# Print parameters
###############################################

print("\n===== TRUE PARAMETERS =====")
print(f"a      = {a}")
print(f"mu     = {mu}")
print(f"phi    = {phi}")
print(f"omega  = {omega/(2*np.pi):.2f} Hz")
print(f"psi    = {psi/(2*np.pi):.2f} Hz/s")

print("\n===== ESTIMATED PARAMETERS =====")

print(f"a_hat     = {a_hat}")
print(f"mu_hat    = {mu_hat}")
print(f"phi_hat   = {phi_hat}")
print(f"omega_hat = {omega_hat/(2*np.pi):.2f} Hz")
print(f"psi_hat   = {psi_hat/(2*np.pi):.2f} Hz/s")
print(f"delta_t   = {delta_t_hat}")
print(f"delta_amp = {delta_amp}")




rqf_val = st.rqf(x, x_hat)

print("\n===== RQF =====")
print("RQF =", rqf_val)

###############################################
# Plot signals
###############################################

plt.figure(figsize=(10,4))

plt.plot(t[:1000], s[:1000], label="noisy signal")
plt.plot(t[:1000], x_hat[:1000], 'k--', label="estimated signal")

plt.xlabel("time (s)")
plt.title("Signal reconstruction")
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
