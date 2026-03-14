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

Real-world audio signal estimation and frame-by-frame reconstruction

Steps:
1. Estimate the signal parameters from the noisy observation using
   a reassignment-based estimation method.

2. Reconstruct the signal from the estimated parameters.

3. Compare the reconstructed signal with the original one using
   error metric (RQF) and visualize the results.

The goal is to validate the accuracy of the parameter estimation
algorithm under noisy conditions.
"""

import soundfile as sf

import matplotlib.pyplot as plt
import reassignment as rs
import my_stft as st
import numpy as np
from scipy.io.wavfile import write

dur = 10000

audiofile = "./leon_1_01.wav"
x, Fs = sf.read(audiofile)

L = min(dur * Fs, len(x))
x = x[0:L,1]

#plt.plot(x)
#plt.show()

N  = 1024
Nh = N//2


## analysis parameters
rec  = 2
step = N // rec
nb_frames = (L - N) // step



q_method = 3   ##FM estimation method
a_method = 1   ##AM estimation method
k = 2

x_hat = np.zeros(L)

###############################################
# Frame-by-frame processing
###############################################

for i in range(nb_frames):

    i0 = i * step
    i1 = i0 + N

    frame = x[i0:i1]

    t_frame = st.time_axis(N, Fs)

    # (a_hat, mu_hat, phi_hat, omega_hat, psi_hat,
    #  delta_t_hat, delta_amp, m_idx, Xw, q) = rs.my_reassignment(
    #     frame, Fs, k, q_method, a_method, np.arange(0,Nh)
    # )


    # # reconstruct frame
    # frame_hat = 2*np.real(
    #     a_hat * np.exp(mu_hat*t_frame + 1j*(phi_hat + omega_hat*t_frame + psi_hat*t_frame**2/2))
    # )


    params_list = rs.my_reassignment_multi(frame, Fs, k, q_method, a_method, threshold=0.18)

    # --- Initialize frame reconstruction ---
    frame_hat = np.zeros(N)
    
    #print("Processing frame",i,"/",nb_frames,"  -  ", len(params_list))
    print(f"Processing frame {i+1}/{nb_frames} - {len(params_list)} peaks detected")
    
    # --- Synthesize each component ---
    for p in params_list:
        a_hat   = p["a"]
        mu_hat  = p["mu"]
        phi_hat = p["phi"]
        omega_hat = p["omega"]
        psi_hat   = p["psi"]
        

        # reconstruct sinusoidal component
        frame_hat += 2*np.real(
            a_hat * np.exp(mu_hat*t_frame + 1j*(phi_hat + omega_hat*t_frame + psi_hat*t_frame**2/2))
        )

    # --- Overlap-add reconstruction ---
    #plt.plot(frame_hat, 'r-.')
    #plt.plot(frame)
    #plt.show(block=False)  # ne bloque pas l'exécution
    #plt.pause(0.2)          # pause 0.5 seconde
    #plt.clf()               # efface la figure pour la prochaine frame
    x_hat[i0:i1] += frame_hat * st.hann_window(N)  # on pondère avec la fenêtre
    #x_hat[i0:i1] += frame_hat


###############################################
# Reconstruction quality
###############################################

rqf_val = st.rqf(x[:len(x_hat)], x_hat)

print("\n===== RQF =====")
print("RQF =", rqf_val)


# Save to wav
print("\n===== Saving result to x_hat.wav =====")
x_int16 = np.int16(x_hat / np.max(np.abs(x_hat)) * 32767)
write("x_hat.wav", Fs, x_int16)



###############################################
# Plot signals
###############################################

# plt.figure(figsize=(10,4))

# plt.plot(t[:4000], s[:4000], label="noisy signal")
# plt.plot(t[:4000], x_hat[:4000], 'k--', label="reconstructed")

# plt.xlabel("time (s)")
# plt.title("Frame-by-frame reconstruction")
# plt.legend()
# plt.grid()


###############################################
# Spectrogram
###############################################

Sw,z,z = st.my_stft(x_hat, st.hann_window(N), rec=8)

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
plt.title("Spectrogram of the reconstructed signal")
plt.colorbar(label="dB")

plt.show()