# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import time

# Read audio

# noise:
fs1, d = read('lecture3_rls/data/Noise.wav')
d = d/max(d)
# Signal
fs1, x = read('lecture3_rls/data/Music.wav')
x = x/max(x)
# Noisy signal
Fs, u = read('lecture3_rls/data/Noisy_Music.wav')
u_max = max(u)
u = u/u_max

N = len(u)
# Filter Order:
M = 12

# *************************************************************************
# *************************************************************************
# *************************************************************************
# LMS Implementation
t0 = time.time()
mu = 0.01
w = np.random.randn(M)
padded_u = np.hstack((np.zeros(M-1), u))
y = np.zeros(N)
for n in range(0, N):
    u_vect = padded_u[n:n+M]
    e = d[n] - np.matmul(w, u_vect)
    w = w + mu*e*u_vect
    y[n] = np.matmul(w, u_vect)

# y2 = [zeros(sampleDelay,1) ; y] ;
filtered_signal_LMS = u - y
write("filtered_signal_LMS.wav", Fs, filtered_signal_LMS)
# Below is 16 bit version
#write(f"filtered_signal_LMS_16b.wav", Fs, (filtered_signal_LMS*u_max).astype(np.int16))
t1 = time.time()


# *************************************************************************

# NLMS
t2 = time.time()
mu = 1
w = np.random.randn(M)
padded_u = np.hstack((np.zeros(M-1), u))
y = np.zeros(N)
Eps = 0.0001
for n in range(0, N):
    u_vect = padded_u[n:n+M]
    mu1 = mu/(Eps + pow(np.linalg.norm(u_vect, ord=2), 2))
    e = d[n] - np.matmul(w, u_vect)
    w = w + mu1*e*u_vect
    y[n] = np.matmul(w, u_vect)

filtered_signal_NLMS = u - y
write("filtered_signal_NLMS.wav", Fs, filtered_signal_NLMS)
# Below is 16 bit version
# write(f"filtered_signal_NLMS_16b.wav", Fs, (filtered_signal_NLMS*u_max).astype(np.int16))
t3 = time.time()


# *************************************************************************

# RLS:
t4 = time.time()
lmd = 1 - 1/(0.1*M)
delta = 0.01
P = 1/delta*np.identity(M)
w = np.random.randn(M)
padded_u = np.hstack((np.sqrt(delta)*np.random.randn(M-1), u))
y = np.zeros(N)
for n in range(0, N):
    u_vect = padded_u[n:n+M]
    PI = np.matmul(P, u_vect)
    gain_k = PI/(lmd + np.matmul(u_vect, PI))
    prior_error = d[n] - np.matmul(w, u_vect)
    w = w + prior_error*gain_k
    P = P/lmd - np.outer(gain_k, u_vect@P)/lmd
    y[n] = np.matmul(w, u_vect)

filtered_signal_RLS = u - y
write("filtered_signal_RLS.wav", Fs, filtered_signal_RLS)
# Below is 16 bit version
# write(f"filtered_signal_RLS_16b.wav", Fs, (filtered_signal_RLS*u_max).astype(np.int16))
t5 = time.time()

print(f"LMS took  {t1-t0:>.3f}")
print(f"NLMS took {t3-t2:>.3f}")
print(f"RLS took  {t5-t4:>.3f}")
