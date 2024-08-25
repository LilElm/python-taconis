import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# Define the Lorentzian function
def lorentzian(x, x0, gamma):
    return (1/np.pi) * (gamma / ((x - x0)**2 + gamma**2))

# Parameters for the Lorentzian function
x0 = 80  # Center
gamma = 1  # Half-width at half-maximum (HWHM)
N = 10240  # Number of points
x = np.linspace(50, 100, N)
lorentz_signal = lorentzian(x, x0, gamma)


# Define a Sine wave
f = 100 #Hz
a = 0.01
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)

lorentz_signal = lorentz_signal + sine


# Define a Sine wave
f = 107 #Hz
a = 0.04
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)

lorentz_signal = lorentz_signal + sine

# Define a Sine wave
f = 27 #Hz
a = 0.01
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)

lorentz_signal = lorentz_signal + sine




# Perform the Fourier transform
lorentz_fft = fft(lorentz_signal)
freqs = fftfreq(N, x[1] - x[0])
lorentz_fft_shifted = fftshift(lorentz_fft)
freqs_shifted = fftshift(freqs)

# Plot the original Lorentzian signal
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, lorentz_signal)
plt.title('Lorentzian Signal')
plt.xlabel('x')
plt.ylabel('Amplitude')

# Plot the Fourier transform
plt.subplot(1, 2, 2)
plt.plot(freqs_shifted, np.abs(lorentz_fft_shifted))
plt.title('Fourier Transform of Lorentzian Signal')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid()
plt.show()