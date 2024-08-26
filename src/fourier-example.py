import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import butter, sosfiltfilt

# Define the Lorentzian function
def lorentzian(x, x0, gamma):
    return (1/np.pi) * (gamma / ((x - x0)**2 + gamma**2))

# Parameters for the Lorentzian function
x0 = 80  # Center
gamma = 1  # Half-width at half-maximum (HWHM)
N = 10240  # Number of points
x = np.linspace(50, 100, N)
lorentz_signal = lorentzian(x, x0, gamma)


# Add a Sine wave
f = 100 #Hz
a = 0.01
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)
lorentz_signal = lorentz_signal + sine


# Add another Sine wave
f = 107 #Hz
a = 0.04
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)
lorentz_signal = lorentz_signal + sine

# Add yet another Sine wave
f = 27 #Hz
a = 0.01
phi = 0
sine = a * np.sin(2 * np.pi * f * x + phi)
lorentz_signal = lorentz_signal + sine




# Perform the Fourier transform
x_fft = fftfreq(N, x[1] - x[0])
y_fft = fft(lorentz_signal)

x_fft_shifted = fftshift(x_fft)
y_fft_shifted = fftshift(y_fft)

# Implement a low-pass filter
cutoff = 26 # Hz
y_fft_filtered = np.where(np.abs(x_fft) < cutoff, y_fft, 0)
filtered_signal = ifft(y_fft_filtered)



# Plot the original Lorentzian signal
fig = plt.figure()#figsize=(12, 6))
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(x, lorentz_signal)
#ax.title('Lorentzian Signal')
#ax.xlabel('x')
#ax.ylabel('Amplitude')

# Plot the Fourier transform
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(x_fft_shifted, np.abs(y_fft_shifted))
ax2.set_xbound(0,)


# Plot the filtered signal
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(x, filtered_signal)
plt.show()


















