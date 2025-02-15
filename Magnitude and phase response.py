import numpy as np
import matplotlib.pyplot as plt

def plot_fft_spectrum(data, fs, N_fft):
    """
    Calculate and plot the magnitude and phase spectrum using FFT.

    Parameters:
    data (numpy array): Input signal data.
    fs (float): Sampling frequency in Hz.
    N_fft (int): FFT size.

    Returns:
    None
    """
    # Calculate FFT
    data_fft = np.fft.fftshift(np.fft.fft(data, N_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / fs))

    # Magnitude and Phase
    magnitude = np.abs(data_fft)
    phase = np.angle(data_fft)

    # Plot Magnitude Spectrum
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, magnitude, color="blue")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')

    # Plot Phase Spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freqs, phase, color="orange")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Spectrum')

    plt.tight_layout()
    plt.show()

# Example usage
fs = 50e6  # Example sampling rate in Hz
N_fft = 4096  # FFT size

# Replace this with your actual data
signal_packet = first_packet # Example data

plot_fft_spectrum(signal_packet, fs, N_fft)
