import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hamming
from scipy.integrate import simps

def calculate_and_plot_psd_with_snr(data, fs, nfft_welch, signal_band, noise_band_neg, noise_band_pos):
    """
    Calculate and plot the Power Spectral Density (PSD) and compute the SNR.

    Parameters:
    data (numpy array): Input signal data.
    fs (float): Sampling frequency in Hz.
    nfft_welch (int): Number of FFT points for Welch's method.
    signal_band (tuple): Frequency range of the signal (e.g., (f_min, f_max) in Hz).
    noise_band_neg (tuple): Frequency range of the negative frequencies for noise power calculation.
    noise_band_pos (tuple): Frequency range of the positive frequencies for noise power calculation.

    Returns:
    float: Mean PSD value.
    float: SNR in dB.
    """
    if len(data) < nfft_welch:
        print("Insufficient data length for Welch's method")
        return None, None

    # Apply Hamming window
    window = hamming(len(data))
    data_windowed = data * window

    # Calculate PSD using Welch's method
    f, Pxx_den = welch(data_windowed, fs, nperseg=nfft_welch, return_onesided=False)

    # Shift frequency and PSD for negative frequencies
    Pxx_den_shifted = np.fft.fftshift(Pxx_den)
    f_shifted = np.fft.fftshift(f)

    # Calculate the mean PSD value
    mean_psd_value = np.mean(Pxx_den_shifted)
    print(f"Mean PSD Value: {mean_psd_value:.2e} V^2/Hz")

    # Calculate signal power
    signal_indices = (f_shifted >= signal_band[0]) & (f_shifted <= signal_band[1])
    signal_power = simps(Pxx_den_shifted[signal_indices], f_shifted[signal_indices])

    # Calculate noise power for negative frequency band
    noise_indices_neg = (f_shifted >= noise_band_neg[0]) & (f_shifted <= noise_band_neg[1])
    noise_power_neg = simps(Pxx_den_shifted[noise_indices_neg], f_shifted[noise_indices_neg])

    # Calculate noise power for positive frequency band
    noise_indices_pos = (f_shifted >= noise_band_pos[0]) & (f_shifted <= noise_band_pos[1])
    noise_power_pos = simps(Pxx_den_shifted[noise_indices_pos], f_shifted[noise_indices_pos])

    # Calculate total noise power by summing both negative and positive noise powers
    total_noise_power = noise_power_neg + noise_power_pos

    # Compute SNR
    snr_db = 10 * np.log10(signal_power / total_noise_power)
    print(f"SNR: {snr_db:.2f} dB")

    # Plot the PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_shifted / 1e6, Pxx_den_shifted, label="PSD")  # Convert frequency to MHz
    plt.axvspan(signal_band[0] / 1e6, signal_band[1] / 1e6, color='green', alpha=0.3, label="Signal Band")
    plt.axvspan(noise_band_neg[0] / 1e6, noise_band_neg[1] / 1e6, color='red', alpha=0.3, label="Noise Band (Negative)")
    plt.axvspan(noise_band_pos[0] / 1e6, noise_band_pos[1] / 1e6, color='red', alpha=0.3, label="Noise Band (Positive)")
    plt.axhline(1.1 * mean_psd_value, color='r', linestyle='--', label="Mean PSD")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.title("Power Spectral Density (PSD)")
    plt.legend()
    plt.show()

    return mean_psd_value, snr_db

# Example usage
fs = 50e6  # Sampling frequency (50 MHz)
nfft_welch = 4096  # Number of FFT points for Welch's method
signal_band = (-4e6, 4e6)  # Example signal band in Hz

# Define the negative and positive noise bands separately
noise_band_neg = (-25e6, -4e6)  # Noise band for negative frequencies
noise_band_pos = (4e6, 25e6)    # Noise band for positive frequencies

# Assuming first_packet is the signal data you want to analyze
mean_psd, snr_db = calculate_and_plot_psd_with_snr(first_packet, fs, nfft_welch, signal_band, noise_band_neg, noise_band_pos)
