import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hamming

def calculate_and_plot_psd(data, fs, nfft_welch):
    """
    Calculate and plot the Power Spectral Density (PSD) using Welch's method.

    Parameters:
    data (numpy array): Input signal data.
    fs (float): Sampling frequency in Hz.
    nfft_welch (int): Number of FFT points for Welch's method.

    Returns:
    float: Mean PSD value.
    """
    # Ensure data length is sufficient for Welch's method
    if len(data) < nfft_welch:
        print("Insufficient data length for Welch's method")
        return None

    # Apply Hamming window
    window = hamming(len(data))
    data_windowed = data * window

    # Calculate PSD using Welch's method
    f, Pxx_den = welch(
        data_windowed, fs, nperseg=nfft_welch, return_onesided=False
    )

    # Shift frequency and PSD for negative frequencies
    Pxx_den_shifted_data = np.fft.fftshift(Pxx_den)
    f_shifted_data = np.fft.fftshift(f)

    # Calculate the mean PSD value
    mean_psd_value = np.mean(Pxx_den_shifted_data)
    print(f"Mean PSD Value: {mean_psd_value:.2e} V^2/Hz")

    # Plot the PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_shifted_data / 1e6, Pxx_den_shifted_data)  # Convert frequency to MHz
    plt.axhline(1.1 * Pxx_den_shifted_data.mean(), color='r', linestyle='--', label="Mean PSD")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.title("Power Spectral Density (PSD)")
    plt.legend()
    plt.show()

    return mean_psd_value

# Example usage
fs = 50e6  # Sampling frequency (50 MHz)
nfft_welch = 4096  # Number of FFT points for Welch's method

# Replace this with your actual data
signal_packet = first_packet  # Example data

mean_psd = calculate_and_plot_psd(signal_packet, fs, nfft_welch)
