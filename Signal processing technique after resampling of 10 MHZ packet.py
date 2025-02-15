import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, hamming, resample
from scipy.stats import skew, kurtosis

# Function to resample data
def resample_data(data, original_rate, target_rate):
    resampled_data = resample(data, int(len(data) * target_rate / original_rate))
    return resampled_data

# Function to calculate and plot PSD
def calculate_and_plot_psd(data, fs, nfft_welch):
    window = hamming(len(data))
    data_windowed = data * window
    f, Pxx_den = welch(data_windowed, fs, nperseg=nfft_welch, return_onesided=False)
    f_shifted = np.fft.fftshift(f)
    Pxx_den_shifted = np.fft.fftshift(Pxx_den)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_shifted / 1e6, Pxx_den_shifted)
    plt.axhline(1.1 * Pxx_den_shifted.mean(), color='r', linestyle='--', label="Mean PSD")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.title("Power Spectral Density (PSD)")
    plt.legend()
    plt.show()
    return f_shifted, Pxx_den_shifted

# Function to plot FFT spectrum (Magnitude and Phase)
def plot_fft_spectrum(data, fs, N_fft):
    data_fft = np.fft.fftshift(np.fft.fft(data, N_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / fs))

    magnitude = np.abs(data_fft)
    phase = np.angle(data_fft)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, magnitude, color="blue")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')

    plt.subplot(2, 1, 2)
    plt.plot(freqs, phase, color="orange")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Spectrum')

    plt.tight_layout()
    plt.show()

# Function to calculate skewness using moment-based method
def moment_based_skew(distribution):
    magnitude = np.abs(distribution)
    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((magnitude - mean) / std) ** 3)
    return skewness

# Function to calculate kurtosis using moment-based method
def moment_based_kurtosis(distribution):
    magnitude = np.abs(distribution)
    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    kurt = (1 / n) * np.sum(((magnitude - mean) / std) ** 4) - 3
    return kurt

# Classification functions
def classify_skewness(skew_value):
    if skew_value > 0:
        return "Positive Skewness"
    elif skew_value < 0:
        return "Negative Skewness"
    else:
        return "Symmetrical Distribution"

def classify_kurtosis(kurt_value):
    if kurt_value > 3:
        return "Leptokurtic (Sharp peak)"
    elif kurt_value < 3:
        return "Platykurtic (Flat peak)"
    else:
        return "Mesokurtic (Normal peak)"

# Function to plot spectrogram
def plot_spectrogram(data, fs, vmin=-150):
    plt.figure(figsize=(10, 4))
    plt.specgram(data, Fs=fs, scale='dB', vmin=vmin)
    plt.title("Spectrogram", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.show()

# Main Workflow
if __name__ == "__main__":
    data=first_packet

    # Define parameters
    original_rate = 50e6  # Original sampling rate (50 MHz)
    target_rate = 8e6    # Target sampling rate (16 MHz)
    nfft_welch = 256     # FFT size for Welch's method

    # Resample the data
    resampled_data = resample_data(data, original_rate, target_rate)

    # Plot spectrogram of resampled data
    plot_spectrogram(resampled_data, target_rate)

    # Calculate and plot PSD
    f_shifted, Pxx_den_shifted = calculate_and_plot_psd(resampled_data, target_rate, nfft_welch)
    
    # Plot FFT spectrum of resampled data
    plot_fft_spectrum(resampled_data, target_rate, N_fft=2048)

    # Calculate skewness and kurtosis
    skew_value = moment_based_skew(resampled_data)
    kurtosis_value = moment_based_kurtosis(resampled_data)
    skew_classification = classify_skewness(skew_value)
    kurtosis_classification = classify_kurtosis(kurtosis_value)

    # Print skewness and kurtosis results
    print(f"Skewness: {skew_value:.4f} -> {skew_classification}")
    print(f"Kurtosis: {kurtosis_value:.4f} -> {kurtosis_classification}\n")

    # KDE plot for resampled data
    plt.figure(figsize=(10, 4))
    sns.kdeplot(np.abs(resampled_data), shade=True)
    plt.title("KDE Plot of Resampled Data")
    plt.xlabel("Amplitude")
    plt.ylabel("Density")
    plt.show()
