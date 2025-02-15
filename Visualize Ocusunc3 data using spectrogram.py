import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def load_data_and_create_time_array(filepath, sampling_rate):
    """
    Load complex signal data from a .dat file and create a time array.
    
    Parameters:
        filepath (str): The path to the .dat file containing the signal data.
        sampling_rate (float): The sampling rate in Hz.
    
    Returns:
        np.ndarray: Loaded complex signal data.
        np.ndarray: Time array corresponding to the signal data in seconds.
    """
    with open(filepath, "rb") as f:
        signal_data = np.fromfile(f, dtype=np.complex64)
    
    time_array = np.arange(len(signal_data)) / sampling_rate
    return signal_data, time_array

def calculate_signal_duration(num_samples, sampling_rate):
    """
    Calculate the duration of a signal in seconds.
    
    Parameters:
        num_samples (int): The number of samples in the signal.
        sampling_rate (float): The sampling rate in Hz.
    
    Returns:
        float: The duration of the signal in seconds.
    """
    return num_samples / sampling_rate

def plot_spectrogram(signal, sampling_rate):
    """
    Compute and plot the spectrogram of the signal.
    
    Parameters:
        signal (np.ndarray): The input signal.
        sampling_rate (float): The sampling rate in Hz.
    """
    plt.figure(figsize=(10, 4))
    plt.specgram(signal, Fs=sampling_rate)
    plt.title('Spectrogram of Ocusync3 Data', fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.show()

# Example usage
filepath = "/home/sandeep/Downloads/sample_bbbs_assignment_id100.dat"
sampling_rate = 50e6  # 50 Msps

# Load data and create time array
ocusync3_data, time = load_data_and_create_time_array(filepath, sampling_rate)

# Calculate signal duration
signal_duration = calculate_signal_duration(len(ocusync3_data), sampling_rate)
print(f"Number of samples: {len(ocusync3_data)}")
print(f"Duration of the original signal: {signal_duration:.6f} seconds")

# Plot spectrogram
plot_spectrogram(ocusync3_data, sampling_rate)
