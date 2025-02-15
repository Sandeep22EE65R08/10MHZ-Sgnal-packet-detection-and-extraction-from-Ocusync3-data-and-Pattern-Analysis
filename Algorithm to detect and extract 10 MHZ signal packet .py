import numpy as np
import matplotlib.pyplot as plt

def filter_signal_in_frequency_domain(signal, sampling_rate, cutoff_freq):
    """
    Apply a frequency-domain filter to the signal using the Discrete Fourier Transform (DFT).

    Parameters:
        signal (np.ndarray): The input signal to be filtered.
        sampling_rate (float): The sampling rate in Hz.
        cutoff_freq (float): The cutoff frequency for the filter.

    Returns:
        np.ndarray: The filtered signal in the time domain.
    """
    # Compute the DFT using numpy.fft.fft
    dft = np.fft.fft(signal)

    # Create a frequency array
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Create the filter: keep frequencies within the cutoff frequency range
    filter_mask = np.abs(freqs) < cutoff_freq

    # Apply the filter to the DFT
    dft_filtered = dft * filter_mask

    # Compute the inverse DFT to get the filtered signal in the time domain
    signal_filtered = np.fft.ifft(dft_filtered)

    return signal_filtered

def detect_packets_energy(data, sampling_rate, window_us, threshold_factor):
    """
    Detect signal packets based on energy accumulation.

    Parameters:
        data (np.ndarray): Input complex signal.
        sampling_rate (float): Sampling rate in Hz.
        window_us (float): Window size in microseconds.
        threshold_factor (float): Multiplier for the energy threshold.

    Returns:
        packets (list): List of (start_index, end_index) for detected packets.
        energy_profile (np.ndarray): Energy profile computed over sliding windows.
        threshold (float): Energy threshold used for detection.
    """
    # Convert window size from microseconds to samples
    window_samples = int((window_us / 1e6) * sampling_rate)

    # Calculate energy profile using a sliding window
    energy_profile = np.convolve(np.abs(data)**2, np.ones(window_samples), mode='same')

    # Set threshold for detection
    noise_level = np.mean(energy_profile)  # Baseline noise level
    threshold = threshold_factor * noise_level

    # Detect packets based on threshold
    detected = energy_profile > threshold
    packets = []
    start = None

    for i, is_high in enumerate(detected):
        if is_high and start is None:
            start = i  # Start of a packet
        elif not is_high and start is not None:
            end = i  # End of the packet
            packets.append((start, end))
            start = None

    # Handle the last packet if it continues to the end
    if start is not None:
        packets.append((start, len(detected)))

    return packets, energy_profile, threshold

# Parameters
sampling_rate = 50e6  # Sampling rate in Hz (50 Msps)
window_us = 20  # Window size in microseconds (for first signal packet)
threshold_factor = 150  # Threshold multiplier
cutoff_freq = 4e6  # Cutoff frequency for filtering

# Assuming `ocusync3_data` is the input signal
data = ocusync3_data  # Define your input signal here

# Step 1: Filter the signal
signal_filtered = filter_signal_in_frequency_domain(data, sampling_rate, cutoff_freq)

# Step 2: Detect Signal Packets
packets, energy_profile, threshold = detect_packets_energy(signal_filtered, sampling_rate, window_us, threshold_factor)

# Step 3: Plot Energy Profile with Detected Regions
plt.figure(figsize=(20, 6))
plt.plot(energy_profile, label="Energy Profile", color="blue")
plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
plt.title("Energy Profile with Detected Signal Packets")
plt.xlabel("Sample Index")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

# Step 4: Analyze and Plot Detected Packets
print(f"Number of detected packets: {len(packets)}")
for idx, (start_idx, end_idx) in enumerate(packets, start=1):
    start_time = start_idx / sampling_rate
    end_time = end_idx / sampling_rate
    print(f"Packet {idx}: Start Time = {start_time:.6f} s, End Time = {end_time:.6f} s, Duration = {end_time - start_time:.6f} s")

    # Extract corresponding signal samples
    first_packet = signal_filtered[start_idx:end_idx]

    # Plot the extracted packet
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(start_idx, end_idx) / sampling_rate, np.abs(first_packet), label=f"Packet {idx}")
    plt.title(f"Extracted Signal Packet {idx}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

# Step 5: Plot Amplitude vs. Sample for the First Packet
if packets:
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(first_packet), color='b')
    plt.title('Amplitude vs. Sample for the First Packet')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Step 6: Plot Spectrogram of the First Packet
    plt.figure(figsize=(10, 4))
    plt.specgram(first_packet, Fs=sampling_rate)
    plt.title('Spectrogram of 10 MHz signal packet', fontsize=20, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.show()
