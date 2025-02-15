import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def smooth_energy_curve(energy_curve, window_size):
    """
    Smooth the energy curve using a moving average.
    
    Parameters:
    - energy_curve: The raw energy values (squared magnitudes).
    - window_size: The size of the moving average window.
    
    Returns:
    - smoothed_energy: The smoothed energy curve.
    """
    return uniform_filter1d(energy_curve, size=window_size)

def extract_patterns_with_smoothing(filtered_packet, energy_threshold, pattern_length, window_size):
    """
    Detect and extract repeating patterns from the filtered packet using energy detection with smoothing.
    
    Parameters:
    - filtered_packet: The signal data after filtering.
    - energy_threshold: The energy threshold below which a pattern is detected.
    - pattern_length: The length of each pattern to extract (3600 samples).
    - window_size: The size of the smoothing window for the energy curve.
    
    Returns:
    - extracted_patterns: List of extracted patterns.
    - detected_starts: List of start indices of detected patterns.
    - detected_ends: List of end indices of detected patterns.
    - smoothed_energy: The smoothed energy curve.
    """
    # Compute energy per sample
    energy_per_sample = np.abs(filtered_packet)**2

    # Apply smoothing to the energy curve
    smoothed_energy = smooth_energy_curve(energy_per_sample, window_size)

    # Initialize storage for extracted patterns and indices
    detected_starts = []
    detected_ends = []
    extracted_patterns = []

    # Identify indices where smoothed energy is below the threshold
    below_threshold_indices = np.where(smoothed_energy < energy_threshold)[0]

    # Detect continuous regions below the threshold
    current_start = None
    for i in range(1, len(below_threshold_indices)):
        if below_threshold_indices[i] != below_threshold_indices[i - 1] + 1:
            # Found the end of a region
            if current_start is not None:
                drop_duration = below_threshold_indices[i - 1] - current_start + 1
                if drop_duration >= pattern_length:
                    # Extract exactly 'pattern_length' samples
                    end_sample = current_start + pattern_length
                    detected_starts.append(current_start)
                    detected_ends.append(end_sample)
                    extracted_patterns.append(first_packet[current_start:end_sample])
            # Start a new region
            current_start = below_threshold_indices[i]
    
    # Final region check
    if current_start is not None:
        drop_duration = below_threshold_indices[-1] - current_start + 1
        if drop_duration >= pattern_length:
            end_sample = current_start + pattern_length
            detected_starts.append(current_start)
            detected_ends.append(end_sample)
            extracted_patterns.append(filtered_packet[current_start:end_sample])

    return extracted_patterns, detected_starts, detected_ends, smoothed_energy

# Example Usage
# Assuming `filtered_packet` is already defined

# Parameters
energy_threshold = 0.0138  # Energy threshold for detection
pattern_length = 3600    # Length of each pattern (72 microseconds at 50 Msps)
window_size = 5        # Window size for smoothing

# Detect and extract patterns
extracted_patterns, detected_starts, detected_ends, smoothed_energy = extract_patterns_with_smoothing(
    first_packet, energy_threshold, pattern_length, window_size
)

# Output the detected patterns' indices
for idx, (start, end) in enumerate(zip(detected_starts, detected_ends)):
    print(f"Pattern {idx + 1}: Start Sample = {start}, End Sample = {end}")

print(f"Number of patterns detected: {len(extracted_patterns)}")

# Plot the smoothed energy curve
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(smoothed_energy)), smoothed_energy, label="Smoothed Energy", color="red")
plt.axhline(y=energy_threshold, color="blue", linestyle="--", label="Energy Threshold")
for start, end in zip(detected_starts, detected_ends):
    plt.axvspan(start, end, color='green', alpha=0.3, label="Detected Pattern" if 'Detected Pattern' not in plt.gca().get_legend_handles_labels()[1] else None)

plt.title("Smoothed Energy Curve with Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Energy (Squared Magnitude)")
plt.legend()
plt.grid(True)
plt.show()

# Visualize the extracted patterns
for idx, pattern in enumerate(extracted_patterns):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(pattern)), np.abs(pattern), label=f"Pattern {idx + 1}")
    plt.title(f"Extracted Pattern {idx + 1}")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize the spectrogram of the pattern
    plt.figure(figsize=(10, 4))
    plt.specgram(pattern, NFFT=256, Fs=50e6, noverlap=128, scale='dB')
    plt.title(f"Spectrogram of Pattern {idx + 1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power/Frequency [dB]")
    plt.show()
