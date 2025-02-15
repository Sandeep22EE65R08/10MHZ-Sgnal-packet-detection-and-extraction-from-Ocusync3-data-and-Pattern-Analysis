def calculate_signal_duration_and_samples(num_samples, sampling_rate):
    """
    Calculate the duration of a signal and print the number of samples.

    Parameters:
        num_samples (int): The number of samples in the signal.
        sampling_rate (float): The sampling rate in Hz.

    Returns:
        float: The duration of the signal in seconds.
    """
    signal_duration = num_samples / sampling_rate  # Duration in seconds
    print(f"Number of samples: {num_samples}")
    return signal_duration

# Example usage
num_samples = len(first_packet)  # Replace with your actual signal data
sampling_rate = 50e6  # Replace with your actual sampling rate

# Calculate signal duration and print the number of samples
signal_duration = calculate_signal_duration_and_samples(num_samples, sampling_rate)

# Print the duration
print(f"Duration of the 10 MHz signal packet: {signal_duration:.6f} seconds")
