import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def moment_based_skew(distribution):
    """
    Calculate skewness using the moment-based method.
    Uses the magnitude of the complex signal.
    """
    magnitude = np.abs(distribution)
    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)

    # Moment-based skewness formula
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((magnitude - mean) / std) ** 3)
    return skewness

def moment_based_kurtosis(distribution):
    """
    Calculate kurtosis using the moment-based method.
    Uses the magnitude of the complex signal.
    """
    magnitude = np.abs(distribution)
    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)

    # Moment-based kurtosis formula (excess kurtosis)
    kurt = (1 / n) * np.sum(((magnitude - mean) / std) ** 4) - 3
    return kurt

# Functions to classify skewness and kurtosis
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

# For the first_packet
skew_value = moment_based_skew(first_packet)
kurtosis_value = moment_based_kurtosis(first_packet)

# Classify results
skew_classification = classify_skewness(skew_value)
kurtosis_classification = classify_kurtosis(kurtosis_value)

# Print results
print(f"Skewness: {skew_value:.4f} -> {skew_classification}")
print(f"Kurtosis: {kurtosis_value:.4f} -> {kurtosis_classification}\n")

# KDE plot for the first_packet
plt.figure(figsize=(10, 4))
sns.kdeplot(np.abs(first_packet), shade=True)
plt.title("KDE Plot of Extracted First Packet")
plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.show()
