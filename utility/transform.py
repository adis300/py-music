from scipy.fftpack import fft
import numpy as np


def fft_2sided(signal, sample_rate):

    # Number of sample points
    L = len(signal)  # Length of signal
    # T = 1.0 / _SAMPLE_RATE  # Sampling period
    # t = np.linspace(0.0, (L - 1) * T, L)  # Time vector

    y_fft = fft(signal)

    s2 = np.abs(y_fft / L)  # two-sided spectrum s2
    f = np.linspace(0.0, sample_rate / 2, L // 2 + 1)
    return f, s2


def fft_1sided(signal, sample_rate):
    # Number of sample points
    L = len(signal)  # Length of signal
    # T = 1.0 / _SAMPLE_RATE  # Sampling period

    # t = np.linspace(0.0, (L - 1) * T, L)  # Time vector
    y_fft = fft(signal)

    s2 = np.abs(y_fft / L)  # normalized two-sided spectrum s2
    s1 = s2[: (L // 2 + 1)]  # one-sided spectrum s1
    f = np.linspace(0.0, sample_rate / 2, L // 2 + 1)

    return f, s1
