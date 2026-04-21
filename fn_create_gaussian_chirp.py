
import numpy as np


def fn_create_gaussian_chirp(fs, duration, f0, f1):
    """
    Create a linear chirp multiplied by a Gaussian window.
    Output is normalized to [-1, 1] approximately.
    """
    n = int(round(fs * duration))
    if n < 16:
        raise ValueError("Pulse duration too short for selected AWG sample rate.")

    t = np.arange(n, dtype=np.float64) / fs

    # Linear chirp phase:
    # phi(t) = 2*pi*(f0*t + 0.5*k*t^2), where k = (f1-f0)/T
    k = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t**2)
    chirp = np.sin(phase)

    # Gaussian window centered in the pulse
    tc = duration / 2.0
    sigma = duration / 6.0   # reasonably compact Gaussian
    window = np.exp(-0.5 * ((t - tc) / sigma) ** 2)

    signal = chirp * window

    # Normalize for arbitrary waveform generator
    peak = np.max(np.abs(signal))
    if peak <= 0:
        raise ValueError("Generated chirp has zero amplitude.")
    signal /= peak

    return t, signal