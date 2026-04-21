import numpy as np

def fn_create_tone_burst(fs, centre_freq, num_cycles,
                         window_type='hanning',
                         total_length=None,
                         centre_time=None):
    """
    Generate a windowed sinusoidal tone burst with controllable centre time.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz]
    centre_freq : float
        Tone burst centre frequency [Hz]
    num_cycles : int
        Number of cycles
    window_type : str
        'hanning', 'hann', 'rect'
    total_length : int or None
        Total number of samples of the output waveform.
        If None, use only the burst length.
    centre_time : float or None
        Desired centre time of the burst [s].
        If None, burst is centred in its own short waveform.
        If total_length is given and centre_time=0, burst starts near t=0.

    Returns
    -------
    t : ndarray
        Time vector, shape (N,)
    signal : ndarray
        Tone burst signal, shape (N,), normalized to ±1
    """

    burst_duration = num_cycles / centre_freq
    n_burst = int(np.round(burst_duration * fs))
    n_burst = max(n_burst, 8)

    # Burst-local time axis
    t_burst = np.arange(n_burst) / fs

    # Carrier
    burst = np.sin(2 * np.pi * centre_freq * t_burst)

    # Window
    wt = window_type.lower()
    if wt in ['hanning', 'hann']:
        win = np.hanning(n_burst)
    elif wt == 'rect':
        win = np.ones(n_burst)
    else:
        raise ValueError(f"Unsupported window_type: {window_type}")

    burst = burst * win

    # Normalize burst to ±1
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst = burst / peak

    # If no total length requested, just return the burst itself
    if total_length is None:
        return t_burst, burst

    # Full output waveform
    N = int(total_length)
    signal = np.zeros(N, dtype=float)
    t = np.arange(N) / fs

    # Default centre_time:
    # if not specified, place burst at beginning with its own centre
    if centre_time is None:
        centre_time = burst_duration / 2

    centre_index = int(np.round(centre_time * fs))
    start_index = centre_index - n_burst // 2
    end_index = start_index + n_burst

    # Clip to array bounds if needed
    src_start = 0
    src_end = n_burst

    if start_index < 0:
        src_start = -start_index
        start_index = 0

    if end_index > N:
        src_end = n_burst - (end_index - N)
        end_index = N

    if start_index < end_index and src_start < src_end:
        signal[start_index:end_index] = burst[src_start:src_end]

    return t, signal