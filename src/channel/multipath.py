import numpy as np
from scipy.signal import lfilter


def apply_multipath(
    signal,
    path_gains,
    path_delays
):
    """
    Apply a discrete-time multipath channel.

    Parameters
    ----------
    signal : array-like
        Complex input waveform.

    path_gains : array-like
        Complex or real gain for each propagation path.

    path_delays : array-like
        Delay of each path in samples.

    Returns
    -------
    output : np.ndarray
        Complex multipath-distorted waveform.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError("signal must be complex-valued.")

    path_gains = np.asarray(path_gains, dtype=np.complex128)
    path_delays = np.asarray(path_delays)

    if len(path_gains) != len(path_delays):
        raise ValueError(
            "path_gains and path_delays must have the same length."
        )

    if len(path_gains) == 0:
        raise ValueError("At least one propagation path is required.")

    if np.any(path_delays < 0):
        raise ValueError("Path delays must be non-negative.")

    if not np.all(np.equal(path_delays, path_delays.astype(int))):
        raise ValueError(
            "Path delays must be integer numbers of samples."
        )

    path_delays = path_delays.astype(int)

    # Channel impulse response
    max_delay = np.max(path_delays)

    h = np.zeros(
        max_delay + 1,
        dtype=np.complex128
    )

    for gain, delay in zip(path_gains, path_delays):
        h[delay] += gain

    # Apply channel through convolution
    output = np.convolve(signal, h)

    return output

def apply_time_varying_multipath(
    signal,
    path_delays,
    path_channels
):
    """
    Apply a time-varying multipath fading channel.

    Parameters
    ----------
    signal : np.ndarray
        Complex input waveform.

    path_delays : array-like
        Delay of each path in samples.

    path_channels : list of np.ndarray
        Time-varying complex channel coefficient for
        each path.

    Returns
    -------
    output : np.ndarray
        Time-varying multipath-faded waveform.
    """

    signal = np.asarray(signal)

    path_delays = np.asarray(
        path_delays,
        dtype=int
    )

    if len(path_delays) != len(path_channels):
        raise ValueError(
            "Number of delays and channels must match."
        )

    if np.any(path_delays < 0):
        raise ValueError(
            "Path delays must be non-negative."
        )

    max_delay = np.max(path_delays)

    output = np.zeros(
        len(signal) + max_delay,
        dtype=np.complex128
    )

    for delay, channel in zip(
        path_delays,
        path_channels
    ):

        channel = np.asarray(channel)

        if len(channel) != len(signal):
            raise ValueError(
                "Each path channel must have "
                "the same length as signal."
            )

        delayed_signal = np.zeros(
            len(signal) + max_delay,
            dtype=np.complex128
        )

        delayed_signal[
            delay:delay + len(signal)
        ] = (
            signal * channel
        )

        output += delayed_signal

    return output