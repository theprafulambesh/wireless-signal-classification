import numpy as np


def apply_cfo(signal, cfo_hz, sample_rate_hz, phase_rad=0.0):
    """
    Apply carrier frequency offset (CFO) to a complex baseband signal.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.

    cfo_hz : float
        Carrier frequency offset in Hz.

    sample_rate_hz : float
        Sampling frequency in Hz.

    phase_rad : float, optional
        Initial phase offset in radians.

    Returns
    -------
    shifted_signal : np.ndarray
        Signal after applying CFO.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError(
            "Signal must be a complex-valued baseband signal."
        )

    if sample_rate_hz <= 0:
        raise ValueError(
            "sample_rate_hz must be greater than zero."
        )

    n = np.arange(len(signal))

    Ts = 1 / sample_rate_hz

    phase = (
        2 * np.pi * cfo_hz * n * Ts
        + phase_rad
    )

    rotation = np.exp(1j * phase)

    shifted_signal = signal * rotation

    return shifted_signal