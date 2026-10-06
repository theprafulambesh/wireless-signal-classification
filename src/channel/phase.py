import numpy as np


def apply_phase_offset(signal, phase_rad):
    """
    Apply a constant phase offset to a complex baseband signal.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.

    phase_rad : float
        Constant phase offset in radians.

    Returns
    -------
    shifted_signal : np.ndarray
        Phase-rotated complex signal.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError("Signal must be a complex-valued baseband signal.")

    return signal * np.exp(1j * phase_rad)


def random_phase_offset(signal, rng=None):
    """
    Apply a random constant phase offset.

    Phase is sampled uniformly from [-pi, pi].

    Returns
    -------
    shifted_signal : np.ndarray
        Phase-rotated signal.

    phase_rad : float
        Random phase that was applied.
    """

    if rng is None:
        rng = np.random.default_rng()

    phase_rad = rng.uniform(-np.pi, np.pi)

    shifted_signal = apply_phase_offset(
        signal,
        phase_rad
    )

    return shifted_signal, phase_rad