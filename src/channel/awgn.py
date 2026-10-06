import numpy as np


def add_awgn(signal, snr_db, rng=None):
    """
    Add complex AWGN to a complex baseband signal.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.

    snr_db : float
        Desired signal-to-noise ratio in dB.

    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    noisy_signal : np.ndarray
        Signal with complex AWGN added.

    noise : np.ndarray
        Generated complex AWGN.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError("Signal must be a complex-valued baseband signal.")

    if rng is None:
        rng = np.random.default_rng()

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Average signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    if signal_power <= 0:
        raise ValueError("Signal power must be greater than zero.")

    # Required total noise power
    noise_power = signal_power / snr_linear

    # Complex Gaussian noise:
    # E[|w|^2] = noise_power
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape)
        + 1j * rng.standard_normal(signal.shape)
    )

    noisy_signal = signal + noise

    return noisy_signal, noise