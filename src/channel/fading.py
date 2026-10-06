import numpy as np


def generate_rayleigh_channel(
    rng=None,
    normalize_power=True
):
    """
    Generate one complex Rayleigh fading coefficient.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random-number generator.

    normalize_power : bool
        If True, normalize E[|h|^2] approximately to 1.

    Returns
    -------
    h : complex
        Complex Rayleigh fading coefficient.
    """

    if rng is None:
        rng = np.random.default_rng()

    h = (
        rng.normal()
        + 1j * rng.normal()
    )

    if normalize_power:
        h /= np.sqrt(2)

    return h


def apply_flat_fading(
    signal,
    channel_gain
):
    """
    Apply a flat complex fading coefficient.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.

    channel_gain : complex
        Complex channel coefficient.

    Returns
    -------
    faded_signal : np.ndarray
        Faded signal.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError(
            "Signal must be complex-valued."
        )

    return channel_gain * signal


def generate_rician_channel(
    K,
    rng=None,
    los_phase=0.0
):
    """
    Generate one complex Rician fading coefficient.

    Parameters
    ----------
    K : float
        Rician K-factor.

    rng : np.random.Generator, optional
        Random-number generator.

    los_phase : float
        Phase of the deterministic LOS component.

    Returns
    -------
    h : complex
        Complex Rician fading coefficient.
    """

    if K < 0:
        raise ValueError(
            "K must be non-negative."
        )

    if rng is None:
        rng = np.random.default_rng()

    los_component = np.exp(
        1j * los_phase
    )

    scattered_component = (
        rng.normal()
        + 1j * rng.normal()
    ) / np.sqrt(2)

    h = (
        np.sqrt(K / (K + 1))
        * los_component
        +
        np.sqrt(1 / (K + 1))
        * scattered_component
    )

    return h

def generate_correlated_rayleigh_channel(
    num_samples,
    rho=0.99,
    rng=None
):
    """
    Generate a temporally correlated complex Rayleigh
    fading process.

    Parameters
    ----------
    num_samples : int
        Number of channel coefficients.

    rho : float
        Temporal correlation coefficient.

        Values close to 1 produce slowly varying fading.

    rng : np.random.Generator, optional
        Random-number generator.

    Returns
    -------
    h : np.ndarray
        Complex time-varying Rayleigh channel.
    """

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if not 0 <= rho < 1:
        raise ValueError(
            "rho must satisfy 0 <= rho < 1."
        )

    if rng is None:
        rng = np.random.default_rng()

    h = np.zeros(
        num_samples,
        dtype=np.complex128
    )

    # Initial channel state
    h[0] = (
        rng.normal()
        + 1j * rng.normal()
    ) / np.sqrt(2)

    for n in range(1, num_samples):

        innovation = (
            rng.normal()
            + 1j * rng.normal()
        ) / np.sqrt(2)

        h[n] = (
            rho * h[n - 1]
            +
            np.sqrt(1 - rho**2)
            * innovation
        )

    return h

def apply_time_varying_fading(
    signal,
    channel
):
    """
    Apply a time-varying complex fading process.

    Parameters
    ----------
    signal : np.ndarray
        Complex input waveform.

    channel : np.ndarray
        Complex channel coefficient for every sample.

    Returns
    -------
    output : np.ndarray
        Faded waveform.
    """

    signal = np.asarray(signal)
    channel = np.asarray(channel)

    if len(signal) != len(channel):
        raise ValueError(
            "signal and channel must have the same length."
        )

    if not np.iscomplexobj(signal):
        raise ValueError(
            "signal must be complex-valued."
        )

    if not np.iscomplexobj(channel):
        raise ValueError(
            "channel must be complex-valued."
        )

    return signal * channel