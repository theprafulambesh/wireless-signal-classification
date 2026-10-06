import numpy as np
from scipy.signal import lfilter


def rrc_filter(beta, span, sps):
    """
    Generate Root Raised Cosine (RRC) FIR filter coefficients.

    beta : roll-off factor
    span : filter length in symbols
    sps  : samples per symbol
    """

    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1.")

    if span <= 0:
        raise ValueError("span must be positive.")

    if sps <= 0:
        raise ValueError("sps must be positive.")






























    

    N = span * sps

    t = np.arange(-N / 2, N / 2 + 1) / sps

    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):

        if np.isclose(ti, 0.0):

            h[i] = 1 + beta * (4 / np.pi - 1)

        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):

            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * beta))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * beta))
                )
            )

        else:

            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                +
                4 * beta * ti
                * np.cos(np.pi * ti * (1 + beta))
            )

            denominator = (
                np.pi * ti
                * (1 - (4 * beta * ti) ** 2)
            )

            h[i] = numerator / denominator

    # Normalize filter energy.
    h /= np.sqrt(np.sum(h ** 2))

    return h


def pulse_shape(symbols, beta=0.35, span=8, sps=8):
    """
    Upsample symbols and apply RRC pulse shaping.
    """

    symbols = np.asarray(symbols)

    # Insert zeros between symbols.
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)

    upsampled[::sps] = symbols

    # RRC filtering.
    waveform = lfilter(rrc_filter(beta, span, sps), 1, upsampled)

    return waveform