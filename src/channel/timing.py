import numpy as np
from scipy.interpolate import interp1d


def apply_fractional_timing_offset(signal, offset_samples):
    """
    Apply a fractional timing offset using interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.

    offset_samples : float
        Timing offset expressed in samples.

        Positive value:
            waveform is delayed.

        Negative value:
            waveform is advanced.

    Returns
    -------
    shifted_signal : np.ndarray
        Signal after applying the fractional timing offset.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError(
            "Signal must be a complex-valued baseband signal."
        )

    if abs(offset_samples) >= len(signal):
        raise ValueError(
            "Timing offset is too large for the signal length."
        )

    sample_indices = np.arange(len(signal))

    real_interpolator = interp1d(
        sample_indices,
        np.real(signal),
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    imag_interpolator = interp1d(
        sample_indices,
        np.imag(signal),
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    shifted_indices = (
        sample_indices - offset_samples
    )

    shifted_real = real_interpolator(
        shifted_indices
    )

    shifted_imag = imag_interpolator(
        shifted_indices
    )

    shifted_signal = (
        shifted_real
        + 1j * shifted_imag
    )

    return shifted_signal