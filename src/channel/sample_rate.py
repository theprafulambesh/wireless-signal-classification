import numpy as np
from scipy.interpolate import interp1d


def apply_sample_rate_offset(
    signal,
    sample_rate_hz,
    sro_ppm
):
    """
    Apply sample-rate offset (SRO) to a complex baseband signal.

    Parameters
    ----------
    signal : array-like
        Complex input signal sampled at the transmitter sample rate.

    sample_rate_hz : float
        Nominal transmitter sample rate in Hz.

    sro_ppm : float
        Sample-rate offset in parts per million.

        Positive value:
            receiver sample rate is higher.

        Negative value:
            receiver sample rate is lower.

    Returns
    -------
    output : np.ndarray
        Complex signal sampled according to the receiver clock.
    """

    signal = np.asarray(signal)

    if not np.iscomplexobj(signal):
        raise ValueError("signal must be complex-valued.")

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if abs(sro_ppm) >= 1e6:
        raise ValueError("sro_ppm is unrealistically large.")

    # Convert ppm to fractional frequency error
    epsilon = sro_ppm * 1e-6

    # Receiver sample rate
    receiver_sample_rate = sample_rate_hz * (1.0 + epsilon)

    # Number of output samples.
    #
    # For this first simulator we keep approximately the same
    # observation duration as the input signal.
    duration = len(signal) / sample_rate_hz

    num_output_samples = int(
        np.floor(duration * receiver_sample_rate)
    )

    if num_output_samples < 2:
        raise ValueError(
            "Signal is too short for the requested sample-rate offset."
        )

    # Original transmitter sample locations
    tx_indices = np.arange(len(signal))

    # Receiver sample times
    receiver_times = (
        np.arange(num_output_samples)
        / receiver_sample_rate
    )

    # Convert receiver times to positions on TX sample grid
    source_indices = receiver_times * sample_rate_hz

    # Interpolate I and Q independently
    real_interpolator = interp1d(
        tx_indices,
        np.real(signal),
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    imag_interpolator = interp1d(
        tx_indices,
        np.imag(signal),
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    output_real = real_interpolator(source_indices)
    output_imag = imag_interpolator(source_indices)

    output = output_real + 1j * output_imag

    return output