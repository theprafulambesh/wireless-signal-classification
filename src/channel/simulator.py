import numpy as np

from channel.awgn import add_awgn
from channel.phase import apply_phase_offset
from channel.cfo import apply_cfo
from channel.timing import apply_fractional_timing_offset
from channel.sample_rate import apply_sample_rate_offset

from channel.multipath import (
    apply_multipath,
    apply_time_varying_multipath
)

from channel.fading import (
    generate_correlated_rayleigh_channel,
    generate_rician_channel
)


def _apply_channel_fading(
    signal,
    config,
    rng
):

    path_delays = np.asarray(
        config.path_delays,
        dtype=int
    )

    path_gains = np.asarray(
        config.path_gains,
        dtype=np.complex128
    )

    if len(path_delays) != len(path_gains):
        raise ValueError(
            "path_delays and path_gains "
            "must have the same length."
        )

    if config.fading_type == "none":

        return apply_multipath(
            signal,
            path_gains=path_gains,
            path_delays=path_delays
        )

    path_channels = []

    for gain in path_gains:

        if config.fading_type == "rayleigh":

            h = generate_correlated_rayleigh_channel(
                num_samples=len(signal),
                rho=config.fading_rho,
                rng=rng
            )

        elif config.fading_type == "rician":

            h = np.array([
                generate_rician_channel(
                    K=config.rician_k,
                    rng=rng
                )
                for _ in range(len(signal))
            ])

        else:

            raise ValueError(
                f"Unsupported fading type: "
                f"{config.fading_type}"
            )

        path_channels.append(
            gain * h
        )

    return apply_time_varying_multipath(
        signal,
        path_delays=path_delays,
        path_channels=path_channels
    )


def simulate_channel(
    signal,
    config,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng()

    received = np.asarray(
        signal,
        dtype=np.complex128
    ).copy()

    # 1. Multipath + fading
    received = _apply_channel_fading(
        received,
        config,
        rng
    )

    # 2. Phase offset
    if config.phase_offset_rad != 0:

        received = apply_phase_offset(
            received,
            config.phase_offset_rad
        )

    # 3. CFO
    if config.cfo_hz != 0:

        received = apply_cfo(
            received,
            cfo_hz=config.cfo_hz,
            sample_rate_hz=config.sample_rate_hz
        )

    # 4. Fractional timing offset
    if config.timing_offset_samples != 0:

        received = apply_fractional_timing_offset(
            received,
            config.timing_offset_samples
        )

    # 5. SRO
    if config.sro_ppm != 0:

        received = apply_sample_rate_offset(
            received,
            sample_rate_hz=config.sample_rate_hz,
            sro_ppm=config.sro_ppm
        )

    # 6. AWGN
    if config.snr_db is not None:

        received, noise = add_awgn(
            received,
            snr_db=config.snr_db,
            rng=rng
        )

    return received