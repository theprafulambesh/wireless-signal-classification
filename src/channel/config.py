from dataclasses import dataclass


@dataclass
class ChannelConfig:

    # Sample rate
    sample_rate_hz: float

    # AWGN
    snr_db: float

    # Carrier frequency offset
    cfo_hz: float

    # Initial carrier phase
    phase_offset_rad: float

    # Fractional timing offset
    timing_offset_samples: float

    # Sample-rate offset
    sro_ppm: float

    # Multipath
    path_delays: list
    path_gains: list

    # Fading
    fading_type: str = "none"

    # Temporal correlation
    fading_rho: float = 0.99

    # Rician K factor
    rician_k: float = 0.0