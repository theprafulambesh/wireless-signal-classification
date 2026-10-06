import numpy as np


def bits_to_integers(bits, bits_per_symbol):
    """
    Convert a 1-D bit array into integer symbol indices.

    Example:
        bits = [1, 0, 1, 1]
        bits_per_symbol = 2

        -> [2, 3]
    """

    bits = np.asarray(bits)

    if len(bits) % bits_per_symbol != 0:
        raise ValueError("Number of bits must be divisible by bits_per_symbol.")

    grouped_bits = bits.reshape(-1, bits_per_symbol)

    weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)

    return grouped_bits @ weights


def generate_bits(num_bits, rng=None):
    """Generate random binary data."""

    if rng is None:
        rng = np.random.default_rng()

    return rng.integers(0, 2, size=num_bits)


def bpsk_modulate(bits):
    """
    BPSK:
        0 -> +1
        1 -> -1
    """

    bits = np.asarray(bits)

    symbols = 1 - 2 * bits

    return symbols.astype(np.complex128)


def qpsk_modulate(bits):
    """
    Gray-coded QPSK.

    Mapping:
        00 -> ( 1 + j) / sqrt(2)
        01 -> (-1 + j) / sqrt(2)
        11 -> (-1 - j) / sqrt(2)
        10 -> ( 1 - j) / sqrt(2)
    """

    symbol_indices = bits_to_integers(bits, 2)

    mapping = np.array([
        (1 + 1j) / np.sqrt(2),    # 00
        (-1 + 1j) / np.sqrt(2),   # 01
        (1 - 1j) / np.sqrt(2),    # 10
        (-1 - 1j) / np.sqrt(2)    # 11
    ])

    return mapping[symbol_indices]


def psk8_modulate(bits):
    """
    8-PSK modulation.

    3 bits -> 1 symbol.
    """

    symbol_indices = bits_to_integers(bits, 3)

    phase = 2 * np.pi * symbol_indices / 8

    return np.exp(1j * phase)


def qam16_modulate(bits):
    """
    16-QAM modulation using Gray-coded I/Q levels.

    Each symbol carries 4 bits:
        2 bits -> I
        2 bits -> Q
    """

    bits = np.asarray(bits)

    if len(bits) % 4 != 0:
        raise ValueError("Number of bits must be divisible by 4.")

    groups = bits.reshape(-1, 4)

    # Gray-coded mapping:
    # 00 -> -3
    # 01 -> -1
    # 11 -> +1
    # 10 -> +3

    gray_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1): 1,
        (1, 0): 3
    }

    I = np.array([
        gray_map[tuple(pair)]
        for pair in groups[:, :2]
    ])

    Q = np.array([
        gray_map[tuple(pair)]
        for pair in groups[:, 2:]
    ])

    # Normalize average symbol power to approximately 1.
    normalization = np.sqrt(10)

    return (I + 1j * Q) / normalization


def modulate(bits, modulation):
    """
    General modulation interface.
    """

    modulation = modulation.upper()

    if modulation == "BPSK":
        return bpsk_modulate(bits)

    elif modulation == "QPSK":
        return qpsk_modulate(bits)

    elif modulation == "8PSK":
        return psk8_modulate(bits)

    elif modulation == "16QAM":
        return qam16_modulate(bits)

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")