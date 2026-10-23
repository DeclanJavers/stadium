"""Portable deterministic randomness (SplitMix64), so the Rust port reproduces every stadium
bit-for-bit. numpy's generators are not reproducible outside numpy."""
MASK = (1 << 64) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def uniform(seed, i):
    """Deterministic uniform in [0, 1) for (seed, index)."""
    return (splitmix64((seed * 1000003 + i) & MASK) >> 11) / float(1 << 53)


def crc32(text):
    import zlib
    return zlib.crc32(str(text).encode())
