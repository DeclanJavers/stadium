//! SplitMix64 and CRC-32, matching kit/stands/prng.py exactly.

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic uniform in [0, 1) for (seed, index).
pub fn uniform(seed: u64, i: u64) -> f64 {
    (splitmix64(seed.wrapping_mul(1_000_003).wrapping_add(i)) >> 11) as f64 / (1u64 << 53) as f64
}

/// zlib-compatible CRC-32.
pub fn crc32(s: &str) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for b in s.bytes() {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}
