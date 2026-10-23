//! Python / numpy numeric semantics, reproduced exactly so the port matches the reference generators.

/// Python's round(): half to even.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Python float floor division `a // b` (CPython's float_floor_div).
pub fn floordiv(a: f64, b: f64) -> f64 {
    let mut m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
            div -= 1.0;
        }
    }
    let _ = m;
    if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        (0.0f64).copysign(a / b)
    }
}

/// Python integer floor division for i64.
pub fn ifloordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

/// numpy.linspace(start, stop, n) (endpoint included).
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    v[n - 1] = stop;
    v
}

/// numpy.arange(start, stop, step) for positive steps.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    if n <= 0.0 {
        return vec![];
    }
    (0..n as usize).map(|i| start + i as f64 * step).collect()
}

/// numpy.interp for increasing xp.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let mut i = xp.partition_point(|v| *v <= x) - 1;
    if i >= xp.len() - 1 {
        i = xp.len() - 2;
    }
    let t = (x - xp[i]) / (xp[i + 1] - xp[i]);
    fp[i] + t * (fp[i + 1] - fp[i])
}

pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Python round(x, n) for display values (half-even on the scaled value; parity within 1e-n).
pub fn round_to(x: f64, n: i32) -> f64 {
    let f = 10f64.powi(n);
    round_half_even(x * f) / f
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_semantics() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(floordiv(7.0, 0.5), 14.0);
        assert_eq!(floordiv(-7.5, 2.0), -4.0);
        assert_eq!(ifloordiv(-7, 2), -4);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(arange(4.5, 20.0, 4.5).len(), 4);
        assert!((interp(1.5, &[0.0, 1.0, 2.0], &[0.0, 10.0, 30.0]) - 20.0).abs() < 1e-12);
    }
}
