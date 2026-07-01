//! xorshift32 PRNG. Implemented by hand since no external crates are allowed.

/// Seedable xorshift32 generator. Deterministic for a given seed.
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Seed must be non-zero; xorshift is stuck at 0.
    pub fn new(seed: u32) -> Self {
        Rng { state: seed | 1 }
    }

    /// Next raw 32-bit value.
    pub fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Uniform `f32` in `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Fisher–Yates in-place shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u32() as usize) % (i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_reproducible() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn next_f32_stays_in_unit_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = Rng::new(1);
        let mut v: Vec<i32> = (0..50).collect();
        rng.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..50).collect::<Vec<_>>());
    }
}
