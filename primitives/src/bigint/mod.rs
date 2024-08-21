pub mod check_carry_to_zero;
pub mod utils;

#[cfg(test)]
pub mod tests;

#[derive(Debug, Clone)]
pub struct OverflowInt<T> {
    // The limbs, e.g. [a_0, a_1, a_2, ...] , represents a_0 + a_1 x + a_2 x^2
    // T will be AB::Expr in practice, for example when the OverflowInt represents x * y
    // a0 = x0 * y0
    // a1 = x0 * y1 + x1 * y0
    pub limbs: Vec<T>,

    // All limbs should be within [-2^max_overflow_bits, 2^max_overflow_bits)
    // Tthis can be larger than the limb bits of x , y in the above example.
    pub max_overflow_bits: usize,
}
