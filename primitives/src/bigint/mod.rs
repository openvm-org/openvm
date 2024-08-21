use p3_air::AirBuilder;

pub mod check_carry_to_zero;
pub mod utils;

#[cfg(test)]
pub mod tests;

#[derive(Debug, Clone)]
pub struct OverflowInt<T> {
    // The limbs, e.g. [a_0, a_1, a_2, ...] , represents a_0 + a_1 x + a_2 x^2
    pub limbs: Vec<T>,

    // All limbs should be within [-2^max_overflow_bits, 2^max_overflow_bits)
    // Note that this can be larger than the limb bits.
    pub max_overflow_bits: usize,
}

impl<T> OverflowInt<T> {
    // From a standard (not overflowed) big int
    pub fn new<AB: AirBuilder>(v: Vec<AB::Var>, limb_bits: usize) -> OverflowInt<AB::Expr> {
        let mut limbs: Vec<AB::Expr> = Vec::with_capacity(v.len());
        for v in v.into_iter() {
            limbs.push(v.into());
        }
        OverflowInt {
            limbs,
            max_overflow_bits: limb_bits,
        }
    }
}
