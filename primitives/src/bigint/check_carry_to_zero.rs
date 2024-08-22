use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{utils::range_check, OverflowInt};

pub struct CheckCarryToZeroCols<T> {
    pub carries: Vec<T>,
}

pub struct CheckCarryToZeroSubAir {
    // The number of bits for each limb (not overflowed). Example: 10.
    pub limb_bits: usize,
    // The max number of bits for overflowed limbs.
    pub max_overflow_bits: usize,

    // Carry can be negative, so this is the max abs of negative carry.
    // We will add this to carries to make them positive so we can range check them.
    pub carry_min_value_abs: usize,
    // The max number of bits for carry + carry_min_value_abs.
    pub carry_bits: usize,

    pub range_checker_bus: usize,
    // The range checker decomp bits.
    pub decomp: usize,
}

impl CheckCarryToZeroSubAir {
    pub fn new(
        limb_bits: usize,
        range_checker_bus: usize,
        decomp: usize,
        max_overflow_bits: usize,
    ) -> Self {
        let carry_bits = max_overflow_bits - limb_bits;
        let carry_min_value_abs = 1 << carry_bits;
        let carry_abs_bits = carry_bits + 1;
        Self {
            limb_bits,
            max_overflow_bits,
            carry_min_value_abs,
            carry_bits: carry_abs_bits,
            range_checker_bus,
            decomp,
        }
    }

    pub fn constrain_carry_to_zero<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        expr: OverflowInt<AB::Expr>,
        cols: CheckCarryToZeroCols<AB::Var>,
    ) {
        assert_eq!(expr.limbs.len(), cols.carries.len());
        assert_eq!(self.max_overflow_bits, expr.max_overflow_bits);
        // 1. Constrain the limbs size of carries.
        for &carry in cols.carries.iter() {
            range_check(
                builder,
                self.range_checker_bus,
                self.decomp,
                self.carry_bits,
                carry + AB::F::from_canonical_usize(self.carry_min_value_abs),
            );
        }

        // 2. Constrain the carries and expr.
        let mut previous_carry = AB::Expr::zero();
        for (i, limb) in expr.limbs.iter().enumerate() {
            builder.assert_eq(
                limb.clone() + previous_carry.clone(),
                cols.carries[i] * AB::F::from_canonical_usize(1 << self.limb_bits),
            );
            previous_carry = cols.carries[i].into();
        }
        // The last (highest) carry should be zero.
        builder.assert_eq(previous_carry, AB::Expr::zero());
    }
}
