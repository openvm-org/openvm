use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::AirBuilder;
use p3_field::{AbstractField, PrimeField64};

use super::{utils::range_check, OverflowInt};

pub struct CheckCarryToZeroCols<T> {
    pub carries: Vec<T>,
}

pub struct CheckCarryToZeroSubAir {
    // The number of bits for each limb (not overflowed). Example: 10.
    pub limb_bits: usize,

    // The number of limbs for the input param "expr" to the constrain_carry_to_zero
    // E.g. if the limb_bits = 10 and equation is A * B and A, B can be 256 bits -> 51.
    pub num_limbs: usize,

    // Carry can be negative, so this is the max abs of negative carry.
    pub carry_min_value_abs: usize,
    // todo
    pub carry_bits: usize,

    pub range_checker_bus: usize,
    // The range checker decomp bits.
    pub decomp: usize,
}

impl CheckCarryToZeroSubAir {
    pub fn new(
        limb_bits: usize,
        num_limbs: usize,
        range_checker_bus: usize,
        decomp: usize,
        carry_min_value_abs: usize,
        carry_bits: usize,
    ) -> Self {
        Self {
            limb_bits,
            num_limbs,
            range_checker_bus,
            decomp,
            carry_min_value_abs,
            carry_bits,
        }
    }

    pub fn constrain_carry_to_zero<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        expr: OverflowInt<AB::Expr>,
        cols: CheckCarryToZeroCols<AB::Var>,
    ) {
        println!("carry len: {:?}", cols.carries.len());
        // 1. Constrain the limbs size of carries.
        for &carry in cols.carries.iter() {
            range_check(
                builder,
                self.range_checker_bus,
                self.decomp,
                self.carry_bits,
                // TODO: if carry is positive, would this overflow?
                carry + AB::F::from_canonical_usize(self.carry_min_value_abs),
            );
        }

        // 2. Constrain the carries and expr.
        assert!(expr.limbs.len() == cols.carries.len());
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
