use itertools::Itertools;
use num_bigint::BigUint;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_field::PrimeCharacteristicRing,
};

use super::{
    check_carry_to_zero::{CheckCarryToZeroCols, CheckCarryToZeroSubAir},
    utils::{big_uint_to_limbs, range_check},
    OverflowInt,
};
use crate::SubAir;

#[derive(Clone)]
pub struct CheckCarryModToZeroCols<T> {
    /// Carries for converting the remainder to canonical form
    pub carries: Vec<T>,

    // We will check that expr - quotient * modulus = 0, which imples expr is 0 mod modulus.
    // quotient can be negative, and this means there is no unique way to represent it as limbs but
    // it's fine. Each limb will be range-checked to be in [-2^limb_bits, 2^limb_bits).
    pub quotient: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct CheckCarryModToZeroSubAir {
    pub modulus_limbs: Vec<usize>,

    pub check_carry_to_zero: CheckCarryToZeroSubAir,
}

impl CheckCarryModToZeroSubAir {
    pub fn new(
        modulus: BigUint,
        limb_bits: usize,
        range_checker_bus: BusIndex,
        decomp: usize,
    ) -> Self {
        let check_carry_to_zero = CheckCarryToZeroSubAir::new(limb_bits, range_checker_bus, decomp);
        let modulus_limbs = big_uint_to_limbs(&modulus, limb_bits);
        Self {
            modulus_limbs,
            check_carry_to_zero,
        }
    }
}

impl CheckCarryModToZeroSubAir {
    fn eval_with_quotient_limb_bits_fn<AB, Bits>(
        &self,
        builder: &mut AB,
        (expr, cols, is_valid): (
            OverflowInt<AB::Expr>,
            CheckCarryModToZeroCols<AB::Var>,
            AB::Expr,
        ),
        quotient_limb_bits: Bits,
    ) where
        AB: InteractionBuilder,
        Bits: Fn(usize) -> usize,
    {
        let CheckCarryModToZeroCols { quotient, carries } = cols;
        let mut max_q_limb_bits = 0;
        for (i, &q) in quotient.iter().enumerate() {
            let bits = quotient_limb_bits(i);
            max_q_limb_bits = max_q_limb_bits.max(bits);
            let q_offset = AB::F::from_usize(1 << bits);
            range_check(
                builder,
                self.check_carry_to_zero.range_checker_bus,
                self.check_carry_to_zero.decomp,
                bits + 1,
                q + q_offset,
                is_valid.clone(),
            );
        }
        let q_limbs = quotient.iter().map(|&x| x.into()).collect();
        let overflow_q = OverflowInt::<AB::Expr>::from_computed_limbs(
            q_limbs,
            1 << max_q_limb_bits,
            max_q_limb_bits + 1,
        );
        let p_limbs = self
            .modulus_limbs
            .iter()
            .map(|&x| AB::Expr::from_usize(x))
            .collect_vec();
        let overflow_p =
            OverflowInt::from_unsigned_limbs(p_limbs, self.check_carry_to_zero.limb_bits);

        let expr = expr - overflow_q * overflow_p;
        self.check_carry_to_zero
            .eval(builder, (expr, CheckCarryToZeroCols { carries }, is_valid));
    }

    pub fn eval_with_quotient_limb_bits<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        ctx: (
            OverflowInt<AB::Expr>,
            CheckCarryModToZeroCols<AB::Var>,
            AB::Expr,
        ),
        quotient_limb_bits: &[usize],
    ) {
        assert_eq!(ctx.1.quotient.len(), quotient_limb_bits.len());
        self.eval_with_quotient_limb_bits_fn(builder, ctx, |i| quotient_limb_bits[i]);
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for CheckCarryModToZeroSubAir {
    /// `(expr, cols, is_valid)`
    type AirContext<'a>
        = (
        OverflowInt<AB::Expr>,
        CheckCarryModToZeroCols<AB::Var>,
        AB::Expr,
    )
    where
        AB::Var: 'a,
        AB::Expr: 'a,
        AB: 'a;

    /// Assumes that the parent chip has already asserted `is_valid` is to be boolean.
    /// This is to avoid duplicating that constraint since this subair's eval method is
    /// often called multiple times from the parent air.
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (expr, cols, is_valid): (
            OverflowInt<AB::Expr>,
            CheckCarryModToZeroCols<AB::Var>,
            AB::Expr,
        ),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        self.eval_with_quotient_limb_bits_fn(builder, (expr, cols, is_valid), |_| {
            self.check_carry_to_zero.limb_bits
        });
    }
}
