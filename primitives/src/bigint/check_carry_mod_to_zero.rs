use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_field::AbstractField;

use super::{
    check_carry_to_zero::{CheckCarryToZeroCols, CheckCarryToZeroSubAir},
    utils::{big_uint_to_limbs, range_check},
    OverflowInt,
};

pub struct CheckCarryModToZeroCols<T> {
    pub carries: Vec<T>,

    // We will check that expr - quotient * modulus = 0, which imples expr is 0 mod modulus.
    pub quotient: Vec<T>,
}

pub struct CheckCarryModToZeroSubAir {
    pub modulus_limbs: Vec<usize>,

    pub check_carry_to_zero: CheckCarryToZeroSubAir,
}

impl CheckCarryModToZeroSubAir {
    pub fn new(
        modulus: BigUint,
        limb_bits: usize,
        range_checker_bus: usize,
        decomp: usize,
        max_overflow_bits: usize,
    ) -> Self {
        let check_carry_to_zero =
            CheckCarryToZeroSubAir::new(limb_bits, range_checker_bus, decomp, max_overflow_bits);
        let modulus_limbs = big_uint_to_limbs(modulus, limb_bits);
        Self {
            modulus_limbs,
            check_carry_to_zero,
        }
    }

    pub fn constrain_carry_mod_to_zero<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        expr: OverflowInt<AB::Expr>,
        cols: CheckCarryModToZeroCols<AB::Var>,
    ) {
        let CheckCarryModToZeroCols { quotient, carries } = cols;
        for &q in quotient.iter() {
            range_check(
                builder,
                self.check_carry_to_zero.range_checker_bus,
                self.check_carry_to_zero.decomp,
                self.check_carry_to_zero.limb_bits,
                q,
            );
        }
        let overflow_q = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            quotient,
            self.check_carry_to_zero.limb_bits,
        );
        let p_limbs = self
            .modulus_limbs
            .iter()
            .map(|&x| AB::Expr::from_canonical_usize(x))
            .collect_vec();
        let overflow_p = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(
            p_limbs,
            self.check_carry_to_zero.limb_bits,
        );

        let expr = expr - overflow_q * overflow_p;
        self.check_carry_to_zero.constrain_carry_to_zero(
            builder,
            expr,
            CheckCarryToZeroCols { carries },
        );
    }
}
