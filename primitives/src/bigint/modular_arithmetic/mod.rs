use std::{iter::repeat, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint};
use p3_field::PrimeField64;
use p3_matrix::Matrix;

use crate::{
    bigint::{
        check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
        utils::big_int_to_limbs,
        CanonicalUint, DefaultLimbConfig, OverflowInt,
    },
    var_range::VariableRangeCheckerChip,
};

pub mod add;
pub mod div;
pub mod mul;
pub mod sub;

#[cfg(test)]
mod tests;

// Op(x, y) = r (mod p), where Op is one of +, -, *, /
#[derive(Clone)]
pub struct ModularArithmeticCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub q: Vec<T>,
    pub r: Vec<T>,
    pub carries: Vec<T>,
}

impl<T: Clone> ModularArithmeticCols<T> {
    pub fn from_slice(slc: &[T], num_limbs: usize, q_limbs: usize, carry_limbs: usize) -> Self {
        // The modulus p has num_limbs limbs.
        // So the numbers (x, y, r) we operate on have num_limbs limbs.
        // The carries are for the expression will be 2 * num_limbs - 1 for mul and div, and num_limbs for add and sub.
        // q limbs will be num_limbs for mul and div, and 1 for add and sub.
        let x = slc[0..num_limbs].to_vec();
        let y = slc[num_limbs..2 * num_limbs].to_vec();
        let r = slc[2 * num_limbs..3 * num_limbs].to_vec();
        let carries = slc[3 * num_limbs..3 * num_limbs + carry_limbs].to_vec();
        let q = slc[3 * num_limbs + carry_limbs..3 * num_limbs + carry_limbs + q_limbs].to_vec();

        Self {
            x,
            y,
            q,
            r,
            carries,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.r);
        flattened.extend_from_slice(&self.carries);
        flattened.extend_from_slice(&self.q);

        flattened
    }
}

type Equation3<T, S> = fn(S, S, S) -> OverflowInt<T>;
type Equation5<T, S> = fn(S, S, S, S, S) -> OverflowInt<T>;

pub struct ModularArithmeticAir {
    pub check_carry_sub_air: CheckCarryModToZeroSubAir,
    // The modulus p
    pub modulus: BigUint,
    // The number of limbs of the big numbers we operate on. Should be the number of limbs of modulus.
    pub num_limbs: usize,
    // q and carry limbs can be different depends on the operation.
    pub q_limbs: usize,
    pub carry_limbs: usize,
    pub limb_bits: usize,
    pub range_decomp: usize,
}

impl ModularArithmeticAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modulus: BigUint,
        limb_bits: usize,
        max_overflow_bits: usize,
        num_limbs: usize,
        q_limbs: usize,
        carry_limbs: usize,
        range_bus: usize,
        range_decomp: usize,
    ) -> Self {
        let check_carry_sub_air = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_bits,
            range_bus,
            range_decomp,
            max_overflow_bits,
        );

        Self {
            check_carry_sub_air,
            modulus,
            num_limbs,
            q_limbs,
            carry_limbs,
            limb_bits,
            range_decomp,
        }
    }

    pub fn width(&self) -> usize {
        3 * self.num_limbs + self.q_limbs + self.carry_limbs
    }

    fn get_carry_min_value_abs(&self) -> usize {
        self.check_carry_sub_air
            .check_carry_to_zero
            .carry_min_value_abs
    }

    fn get_carry_bits(&self) -> usize {
        self.check_carry_sub_air.check_carry_to_zero.carry_bits
    }

    // Converting limb from an isize to a field element.
    fn to_f<F: PrimeField64>(x: isize) -> F {
        F::from_canonical_usize(x.unsigned_abs()) * if x >= 0 { F::one() } else { F::neg_one() }
    }

    pub fn eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        equation: Equation3<AB::Expr, OverflowInt<AB::Expr>>,
    ) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularArithmeticCols::<AB::Var>::from_slice(
            &local,
            self.num_limbs,
            self.q_limbs,
            self.carry_limbs,
        );

        let ModularArithmeticCols {
            x,
            y,
            q,
            r,
            carries,
        } = local;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(y, self.limb_bits);
        let r_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(r, self.limb_bits);
        let expr = equation(x_overflow, y_overflow, r_overflow);

        self.check_carry_sub_air.constrain_carry_mod_to_zero(
            builder,
            expr,
            CheckCarryModToZeroCols {
                carries,
                quotient: q,
            },
        );
    }

    pub fn generate_trace_row<F: PrimeField64>(
        &self,
        x: BigUint,
        y: BigUint,
        q: BigInt,
        r: BigUint,
        equation: Equation5<isize, OverflowInt<isize>>,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> ModularArithmeticCols<F> {
        // Quotient and result can be smaller, but padding to the desired length.
        let q_limbs: Vec<isize> = big_int_to_limbs(q.clone(), self.limb_bits)
            .iter()
            .chain(repeat(&0))
            .take(self.q_limbs)
            .copied()
            .collect();
        for &q in q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        let q_f: Vec<F> = q_limbs.iter().map(|&x| Self::to_f(x)).collect();
        let r_canonical =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(r, Some(self.num_limbs));
        let r_f: Vec<F> = r_canonical
            .limbs
            .iter()
            .map(|&x| F::from_canonical_usize(x as usize))
            .collect();

        let x_canonical =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(x, Some(self.num_limbs));
        let y_canonical =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(y, Some(self.num_limbs));
        let p_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(
            self.modulus.clone(),
            Some(self.num_limbs),
        );
        let q_overflow = OverflowInt {
            limbs: q_limbs,
            max_overflow_bits: self.limb_bits + 1,
            limb_max_abs: (1 << self.limb_bits),
        };
        let expr = equation(
            x_canonical.clone().into(),
            y_canonical.clone().into(),
            r_canonical.into(),
            p_canonical.into(),
            q_overflow,
        );
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        let carry_min_abs = self.get_carry_min_value_abs() as isize;
        for (i, &carry) in carries.iter().enumerate() {
            range_checker.add_count((carry + carry_min_abs) as u32, self.get_carry_bits());
            carries_f[i] = Self::to_f(carry);
        }

        ModularArithmeticCols {
            x: x_canonical
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            y: y_canonical
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            q: q_f,
            r: r_f,
            carries: carries_f,
        }
    }
}
