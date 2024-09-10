use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint, Sign};
use num_integer::Integer;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};

use super::{Equation3, Equation5, ModularArithmeticAir, ModularArithmeticCols, OverflowInt};
use crate::{
    sub_chip::{AirConfig, LocalTraceInstructions},
    var_range::VariableRangeCheckerChip,
};
pub struct ModularMultiplicationAir {
    pub arithmetic: ModularArithmeticAir,
}

impl Deref for ModularMultiplicationAir {
    type Target = ModularArithmeticAir;

    fn deref(&self) -> &Self::Target {
        &self.arithmetic
    }
}

impl<F: Field> BaseAir<F> for ModularMultiplicationAir {
    fn width(&self) -> usize {
        self.arithmetic.width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularMultiplicationAir {
    fn eval(&self, builder: &mut AB) {
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| x * y - r;
        self.arithmetic.eval(builder, equation);
    }
}

impl AirConfig for ModularMultiplicationAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularMultiplicationAir {
    type LocalInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let raw_product = x.clone() * y.clone();
        let (q, r) = raw_product.div_mod_floor(&self.modulus);
        let q = BigInt::from_biguint(Sign::Plus, q);
        let equation: Equation5<isize, OverflowInt<isize>> = |x, y, r, p, q| x * y - p * q - r;
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
