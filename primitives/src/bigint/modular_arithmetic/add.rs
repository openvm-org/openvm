use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};

use super::{Equation3, Equation5, ModularArithmeticAir, ModularArithmeticCols, OverflowInt};
use crate::{
    sub_chip::{AirConfig, LocalTraceInstructions},
    var_range::VariableRangeCheckerChip,
};
pub struct ModularAdditionAir {
    pub arithmetic: ModularArithmeticAir,
}

impl Deref for ModularAdditionAir {
    type Target = ModularArithmeticAir;

    fn deref(&self) -> &Self::Target {
        &self.arithmetic
    }
}

impl<F: Field> BaseAir<F> for ModularAdditionAir {
    fn width(&self) -> usize {
        self.arithmetic.width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularAdditionAir {
    fn eval(&self, builder: &mut AB) {
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| x + y - r;
        self.arithmetic.eval(builder, equation);
    }
}

impl AirConfig for ModularAdditionAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularAdditionAir {
    type LocalInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let raw_sum = x.clone() + y.clone();
        let sign = if raw_sum < self.modulus {
            // x + y - r == 0
            Sign::NoSign
        } else {
            Sign::Plus
        };
        let r = raw_sum.clone() % self.modulus.clone();
        let q = BigInt::from_biguint(sign, (raw_sum - r.clone()) / self.modulus.clone());
        let equation: Equation5<isize, OverflowInt<isize>> = |x, y, r, p, q| x + y - r - p * q;
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
