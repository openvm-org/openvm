use std::{ops::Deref, sync::Arc};

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};
use p3_matrix::Matrix;

use super::{Equation3, Equation5, ModularArithmeticAir, ModularArithmeticCols, OverflowInt};
use crate::{
    sub_chip::{AirConfig, LocalTraceInstructions, SubAir},
    var_range::VariableRangeCheckerChip,
};

#[derive(Clone, Debug)]
pub struct ModularAdditionAir {
    pub arithmetic: ModularArithmeticAir,
}

impl Deref for ModularAdditionAir {
    type Target = ModularArithmeticAir;

    fn deref(&self) -> &Self::Target {
        &self.arithmetic
    }
}

impl AirConfig for ModularAdditionAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: Field> BaseAirWithPublicValues<F> for ModularAdditionAir {}
impl<F: Field> BaseAir<F> for ModularAdditionAir {
    fn width(&self) -> usize {
        self.arithmetic.width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularAdditionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularArithmeticCols::<AB::Var>::from_slice(
            &local,
            self.num_limbs,
            self.q_limbs,
            self.carry_limbs,
        );
        SubAir::eval(self, builder, local, ());
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for ModularAdditionAir {
    type IoView = ModularArithmeticCols<AB::Var>;
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, _aux: Self::AuxView) {
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| x + y - r;
        self.arithmetic.eval(builder, io, equation);
    }
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
