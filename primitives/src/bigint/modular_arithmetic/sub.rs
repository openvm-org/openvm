use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};
use p3_matrix::Matrix;

use super::{
    super::utils::big_uint_sub, Equation3, Equation5, ModularArithmeticAir, ModularArithmeticCols,
    OverflowInt,
};
use crate::{
    sub_chip::{AirConfig, LocalTraceInstructions, SubAir},
    var_range::VariableRangeCheckerChip,
};
pub struct ModularSubtractionAir {
    pub arithmetic: ModularArithmeticAir,
}

impl Deref for ModularSubtractionAir {
    type Target = ModularArithmeticAir;

    fn deref(&self) -> &Self::Target {
        &self.arithmetic
    }
}

impl<F: Field> BaseAir<F> for ModularSubtractionAir {
    fn width(&self) -> usize {
        self.arithmetic.width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularSubtractionAir {
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

impl<AB: InteractionBuilder> SubAir<AB> for ModularSubtractionAir {
    type IoView = ModularArithmeticCols<AB::Var>;
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, _aux: Self::AuxView) {
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| x - y - r;
        self.arithmetic.eval(builder, io, equation);
    }
}

impl AirConfig for ModularSubtractionAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularSubtractionAir {
    type LocalInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let mut xp = x.clone();
        while xp < y {
            xp += self.modulus.clone();
        }
        let r = xp - y.clone();
        let q = big_uint_sub(x.clone(), y.clone() + r.clone());
        let q = q / BigInt::from_biguint(Sign::Plus, self.modulus.clone());
        let equation: Equation5<isize, OverflowInt<isize>> = |x, y, r, p, q| x - y - r - p * q;
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
