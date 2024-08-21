use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};

use super::{
    CanonicalUint, Equation3, Equation5, ModularArithmeticAir, ModularArithmeticCols, OverflowInt,
};
use crate::{
    bigint::DefaultLimbConfig,
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
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
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let r = (x.clone() + y.clone()) % self.modulus.clone();
        let q = (x.clone() + y.clone() - r.clone()) / self.modulus.clone();
        let equation: Equation5<isize, CanonicalUint<isize, DefaultLimbConfig>> =
            |x, y, r, p, q| {
                let x = OverflowInt::from(x);
                let y = OverflowInt::from(y);
                let r = OverflowInt::from(r);
                let p = OverflowInt::from(p);
                let q = OverflowInt::from(q);
                x + y - r - p * q
            };
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
