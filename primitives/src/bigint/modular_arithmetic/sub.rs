use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use num_traits::Zero;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};

use super::{
    CanonicalUint, DefaultLimbConfig, Equation3, Equation5, ModularArithmeticAir,
    ModularArithmeticCols, OverflowInt,
};
use crate::{
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
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
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| x - y - r;
        self.arithmetic.eval(builder, equation);
    }
}

impl AirConfig for ModularSubtractionAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularSubtractionAir {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let mut xp = x.clone();
        while xp < y {
            xp += self.modulus.clone();
        }
        let r = xp - y.clone();
        let q = BigUint::zero();
        let equation: Equation5<isize, CanonicalUint<isize, DefaultLimbConfig>> =
            |x, y, r, p, q| {
                let x = OverflowInt::from(x);
                let y = OverflowInt::from(y);
                let r = OverflowInt::from(r);
                let p = OverflowInt::from(p);
                let q = OverflowInt::from(q);
                x - y - r - p * q
            };
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
