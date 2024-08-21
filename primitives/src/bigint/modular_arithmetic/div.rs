use std::{ops::Deref, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
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
pub struct ModularDivisionAir {
    pub arithmetic: ModularArithmeticAir,
}

impl Deref for ModularDivisionAir {
    type Target = ModularArithmeticAir;

    fn deref(&self) -> &Self::Target {
        &self.arithmetic
    }
}

impl<F: Field> BaseAir<F> for ModularDivisionAir {
    fn width(&self) -> usize {
        self.arithmetic.width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularDivisionAir {
    fn eval(&self, builder: &mut AB) {
        let equation: Equation3<AB::Expr, OverflowInt<AB::Expr>> = |x, y, r| r * y - x;
        self.arithmetic.eval(builder, equation);
    }
}

impl AirConfig for ModularDivisionAir {
    type Cols<T> = ModularArithmeticCols<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularDivisionAir {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let exp = self.modulus.clone() - BigUint::from_u8(2).unwrap();
        let y_inv = y.modpow(&exp, &self.modulus);
        let r = x.clone() * y_inv.clone() % self.modulus.clone();
        let q = (y.clone() * r.clone() - x.clone()) / self.modulus.clone();
        let equation: Equation5<isize, CanonicalUint<isize, DefaultLimbConfig>> =
            |x, y, r, p, q| {
                let x = OverflowInt::from(x);
                let y = OverflowInt::from(y);
                let r = OverflowInt::from(r);
                let p = OverflowInt::from(p);
                let q = OverflowInt::from(q);
                r * y - x - p * q
            };
        self.arithmetic
            .generate_trace_row(x, y, q, r, equation, range_checker)
    }
}
