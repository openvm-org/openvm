use std::ops::Deref;

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{EcAddCols, EcDoubleCols};
use crate::{
    bigint::{check_carry_mod_to_zero::CheckCarryModToZeroSubAir, DefaultLimbConfig, OverflowInt},
    sub_chip::AirConfig,
};

pub struct EcAirConfig {
    // e.g. secp256k1 is 2^256 - 2^32 - 977.
    pub prime: BigUint,

    // y^2 = x^3 + b. b=7 for secp256k1.
    pub b: BigUint,

    // The limb config for the EcPoint coordinates.
    pub limb_bits: usize,
    // Number of limbs of the prime and the coordinates.
    pub num_limbs: usize,

    // The subair to constrain big integer operations.
    pub check_carry: CheckCarryModToZeroSubAir,
    // Range checker decomp bits.
    pub decomp: usize,
}

pub struct EcAddUnequalAir {
    pub config: EcAirConfig,
}

impl Deref for EcAddUnequalAir {
    type Target = EcAirConfig;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

pub struct EccDoubleAir {
    pub config: EcAirConfig,
}

impl Deref for EccDoubleAir {
    type Target = EcAirConfig;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

impl EcAirConfig {
    pub fn new(
        prime: BigUint,
        b: BigUint,
        range_checker_bus: usize,
        decomp: usize,
        limb_bits: usize,
        field_element_bits: usize,
    ) -> Self {
        let num_limbs = (prime.bits() + limb_bits - 1) / limb_bits;
        let check_carry = CheckCarryModToZeroSubAir::new(
            prime.clone(),
            limb_bits,
            range_checker_bus,
            decomp,
            field_element_bits,
        );

        EcAirConfig {
            prime,
            b,
            limb_bits,
            num_limbs,
            check_carry,
            decomp,
        }
    }
}

impl<F: Field> BaseAir<F> for EcAddUnequalAir {
    fn width(&self) -> usize {
        EcAddCols::<F, DefaultLimbConfig>::width(self)
    }
}

impl AirConfig for EcAddUnequalAir {
    type Cols<T> = EcAddCols<T, DefaultLimbConfig>;
}

impl<F: Field> BaseAir<F> for EccDoubleAir {
    fn width(&self) -> usize {
        EcDoubleCols::<F, DefaultLimbConfig>::width(self)
    }
}

impl AirConfig for EccDoubleAir {
    type Cols<T> = EcDoubleCols<T, DefaultLimbConfig>;
}

impl<AB: InteractionBuilder> Air<AB> for EcAddUnequalAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = EcAddCols::<AB::Var, DefaultLimbConfig>::from_slice(&local, self.num_limbs);

        let EcAddCols { io, aux } = local;

        // λ = (y2 - y1) / (x2 - x1)
        let lambda =
            OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(aux.lambda, self.limb_bits);
        let x1: OverflowInt<AB::Expr> = io.p1.x.into();
        let x2: OverflowInt<AB::Expr> = io.p2.x.into();
        let y1: OverflowInt<AB::Expr> = io.p1.y.into();
        let y2: OverflowInt<AB::Expr> = io.p2.y.into();
        let expr = lambda.clone() * (x2.clone() - x1.clone()) - y2 + y1.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.lambda_check, aux.is_valid);

        // x3 = λ * λ - x1 - x2
        let x3: OverflowInt<AB::Expr> = io.p3.x.into();
        let expr = lambda.clone() * lambda.clone() - x1.clone() - x2.clone() - x3.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.x3_check, aux.is_valid);

        // t = y1 - λ * x1
        // y3 = -(λ * x3 + t) = -λ * x3 - y1 + λ * x1
        let y3: OverflowInt<AB::Expr> = io.p3.y.into();
        let expr = y3 + lambda.clone() * x3 + y1 - lambda * x1;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.y3_check, aux.is_valid);
    }
}

impl<AB: InteractionBuilder> Air<AB> for EccDoubleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = EcDoubleCols::<AB::Var, DefaultLimbConfig>::from_slice(&local, self.num_limbs);

        let EcDoubleCols { io, aux } = local;

        // λ = (3 * x1^2) / (2 * y1)
        let lambda =
            OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(aux.lambda, self.limb_bits);
        let x1: OverflowInt<AB::Expr> = io.p1.x.into();
        let y1: OverflowInt<AB::Expr> = io.p1.y.into();
        let two = AB::Expr::two();
        let two = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(vec![two], 2);
        let three = AB::Expr::from_canonical_usize(3);
        let three = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(vec![three], 2);
        let expr = lambda.clone() * two * y1.clone() - three * x1.clone() * x1.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.lambda_check, aux.is_valid);

        // Note: below are almost the same as add unequal, but it's just 10 lines so probably not worth to dedup.
        // x3 = λ * λ - x1 - x1
        let x3: OverflowInt<AB::Expr> = io.p2.x.into();
        let expr = lambda.clone() * lambda.clone() - x1.clone() - x1.clone() - x3.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.x3_check, aux.is_valid);
        // t = y1 - λ * x1
        // y3 = -(λ * x3 + t) = -λ * x3 - y1 + λ * x1
        let y3: OverflowInt<AB::Expr> = io.p2.y.into();
        let expr = y3 + lambda.clone() * x3 + y1 - lambda * x1;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.y3_check, aux.is_valid);
    }
}
