use std::sync::Arc;

use halo2_base::{halo2_proofs::halo2curves::bn256::Fr, AssignedValue, Context};
use itertools::Itertools;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{
        extension::{BinomialExtensionField, BinomiallyExtendable},
        BasedVectorSpace, Field,
    },
    p3_baby_bear::BabyBear,
};

use crate::field::baby_bear::{BabyBearChip, BabyBearWire};

// irred poly is x^4 - 11
pub struct BabyBearExt4Chip {
    pub base: Arc<BabyBearChip>,
}

#[derive(Copy, Clone, Debug)]
pub struct BabyBearExt4Wire(pub [BabyBearWire; 4]);
pub type BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

impl BabyBearExt4Wire {
    pub fn to_extension_field(&self) -> BabyBearExt4 {
        let b_val = (0..4).map(|i| self.0[i].to_baby_bear()).collect_vec();
        BabyBearExt4::from_basis_coefficients_slice(&b_val).unwrap()
    }
}

impl BabyBearExt4Chip {
    pub fn new(base_chip: Arc<BabyBearChip>) -> Self {
        BabyBearExt4Chip { base: base_chip }
    }
    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBearExt4) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            value
                .as_basis_coefficients_slice()
                .iter()
                .map(|x| self.base.load_witness(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }
    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: BabyBearExt4) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            value
                .as_basis_coefficients_slice()
                .iter()
                .map(|x| self.base.load_constant(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }
    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.add(ctx, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.iter()
                .map(|x| self.base.neg(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.sub(ctx, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn scalar_mul(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearWire,
    ) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.iter()
                .map(|x| self.base.mul(ctx, *x, b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn select(
        &self,
        ctx: &mut Context<Fr>,
        cond: AssignedValue<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.select(ctx, cond, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire) {
        for x in a.0.iter() {
            self.base.assert_zero(ctx, *x);
        }
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire, b: BabyBearExt4Wire) {
        for (a, b) in a.0.iter().zip(b.0.iter()) {
            self.base.assert_equal(ctx, *a, *b);
        }
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearExt4Wire,
        mut b: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        let mut coeffs = Vec::with_capacity(7);
        for s in 0..7 {
            coeffs.push(self.base.special_inner_product(ctx, &mut a.0, &mut b.0, s));
        }
        let w = self
            .base
            .load_constant(ctx, <BabyBear as BinomiallyExtendable<4>>::W);
        for i in 4..7 {
            coeffs[i - 4] = self.base.mul_add(ctx, coeffs[i], w, coeffs[i - 4]);
        }
        coeffs.truncate(4);
        let c = BabyBearExt4Wire(coeffs.try_into().unwrap());
        debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() * b.to_extension_field()
        );
        c
    }

    pub fn div(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        let b_val = b.to_extension_field();
        let b_inv = b_val.try_inverse().unwrap();

        let c = self.load_witness(ctx, a.to_extension_field() * b_inv);
        // constraint a = b * c
        let prod = self.mul(ctx, b, c);
        self.assert_equal(ctx, a, prod);

        debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() / b.to_extension_field()
        );
        c
    }

    pub fn reduce_max_bits(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
            a.0.into_iter()
                .map(|x| self.base.reduce_max_bits(ctx, x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}
