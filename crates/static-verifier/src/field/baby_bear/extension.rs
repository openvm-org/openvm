#[cfg(test)]
use std::{cell::RefCell, vec::Vec};

use halo2_base::{
    gates::range::RangeChip, halo2_proofs::halo2curves::bn256::Fr, AssignedValue, Context,
};
use itertools::Itertools;
#[cfg(test)]
use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeField64;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{
        extension::{BinomialExtensionField, BinomiallyExtendable},
        BasedVectorSpace, Field, PrimeCharacteristicRing,
    },
    p3_baby_bear::BabyBear,
};

use crate::{
    field::baby_bear::{BabyBearChip, BabyBearWire, BABY_BEAR_EXT_DEGREE},
    utils::guarded_debug_assert_eq,
};

#[cfg(test)]
pub(crate) struct RecordedExtBaseConst {
    pub constant: u64,
    pub cell: AssignedValue<Fr>,
}

#[cfg(test)]
thread_local! {
    static RECORDED_EXT_BASE_CONSTS: RefCell<Vec<RecordedExtBaseConst>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn clear_recorded_ext_base_consts() {
    RECORDED_EXT_BASE_CONSTS.with(|records| records.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn take_recorded_ext_base_consts() -> Vec<RecordedExtBaseConst> {
    RECORDED_EXT_BASE_CONSTS.with(|records| records.borrow_mut().drain(..).collect())
}

// irred poly is x^5 - 2
#[derive(Clone)]
pub struct BabyBearExt5Chip {
    pub base: BabyBearChip,
}

#[derive(Copy, Clone, Debug)]
pub struct BabyBearExt5Wire(pub [BabyBearWire; BABY_BEAR_EXT_DEGREE]);
pub type BabyBearExt5 = BinomialExtensionField<BabyBear, BABY_BEAR_EXT_DEGREE>;

impl BabyBearExt5Wire {
    pub fn to_extension_field(&self) -> BabyBearExt5 {
        let b_val = (0..BABY_BEAR_EXT_DEGREE)
            .map(|i| self.0[i].to_baby_bear())
            .collect_vec();
        BabyBearExt5::from_basis_coefficients_slice(&b_val).unwrap()
    }
}

impl BabyBearExt5Chip {
    pub fn new(base_chip: BabyBearChip) -> Self {
        BabyBearExt5Chip { base: base_chip }
    }
    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBearExt5) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            value
                .as_basis_coefficients_slice()
                .iter()
                .map(|x| self.base.load_witness(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }
    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: BabyBearExt5) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
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
        a: BabyBearExt5Wire,
        b: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.add(ctx, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: BabyBearExt5Wire) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
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
        a: BabyBearExt5Wire,
        b: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
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
        a: BabyBearExt5Wire,
        b: BabyBearWire,
    ) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            a.0.iter()
                .map(|x| self.base.mul(ctx, *x, b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    /// Fused `a * b + c` where `b` is a base-field scalar.
    /// Uses `mul_add` gates to save cells vs separate `scalar_mul` + `add`.
    pub fn scalar_mul_add(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt5Wire,
        b: BabyBearWire,
        c: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            a.0.iter()
                .zip(c.0.iter())
                .map(|(ai, ci)| self.base.mul_add(ctx, *ai, b, *ci))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn select(
        &self,
        ctx: &mut Context<Fr>,
        cond: AssignedValue<Fr>,
        a: BabyBearExt5Wire,
        b: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.select(ctx, cond, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: BabyBearExt5Wire) {
        for x in a.0.iter() {
            self.base.assert_zero(ctx, *x);
        }
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, a: BabyBearExt5Wire, b: BabyBearExt5Wire) {
        for (a, b) in a.0.iter().zip(b.0.iter()) {
            self.base.assert_equal(ctx, *a, *b);
        }
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearExt5Wire,
        mut b: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        const MAX_COEFFS: usize = BABY_BEAR_EXT_DEGREE * 2 - 1;
        let mut coeffs = Vec::with_capacity(MAX_COEFFS);
        for s in 0..MAX_COEFFS {
            coeffs.push(self.base.special_inner_product(ctx, &mut a.0, &mut b.0, s));
        }
        let w = self.base.load_constant(
            ctx,
            <BabyBear as BinomiallyExtendable<BABY_BEAR_EXT_DEGREE>>::W,
        );
        for i in BABY_BEAR_EXT_DEGREE..MAX_COEFFS {
            coeffs[i - BABY_BEAR_EXT_DEGREE] =
                self.base
                    .mul_add(ctx, coeffs[i], w, coeffs[i - BABY_BEAR_EXT_DEGREE]);
        }
        coeffs.truncate(BABY_BEAR_EXT_DEGREE);
        let c = BabyBearExt5Wire(coeffs.try_into().unwrap());
        guarded_debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() * b.to_extension_field()
        );
        c
    }

    pub fn div(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt5Wire,
        b: BabyBearExt5Wire,
    ) -> BabyBearExt5Wire {
        let b_val = b.to_extension_field();
        let b_inv_val = b_val.try_inverse().unwrap();
        // Constrain b is non-zero by checking b * b_inv == 1
        let b_inv = self.load_witness(ctx, b_inv_val);
        let one = self.load_constant(
            ctx,
            BinomialExtensionField::<BabyBear, BABY_BEAR_EXT_DEGREE>::ONE,
        );
        let inv_prod = self.mul(ctx, b, b_inv);
        self.assert_equal(ctx, inv_prod, one);

        // Constrain a = b * c (mod p)
        let c = self.load_witness(ctx, a.to_extension_field() * b_inv_val);
        let prod = self.mul(ctx, b, c);
        self.assert_equal(ctx, a, prod);

        guarded_debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() / b.to_extension_field()
        );
        c
    }

    pub fn reduce_max_bits(&self, ctx: &mut Context<Fr>, a: BabyBearExt5Wire) -> BabyBearExt5Wire {
        BabyBearExt5Wire(
            a.0.into_iter()
                .map(|x| self.base.reduce_max_bits(ctx, x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    pub fn base(&self) -> &BabyBearChip {
        &self.base
    }

    pub fn range(&self) -> &RangeChip<Fr> {
        self.base.range()
    }

    pub fn zero(&self, ctx: &mut Context<Fr>) -> BabyBearExt5Wire {
        self.from_base_const(ctx, BabyBear::ZERO)
    }

    pub fn from_base_const(&self, ctx: &mut Context<Fr>, value: BabyBear) -> BabyBearExt5Wire {
        let base_val = self.base.load_constant(ctx, value);
        #[cfg(test)]
        RECORDED_EXT_BASE_CONSTS.with(|records| {
            records.borrow_mut().push(RecordedExtBaseConst {
                constant: value.as_canonical_u64(),
                cell: base_val.value,
            });
        });
        let z = self.base.load_constant(ctx, BabyBear::ZERO);
        BabyBearExt5Wire([base_val, z, z, z, z])
    }

    pub fn from_base_var(&self, ctx: &mut Context<Fr>, value: BabyBearWire) -> BabyBearExt5Wire {
        let z = self.base.load_constant(ctx, BabyBear::ZERO);
        BabyBearExt5Wire([value, z, z, z, z])
    }

    pub fn mul_base_const(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt5Wire,
        c: BabyBear,
    ) -> BabyBearExt5Wire {
        let c_wire = self.base.load_constant(ctx, c);
        self.scalar_mul(ctx, a, c_wire)
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: BabyBearExt5Wire) -> BabyBearExt5Wire {
        self.mul(ctx, a, a)
    }

    pub fn pow_power_of_two(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt5Wire,
        n: usize,
    ) -> BabyBearExt5Wire {
        let mut result = a;
        for _ in 0..n {
            result = self.square(ctx, result);
        }
        result
    }
}
