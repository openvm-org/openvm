#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
use halo2_base::AssignedValue;
use halo2_base::{gates::range::RangeChip, Context};
#[cfg(test)]
use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeField64;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{EF as NativeEF, F as NativeF},
    openvm_stark_backend::p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
};

use super::base::{BabyBearChip, BabyBearWire, BABY_BEAR_EXT_DEGREE, BABY_BEAR_EXT_W_U64};
use crate::Fr;

#[derive(Copy, Clone, Debug)]
pub struct BabyBearExtWire(pub [BabyBearWire; BABY_BEAR_EXT_DEGREE]);

impl BabyBearExtWire {
    pub fn as_u64(&self) -> [u64; BABY_BEAR_EXT_DEGREE] {
        core::array::from_fn(|i| self.0[i].as_u64())
    }

    pub fn value(&self) -> NativeEF {
        NativeEF::from_basis_coefficients_fn(|i| self.0[i].value())
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
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

#[derive(Debug, Clone, Copy)]
pub struct BabyBearExtChip<'a> {
    base: BabyBearChip<'a>,
}

impl<'a> BabyBearExtChip<'a> {
    pub fn new(base: BabyBearChip<'a>) -> Self {
        Self { base }
    }

    pub fn base(&self) -> &BabyBearChip<'a> {
        &self.base
    }

    pub fn range(&self) -> &RangeChip<Fr> {
        self.base.range()
    }

    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: NativeEF) -> BabyBearExtWire {
        let coeffs = <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&value);
        BabyBearExtWire(core::array::from_fn(|i| {
            self.base.load_witness(ctx, coeffs[i])
        }))
    }

    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: NativeEF) -> BabyBearExtWire {
        let coeffs = <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&value);
        BabyBearExtWire(core::array::from_fn(|i| {
            self.base.load_constant(ctx, coeffs[i])
        }))
    }

    pub fn zero(&self, ctx: &mut Context<Fr>) -> BabyBearExtWire {
        self.load_constant(ctx, NativeEF::ZERO)
    }

    pub fn one(&self, ctx: &mut Context<Fr>) -> BabyBearExtWire {
        self.load_constant(ctx, NativeEF::ONE)
    }

    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        a: &BabyBearExtWire,
        b: &BabyBearExtWire,
    ) -> BabyBearExtWire {
        BabyBearExtWire(core::array::from_fn(|i| {
            self.base.add(ctx, &a.0[i], &b.0[i])
        }))
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        a: &BabyBearExtWire,
        b: &BabyBearExtWire,
    ) -> BabyBearExtWire {
        BabyBearExtWire(core::array::from_fn(|i| {
            self.base.sub(ctx, &a.0[i], &b.0[i])
        }))
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        a: &BabyBearExtWire,
        b: &BabyBearExtWire,
    ) -> BabyBearExtWire {
        let a0 = a.0[0];
        let a1 = a.0[1];
        let a2 = a.0[2];
        let a3 = a.0[3];
        let b0 = b.0[0];
        let b1 = b.0[1];
        let b2 = b.0[2];
        let b3 = b.0[3];

        let t0 = self.base.mul(ctx, &a0, &b0);

        let m01 = self.base.mul(ctx, &a0, &b1);
        let m10 = self.base.mul(ctx, &a1, &b0);
        let t1 = self.base.add2(ctx, m01, m10);

        let m02 = self.base.mul(ctx, &a0, &b2);
        let m11 = self.base.mul(ctx, &a1, &b1);
        let m20 = self.base.mul(ctx, &a2, &b0);
        let t2 = self.base.add3(ctx, m02, m11, m20);

        let m03 = self.base.mul(ctx, &a0, &b3);
        let m12 = self.base.mul(ctx, &a1, &b2);
        let m21 = self.base.mul(ctx, &a2, &b1);
        let m30 = self.base.mul(ctx, &a3, &b0);
        let t3 = self.base.add4(ctx, m03, m12, m21, m30);

        let m13 = self.base.mul(ctx, &a1, &b3);
        let m22 = self.base.mul(ctx, &a2, &b2);
        let m31 = self.base.mul(ctx, &a3, &b1);
        let t4 = self.base.add3(ctx, m13, m22, m31);

        let m23 = self.base.mul(ctx, &a2, &b3);
        let m32 = self.base.mul(ctx, &a3, &b2);
        let t5 = self.base.add2(ctx, m23, m32);

        let t6 = self.base.mul(ctx, &a3, &b3);

        let w = NativeF::from_u64(BABY_BEAR_EXT_W_U64);
        let wt4 = self.base.mul_const(ctx, &t4, w);
        let wt5 = self.base.mul_const(ctx, &t5, w);
        let wt6 = self.base.mul_const(ctx, &t6, w);

        BabyBearExtWire([
            self.base.add2(ctx, t0, wt4),
            self.base.add2(ctx, t1, wt5),
            self.base.add2(ctx, t2, wt6),
            t3,
        ])
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: &BabyBearExtWire) -> BabyBearExtWire {
        self.mul(ctx, a, a)
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: &BabyBearExtWire) -> BabyBearExtWire {
        let zero = self.zero(ctx);
        self.sub(ctx, &zero, a)
    }

    pub fn assert_equal(
        &self,
        ctx: &mut Context<Fr>,
        lhs: &BabyBearExtWire,
        rhs: &BabyBearExtWire,
    ) {
        for i in 0..BABY_BEAR_EXT_DEGREE {
            self.base.assert_equal(ctx, &lhs.0[i], &rhs.0[i]);
        }
    }

    pub fn from_base_const(&self, ctx: &mut Context<Fr>, constant: NativeF) -> BabyBearExtWire {
        let c0 = self.base.load_constant(ctx, constant);
        #[cfg(test)]
        RECORDED_EXT_BASE_CONSTS.with(|records| {
            records.borrow_mut().push(RecordedExtBaseConst {
                constant: constant.as_canonical_u64(),
                cell: c0.0,
            });
        });
        BabyBearExtWire(core::array::from_fn(|idx| {
            if idx == 0 {
                c0
            } else {
                self.base.zero(ctx)
            }
        }))
    }

    pub fn mul_base_const(
        &self,
        ctx: &mut Context<Fr>,
        value: &BabyBearExtWire,
        constant: NativeF,
    ) -> BabyBearExtWire {
        BabyBearExtWire(core::array::from_fn(|idx| {
            self.base.mul_const(ctx, &value.0[idx], constant)
        }))
    }

    pub fn pow_power_of_two(
        &self,
        ctx: &mut Context<Fr>,
        value: &BabyBearExtWire,
        exp_power: usize,
    ) -> BabyBearExtWire {
        let mut acc = *value;
        for _ in 0..exp_power {
            acc = self.mul(ctx, &acc, &acc);
        }
        acc
    }

    pub fn from_base_var(&self, ctx: &mut Context<Fr>, value: &BabyBearWire) -> BabyBearExtWire {
        let zero = self.base.zero(ctx);
        BabyBearExtWire(core::array::from_fn(
            |idx| {
                if idx == 0 {
                    *value
                } else {
                    zero
                }
            },
        ))
    }
}
