use core::array;
#[cfg(test)]
use std::{cell::RefCell, vec::Vec};

use halo2_base::{
    gates::range::RangeChip, halo2_proofs::halo2curves::bn256::Fr, safe_types::SafeBool,
    AssignedValue, Context,
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
    field::baby_bear::{BabyBearChip, BabyBearWire, ReducedBabyBearWire},
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

// irred poly is x^4 - 11
#[derive(Clone)]
pub struct BabyBearExt4Chip {
    pub base: BabyBearChip,
}

/// Generic over the cell representation `F`; see [`BabyBearWire`].
#[derive(Copy, Clone, Debug)]
pub struct BabyBearExt4Wire<F = AssignedValue<Fr>>(pub [BabyBearWire<F>; 4]);

/// An extension-field wire whose BabyBear basis coefficients are all reduced.
///
/// This is the extension-field analogue of `ReducedBabyBearWire`: it is safe for
/// transcript/hash absorption coefficient-by-coefficient. Converting via
/// `BabyBearExt4Wire::from` drops that evidence when the value is used by arithmetic
/// helpers.
#[derive(Copy, Clone, Debug)]
pub struct ReducedBabyBearExt4Wire<F = AssignedValue<Fr>>([ReducedBabyBearWire<F>; 4]);
pub type BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

impl BabyBearExt4Wire {
    pub fn to_extension_field(&self) -> BabyBearExt4 {
        BabyBearExt4::from_basis_coefficients_fn(|i| self.0[i].to_baby_bear())
    }
}

impl<F> ReducedBabyBearExt4Wire<F> {
    pub fn coeffs(&self) -> &[ReducedBabyBearWire<F>; 4] {
        &self.0
    }
    
    /// Wraps coefficient wires in canonicality evidence. Callers must guarantee each
    /// coefficient is constrained to `[0, p)`; this adds no constraints.
    #[cfg(feature = "halo2-gpu")]
    pub(crate) fn assume_reduced(coeffs: [ReducedBabyBearWire<F>; 4]) -> Self {
        ReducedBabyBearExt4Wire(coeffs)
    }
}

impl<F> From<ReducedBabyBearExt4Wire<F>> for BabyBearExt4Wire<F> {
    /// Drops the canonicality evidence and returns the underlying arithmetic wire.
    fn from(wire: ReducedBabyBearExt4Wire<F>) -> Self {
        BabyBearExt4Wire(wire.0.map(BabyBearWire::from))
    }
}

impl<F: Copy> From<&ReducedBabyBearExt4Wire<F>> for BabyBearExt4Wire<F> {
    fn from(wire: &ReducedBabyBearExt4Wire<F>) -> Self {
        (*wire).into()
    }
}

impl BabyBearExt4Chip {
    pub fn new(base_chip: BabyBearChip) -> Self {
        BabyBearExt4Chip { base: base_chip }
    }

    /// Loads each BabyBear coefficient and constrains only that its assigned
    /// advice cell fits in 31 bits.
    ///
    /// The Rust input is canonicalized for the honest witness assignment, but the
    /// circuit does not prove each advice cell is `< p`. Use
    /// `load_reduced_witness` for transcript/hash inputs.
    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBearExt4) -> BabyBearExt4Wire {
        let coeffs = value.as_basis_coefficients_slice();
        BabyBearExt4Wire(array::from_fn(|i| self.base.load_witness(ctx, coeffs[i])))
    }

    /// Loads each coefficient and constrains it to the canonical BabyBear range.
    pub fn load_reduced_witness(
        &self,
        ctx: &mut Context<Fr>,
        value: BabyBearExt4,
    ) -> ReducedBabyBearExt4Wire {
        let coeffs = value.as_basis_coefficients_slice();
        ReducedBabyBearExt4Wire(array::from_fn(|i| {
            self.base.load_reduced_witness(ctx, coeffs[i])
        }))
    }

    /// Loads canonical BabyBear constants for each coefficient and returns them
    /// with reduced type evidence.
    pub fn load_reduced_constant(
        &self,
        ctx: &mut Context<Fr>,
        value: BabyBearExt4,
    ) -> ReducedBabyBearExt4Wire {
        let coeffs = value.as_basis_coefficients_slice();
        // Constants are canonical by construction.
        ReducedBabyBearExt4Wire(array::from_fn(|i| {
            self.base.load_reduced_constant(ctx, coeffs[i])
        }))
    }
    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: BabyBearExt4) -> BabyBearExt4Wire {
        let coeffs = value.as_basis_coefficients_slice();
        BabyBearExt4Wire(array::from_fn(|i| self.base.load_constant(ctx, coeffs[i])))
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

    /// Fused `a * b + c` where `b` is a base-field scalar.
    /// Uses `mul_add` gates to save cells vs separate `scalar_mul` + `add`.
    pub fn scalar_mul_add(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        b: BabyBearWire,
        c: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
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
        cond: SafeBool<Fr>,
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
        guarded_debug_assert_eq!(
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
        let b_inv_val = b_val.try_inverse().unwrap();
        // Constrain b is non-zero by checking b * b_inv == 1
        let b_inv = self.load_witness(ctx, b_inv_val);
        let one = self.load_constant(ctx, BinomialExtensionField::<BabyBear, 4>::ONE);
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

    pub fn reduce_max_bits(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        BabyBearExt4Wire(
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

    pub fn zero(&self, ctx: &mut Context<Fr>) -> BabyBearExt4Wire {
        self.from_base_const(ctx, BabyBear::ZERO)
    }

    pub fn from_base_const(&self, ctx: &mut Context<Fr>, value: BabyBear) -> BabyBearExt4Wire {
        let base_val = self.base.load_constant(ctx, value);
        #[cfg(test)]
        RECORDED_EXT_BASE_CONSTS.with(|records| {
            records.borrow_mut().push(RecordedExtBaseConst {
                constant: value.as_canonical_u64(),
                cell: base_val.value,
            });
        });
        let z = self.base.load_constant(ctx, BabyBear::ZERO);
        BabyBearExt4Wire([base_val, z, z, z])
    }

    pub fn from_base_var(&self, ctx: &mut Context<Fr>, value: BabyBearWire) -> BabyBearExt4Wire {
        let z = self.base.load_constant(ctx, BabyBear::ZERO);
        BabyBearExt4Wire([value, z, z, z])
    }

    pub fn mul_base_const(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        c: BabyBear,
    ) -> BabyBearExt4Wire {
        let c_wire = self.base.load_constant(ctx, c);
        self.scalar_mul(ctx, a, c_wire)
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        self.mul(ctx, a, a)
    }

    pub fn pow_power_of_two(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearExt4Wire,
        n: usize,
    ) -> BabyBearExt4Wire {
        let mut result = a;
        for _ in 0..n {
            result = self.square(ctx, result);
        }
        result
    }
}
