//! Halo2 (`halo2-base`) implementation of the [`chip_traits`](crate::chip_traits) backend.
//!
//! Pure delegation to the concrete chips so that the generic circuit-construction code
//! assigns exactly the same advice cells as the original non-generic code.

use std::sync::Arc;

use halo2_base::{
    gates::{GateInstructions, RangeChip},
    halo2_proofs::halo2curves::bn256::Fr,
    AssignedValue, Context, QuantumCell,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{
    chip_traits::{
        BabyBearExt4Inst, BabyBearInst, ChipBase, DigestHashInst, GateInst, PopulateInputs,
        TranscriptInst,
    },
    field::baby_bear::{
        BabyBearChip, BabyBearExt4, BabyBearExt4Chip, BabyBearExt4Wire, BabyBearWire,
        ReducedBabyBearExt4Wire, ReducedBabyBearWire,
    },
    hash::poseidon2::{compress_bn254_digests, hash_babybear_slice_to_digest},
    transcript::{DigestWire, TranscriptChip},
};

/// Backend bundling the concrete halo2 chips and the circuit-building [`Context`]. The
/// transcript is created lazily by [`TranscriptInst::init_transcript`], mirroring
/// `TranscriptChip::new` at the start of `constrained_verify`.
pub struct Halo2Backend<'ctx> {
    ctx: &'ctx mut Context<Fr>,
    ext: BabyBearExt4Chip,
    transcript: Option<TranscriptChip>,
}

impl<'ctx> Halo2Backend<'ctx> {
    pub fn new(range: Arc<RangeChip<Fr>>, ctx: &'ctx mut Context<Fr>) -> Self {
        Self::from_ext_chip(BabyBearExt4Chip::new(BabyBearChip::new(range)), ctx)
    }

    pub fn from_ext_chip(ext: BabyBearExt4Chip, ctx: &'ctx mut Context<Fr>) -> Self {
        Self {
            ctx,
            ext,
            transcript: None,
        }
    }

    pub fn ext_chip(&self) -> &BabyBearExt4Chip {
        &self.ext
    }

    /// Escape hatch for code that needs the raw context (e.g. test pranks).
    pub fn ctx_mut(&mut self) -> &mut Context<Fr> {
        self.ctx
    }

    /// Disjoint borrows of the extension chip and the context.
    fn parts(&mut self) -> (&BabyBearExt4Chip, &mut Context<Fr>) {
        (&self.ext, &mut *self.ctx)
    }

    /// Disjoint borrows of the base-field chip and the context.
    fn bb_parts(&mut self) -> (&BabyBearChip, &mut Context<Fr>) {
        (self.ext.base(), &mut *self.ctx)
    }

    /// Disjoint borrows of the transcript chip and the context.
    fn transcript_parts(&mut self) -> (&mut TranscriptChip, &mut Context<Fr>) {
        (
            self.transcript
                .as_mut()
                .expect("transcript not initialized; call init_transcript first"),
            &mut *self.ctx,
        )
    }
}

impl ChipBase for Halo2Backend<'_> {
    type F = AssignedValue<Fr>;
}

impl PopulateInputs for Halo2Backend<'_> {
    fn load_witness(&mut self, value: Fr) -> AssignedValue<Fr> {
        self.ctx.load_witness(value)
    }

    fn bb_load_reduced_witness(&mut self, value: BabyBear) -> ReducedBabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.load_reduced_witness(ctx, value)
    }

    fn ext_load_reduced_witness(&mut self, value: BabyBearExt4) -> ReducedBabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.load_reduced_witness(ctx, value)
    }
}

impl GateInst for Halo2Backend<'_> {
    fn load_constant(&mut self, value: Fr) -> AssignedValue<Fr> {
        self.ctx.load_constant(value)
    }

    fn constrain_equal(&mut self, a: AssignedValue<Fr>, b: AssignedValue<Fr>) {
        self.ctx.constrain_equal(&a, &b);
    }

    fn select(
        &mut self,
        when_true: AssignedValue<Fr>,
        when_false: AssignedValue<Fr>,
        cond: AssignedValue<Fr>,
    ) -> AssignedValue<Fr> {
        let (bb, ctx) = self.bb_parts();
        bb.gate().select(ctx, when_true, when_false, cond)
    }

    fn select_const(
        &mut self,
        when_true: Fr,
        when_false: Fr,
        cond: AssignedValue<Fr>,
    ) -> AssignedValue<Fr> {
        let (bb, ctx) = self.bb_parts();
        bb.gate().select(
            ctx,
            QuantumCell::Constant(when_true),
            QuantumCell::Constant(when_false),
            cond,
        )
    }

    fn num_to_bits(&mut self, a: AssignedValue<Fr>, range_bits: usize) -> Vec<AssignedValue<Fr>> {
        let (bb, ctx) = self.bb_parts();
        bb.gate().num_to_bits(ctx, a, range_bits)
    }

    fn inner_product_const(
        &mut self,
        values: &[AssignedValue<Fr>],
        coeffs: &[Fr],
    ) -> AssignedValue<Fr> {
        let (bb, ctx) = self.bb_parts();
        bb.gate().inner_product(
            ctx,
            values.iter().copied(),
            coeffs.iter().copied().map(QuantumCell::Constant),
        )
    }

    fn cell_count(&self) -> usize {
        self.ctx.advice.len()
    }
}

impl BabyBearInst for Halo2Backend<'_> {
    fn bb_load_constant(&mut self, value: BabyBear) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.load_constant(ctx, value)
    }

    fn bb_load_reduced_constant(&mut self, value: BabyBear) -> ReducedBabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.load_reduced_constant(ctx, value)
    }

    fn bb_reduce(&mut self, a: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.reduce(ctx, a)
    }

    fn bb_reduce_max_bits(&mut self, a: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.reduce_max_bits(ctx, a)
    }

    fn bb_add(&mut self, a: BabyBearWire, b: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.add(ctx, a, b)
    }

    fn bb_neg(&mut self, a: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.neg(ctx, a)
    }

    fn bb_sub(&mut self, a: BabyBearWire, b: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.sub(ctx, a, b)
    }

    fn bb_mul(&mut self, a: BabyBearWire, b: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.mul(ctx, a, b)
    }

    fn bb_mul_add(&mut self, a: BabyBearWire, b: BabyBearWire, c: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.mul_add(ctx, a, b, c)
    }

    fn bb_div(&mut self, a: BabyBearWire, b: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.div(ctx, a, b)
    }

    fn bb_assert_zero(&mut self, a: BabyBearWire) {
        let (bb, ctx) = self.bb_parts();
        bb.assert_zero(ctx, a)
    }

    fn bb_assert_equal(&mut self, a: BabyBearWire, b: BabyBearWire) {
        let (bb, ctx) = self.bb_parts();
        bb.assert_equal(ctx, a, b)
    }

    fn bb_zero(&mut self) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.zero(ctx)
    }

    fn bb_one(&mut self) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.one(ctx)
    }

    fn bb_mul_const(&mut self, a: BabyBearWire, c: BabyBear) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.mul_const(ctx, a, c)
    }

    fn bb_square(&mut self, a: BabyBearWire) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.square(ctx, a)
    }

    fn bb_pow_power_of_two(&mut self, a: BabyBearWire, n: usize) -> BabyBearWire {
        let (bb, ctx) = self.bb_parts();
        bb.pow_power_of_two(ctx, a, n)
    }
}

impl BabyBearExt4Inst for Halo2Backend<'_> {
    fn ext_load_constant(&mut self, value: BabyBearExt4) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.load_constant(ctx, value)
    }

    fn ext_load_reduced_constant(&mut self, value: BabyBearExt4) -> ReducedBabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.load_reduced_constant(ctx, value)
    }

    fn ext_add(&mut self, a: BabyBearExt4Wire, b: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.add(ctx, a, b)
    }

    fn ext_neg(&mut self, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.neg(ctx, a)
    }

    fn ext_sub(&mut self, a: BabyBearExt4Wire, b: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.sub(ctx, a, b)
    }

    fn ext_scalar_mul(&mut self, a: BabyBearExt4Wire, b: BabyBearWire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.scalar_mul(ctx, a, b)
    }

    fn ext_scalar_mul_add(
        &mut self,
        a: BabyBearExt4Wire,
        b: BabyBearWire,
        c: BabyBearExt4Wire,
    ) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.scalar_mul_add(ctx, a, b, c)
    }

    fn ext_assert_zero(&mut self, a: BabyBearExt4Wire) {
        let (ext, ctx) = self.parts();
        ext.assert_zero(ctx, a)
    }

    fn ext_assert_equal(&mut self, a: BabyBearExt4Wire, b: BabyBearExt4Wire) {
        let (ext, ctx) = self.parts();
        ext.assert_equal(ctx, a, b)
    }

    fn ext_mul(&mut self, a: BabyBearExt4Wire, b: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.mul(ctx, a, b)
    }

    fn ext_div(&mut self, a: BabyBearExt4Wire, b: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.div(ctx, a, b)
    }

    fn ext_reduce_max_bits(&mut self, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.reduce_max_bits(ctx, a)
    }

    fn ext_zero(&mut self) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.zero(ctx)
    }

    fn ext_from_base_const(&mut self, value: BabyBear) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.from_base_const(ctx, value)
    }

    fn ext_from_base_var(&mut self, value: BabyBearWire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.from_base_var(ctx, value)
    }

    fn ext_mul_base_const(&mut self, a: BabyBearExt4Wire, c: BabyBear) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.mul_base_const(ctx, a, c)
    }

    fn ext_square(&mut self, a: BabyBearExt4Wire) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.square(ctx, a)
    }

    fn ext_pow_power_of_two(&mut self, a: BabyBearExt4Wire, n: usize) -> BabyBearExt4Wire {
        let (ext, ctx) = self.parts();
        ext.pow_power_of_two(ctx, a, n)
    }
}

impl DigestHashInst for Halo2Backend<'_> {
    fn hash_babybear_slice_to_digest(
        &mut self,
        values: &[ReducedBabyBearWire],
    ) -> AssignedValue<Fr> {
        let (bb, ctx) = self.bb_parts();
        hash_babybear_slice_to_digest(ctx, bb.range(), values)
    }

    fn compress_digests(
        &mut self,
        left: AssignedValue<Fr>,
        right: AssignedValue<Fr>,
    ) -> AssignedValue<Fr> {
        let (bb, ctx) = self.bb_parts();
        compress_bn254_digests(ctx, bb.range(), left, right)
    }
}

impl TranscriptInst for Halo2Backend<'_> {
    fn init_transcript(&mut self) {
        self.transcript = Some(TranscriptChip::new(self.ctx, self.ext.base().clone()));
    }

    fn observe(&mut self, value: &ReducedBabyBearWire) {
        let (transcript, ctx) = self.transcript_parts();
        transcript.observe(ctx, value)
    }

    fn observe_ext(&mut self, value: &ReducedBabyBearExt4Wire) {
        let (transcript, ctx) = self.transcript_parts();
        transcript.observe_ext(ctx, value)
    }

    fn observe_commit(&mut self, digest: &DigestWire) {
        let (transcript, ctx) = self.transcript_parts();
        transcript.observe_commit(ctx, digest)
    }

    fn sample(&mut self) -> BabyBearWire {
        let (transcript, ctx) = self.transcript_parts();
        transcript.sample(ctx)
    }

    fn sample_ext(&mut self) -> BabyBearExt4Wire {
        let (transcript, ctx) = self.transcript_parts();
        transcript.sample_ext(ctx)
    }

    fn sample_bits(&mut self, bits: usize) -> AssignedValue<Fr> {
        let (transcript, ctx) = self.transcript_parts();
        transcript.sample_bits(ctx, bits)
    }

    fn check_witness(&mut self, bits: usize, witness: &ReducedBabyBearWire) {
        let (transcript, ctx) = self.transcript_parts();
        transcript.check_witness(ctx, bits, witness)
    }

    fn transcript_load_reduced_constant(
        &mut self,
        value: BabyBear,
    ) -> ReducedBabyBearWire<Self::F> {
        let (transcript, ctx) = self.transcript_parts();
        transcript.baby_bear().load_reduced_constant(ctx, value)
    }
}
