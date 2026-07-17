//! Backend abstraction traits for circuit construction.
//!
//! Each trait mirrors one concrete chip used by the static verifier:
//! [`GateInst`] ↔ `GateChip` (plus raw `Context` cell operations), [`BabyBearInst`] ↔
//! `BabyBearChip`, [`BabyBearExt4Inst`] ↔ `BabyBearExt4Chip`, [`TranscriptInst`] ↔
//! `TranscriptChip`, and [`DigestHashInst`] ↔ the Poseidon2 digest hashing helpers.
//! [`PopulateInputs`] is the exception: it groups the witness-loading methods used by
//! `load_proof_wire` and inherits only from [`ChipBase`].
//!
//! Circuit-construction code is generic over a single backend object `B` bounded by the
//! traits it needs, e.g. `B: TranscriptInst + GateInst`. All chip traits inherit their
//! associated types from [`ChipBase`], so `B::F` (the wire/value representation
//! abstracting `AssignedValue<Fr>`) is unambiguous and never needs to be repeated
//! per-trait in bounds. Backends own their circuit-building context (e.g. a
//! `&mut halo2_base::Context<Fr>` for the halo2 backend, or an IR builder).
//!
//! Trait methods never inspect `B::F` values on the host; all nondeterminism (witness
//! loading, inversions, decompositions) lives inside backend implementations.

use core::fmt::Debug;

use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{
    field::baby_bear::{
        BabyBearExt4, BabyBearExt4Wire, BabyBearWire, ReducedBabyBearExt4Wire, ReducedBabyBearWire,
    },
    transcript::DigestWire,
};

/// Associated types shared by all chip traits.
pub trait ChipBase {
    /// Wire/value representation abstracting `AssignedValue<Fr>`.
    type F: Copy + Debug;
}

/// Witness-loading operations needed to populate proof input wires
/// (see `stages::full_pipeline::load_proof_wire`).
pub trait PopulateInputs: ChipBase {
    fn load_witness(&mut self, value: Fr) -> Self::F;
    fn bb_load_reduced_witness(&mut self, value: BabyBear) -> ReducedBabyBearWire<Self::F>;
    fn ext_load_reduced_witness(&mut self, value: BabyBearExt4)
        -> ReducedBabyBearExt4Wire<Self::F>;
}

/// Raw `Fr`-cell operations mirroring `GateChip` and direct `Context` usage.
pub trait GateInst: ChipBase {
    fn load_constant(&mut self, value: Fr) -> Self::F;
    fn constrain_equal(&mut self, a: Self::F, b: Self::F);
    /// `if cond { when_true } else { when_false }`; `cond` must already be boolean-constrained.
    fn select(&mut self, when_true: Self::F, when_false: Self::F, cond: Self::F) -> Self::F;
    /// [`Self::select`] with constant branch values.
    fn select_const(&mut self, when_true: Fr, when_false: Fr, cond: Self::F) -> Self::F;
    /// Little-endian bit decomposition, constrained to `range_bits` bits.
    fn num_to_bits(&mut self, a: Self::F, range_bits: usize) -> Vec<Self::F>;
    /// Inner product of `values` with constant coefficients.
    fn inner_product_const(&mut self, values: &[Self::F], coeffs: &[Fr]) -> Self::F;
    /// Number of cells assigned so far (for cell profiling).
    fn cell_count(&self) -> usize;
}

/// BabyBear base-field operations mirroring `BabyBearChip`.
pub trait BabyBearInst: GateInst {
    fn bb_load_constant(&mut self, value: BabyBear) -> BabyBearWire<Self::F>;
    fn bb_load_reduced_constant(&mut self, value: BabyBear) -> ReducedBabyBearWire<Self::F>;
    fn bb_reduce(&mut self, a: BabyBearWire<Self::F>) -> BabyBearWire<Self::F>;
    fn bb_reduce_max_bits(&mut self, a: BabyBearWire<Self::F>) -> BabyBearWire<Self::F>;
    fn bb_add(
        &mut self,
        a: BabyBearWire<Self::F>,
        b: BabyBearWire<Self::F>,
    ) -> BabyBearWire<Self::F>;
    fn bb_neg(&mut self, a: BabyBearWire<Self::F>) -> BabyBearWire<Self::F>;
    fn bb_sub(
        &mut self,
        a: BabyBearWire<Self::F>,
        b: BabyBearWire<Self::F>,
    ) -> BabyBearWire<Self::F>;
    fn bb_mul(
        &mut self,
        a: BabyBearWire<Self::F>,
        b: BabyBearWire<Self::F>,
    ) -> BabyBearWire<Self::F>;
    /// `a * b + c`
    fn bb_mul_add(
        &mut self,
        a: BabyBearWire<Self::F>,
        b: BabyBearWire<Self::F>,
        c: BabyBearWire<Self::F>,
    ) -> BabyBearWire<Self::F>;
    fn bb_div(
        &mut self,
        a: BabyBearWire<Self::F>,
        b: BabyBearWire<Self::F>,
    ) -> BabyBearWire<Self::F>;
    fn bb_assert_zero(&mut self, a: BabyBearWire<Self::F>);
    fn bb_assert_equal(&mut self, a: BabyBearWire<Self::F>, b: BabyBearWire<Self::F>);
    fn bb_zero(&mut self) -> BabyBearWire<Self::F>;
    fn bb_one(&mut self) -> BabyBearWire<Self::F>;
    fn bb_mul_const(&mut self, a: BabyBearWire<Self::F>, c: BabyBear) -> BabyBearWire<Self::F>;
    fn bb_square(&mut self, a: BabyBearWire<Self::F>) -> BabyBearWire<Self::F>;
    /// `a^(2^n)`
    fn bb_pow_power_of_two(&mut self, a: BabyBearWire<Self::F>, n: usize) -> BabyBearWire<Self::F>;
}

/// BabyBear quartic-extension operations mirroring `BabyBearExt4Chip`.
pub trait BabyBearExt4Inst: BabyBearInst {
    fn ext_load_constant(&mut self, value: BabyBearExt4) -> BabyBearExt4Wire<Self::F>;
    fn ext_load_reduced_constant(
        &mut self,
        value: BabyBearExt4,
    ) -> ReducedBabyBearExt4Wire<Self::F>;
    fn ext_add(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearExt4Wire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_neg(&mut self, a: BabyBearExt4Wire<Self::F>) -> BabyBearExt4Wire<Self::F>;
    fn ext_sub(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearExt4Wire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_scalar_mul(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearWire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    /// `a * b + c` where `b` is a base-field scalar.
    fn ext_scalar_mul_add(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearWire<Self::F>,
        c: BabyBearExt4Wire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_assert_zero(&mut self, a: BabyBearExt4Wire<Self::F>);
    fn ext_assert_equal(&mut self, a: BabyBearExt4Wire<Self::F>, b: BabyBearExt4Wire<Self::F>);
    fn ext_mul(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearExt4Wire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_div(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        b: BabyBearExt4Wire<Self::F>,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_reduce_max_bits(&mut self, a: BabyBearExt4Wire<Self::F>) -> BabyBearExt4Wire<Self::F>;
    fn ext_zero(&mut self) -> BabyBearExt4Wire<Self::F>;
    fn ext_from_base_const(&mut self, value: BabyBear) -> BabyBearExt4Wire<Self::F>;
    fn ext_from_base_var(&mut self, value: BabyBearWire<Self::F>) -> BabyBearExt4Wire<Self::F>;
    fn ext_mul_base_const(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        c: BabyBear,
    ) -> BabyBearExt4Wire<Self::F>;
    fn ext_square(&mut self, a: BabyBearExt4Wire<Self::F>) -> BabyBearExt4Wire<Self::F>;
    /// `a^(2^n)`
    fn ext_pow_power_of_two(
        &mut self,
        a: BabyBearExt4Wire<Self::F>,
        n: usize,
    ) -> BabyBearExt4Wire<Self::F>;
}

/// Poseidon2 digest hashing/compression mirroring the helpers in `hash::poseidon2`.
pub trait DigestHashInst: ChipBase {
    /// Hash a slice of reduced BabyBear wires into a single Bn254 digest cell.
    fn hash_babybear_slice_to_digest(&mut self, values: &[ReducedBabyBearWire<Self::F>])
        -> Self::F;
    /// Two-to-one Poseidon2 digest compression (Merkle node).
    fn compress_digests(&mut self, left: Self::F, right: Self::F) -> Self::F;
}

/// Stateful Fiat–Shamir transcript mirroring `TranscriptChip`.
pub trait TranscriptInst: BabyBearExt4Inst {
    /// (Re)initialize the transcript sponge to the all-zero state.
    fn init_transcript(&mut self);
    fn observe(&mut self, value: &ReducedBabyBearWire<Self::F>);
    fn observe_ext(&mut self, value: &ReducedBabyBearExt4Wire<Self::F>);
    fn observe_commit(&mut self, digest: &DigestWire<Self::F>);
    fn sample(&mut self) -> BabyBearWire<Self::F>;
    fn sample_ext(&mut self) -> BabyBearExt4Wire<Self::F>;
    /// Sample and truncate to `bits` bits; returns a raw cell in `[0, 2^bits)`.
    fn sample_bits(&mut self, bits: usize) -> Self::F;
    /// Asserts that the proof-of-work `witness` passes with `bits` leading zero bits.
    fn check_witness(&mut self, bits: usize, witness: &ReducedBabyBearWire<Self::F>);

    /// unfortunately we need this to keep vk the same
    /// because transcript BabyBear chip has a cache that's not the same
    fn transcript_load_reduced_constant(&mut self, value: BabyBear)
        -> ReducedBabyBearWire<Self::F>;
}
