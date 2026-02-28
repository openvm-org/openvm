use std::sync::OnceLock;

use halo2_base::{
    AssignedValue, Context,
    QuantumCell::Constant,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
    utils::{biguint_to_fe, fe_to_biguint},
};
use num_bigint::BigUint;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Bn254Scalar, Digest as NativeDigest,
        F as NativeF, Transcript as NativeTranscript, default_transcript,
    },
    openvm_stark_backend::{
        FiatShamirTranscript,
        p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64},
    },
};
use zkhash::{
    ark_ff::{BigInteger as _, PrimeField as _},
    fields::bn256::FpBN256 as ArkBn254,
    poseidon2::poseidon2_instance_bn256::{MAT_DIAG3_M_1, RC3},
};

use crate::{circuit::Fr, utils::bits_for_u64};

use super::baby_bear::{
    BABY_BEAR_BITS, BABY_BEAR_MODULUS_U64, BabyBearArithmeticGadgets, BabyBearExtVar, BabyBearVar,
};

const DIGEST_WIDTH: usize = 1;
const POSEIDON2_WIDTH: usize = 3;
const POSEIDON2_RATE: usize = 2;
const NUM_SPLIT_LIMBS: usize = 3;
const POSEIDON2_ROUNDS_F: usize = 8;
const POSEIDON2_ROUNDS_P: usize = 56;
const MULTI_FIELD32_RATE: usize = 16;
const MULTI_FIELD32_NUM_F_ELMS: usize = 8;
const MAX_U64_DIV_BABY_BEAR_PLUS_ONE: u64 =
    ((u64::MAX as u128 / BABY_BEAR_MODULUS_U64 as u128) + 1) as u64;

#[derive(Clone, Debug)]
struct Poseidon2Params {
    external_rc: Vec<[Fr; POSEIDON2_WIDTH]>,
    internal_rc: Vec<Fr>,
    mat_internal_diag_m_1: [Fr; POSEIDON2_WIDTH],
}

#[derive(Clone, Debug)]
pub struct DigestVar {
    pub elems: [AssignedValue<Fr>; DIGEST_WIDTH],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptEvent {
    Observe(u64),
    Sample(u64),
}

#[derive(Clone, Debug)]
pub struct LoggedTranscript {
    inner: NativeTranscript,
    events: Vec<TranscriptEvent>,
}

impl Default for LoggedTranscript {
    fn default() -> Self {
        Self {
            inner: default_transcript(),
            events: Vec::new(),
        }
    }
}

impl LoggedTranscript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> &[TranscriptEvent] {
        &self.events
    }

    pub fn into_events(self) -> Vec<TranscriptEvent> {
        self.events
    }
}

impl FiatShamirTranscript<NativeConfig> for LoggedTranscript {
    fn observe(&mut self, value: NativeF) {
        self.events
            .push(TranscriptEvent::Observe(value.as_canonical_u64()));
        self.inner.observe(value);
    }

    fn sample(&mut self) -> NativeF {
        let sampled = self.inner.sample();
        self.events
            .push(TranscriptEvent::Sample(sampled.as_canonical_u64()));
        sampled
    }

    fn observe_commit(&mut self, digest: NativeDigest) {
        self.inner.observe_commit(digest);
        for packed in digest {
            for limb in split_bn254_to_babybear_u64(packed) {
                self.events.push(TranscriptEvent::Observe(limb));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TranscriptGadget {
    sponge_state: [AssignedValue<Fr>; POSEIDON2_WIDTH],
    input_buffer: Vec<BabyBearVar>,
    output_buffer: Vec<BabyBearVar>,
}

fn bn254_to_halo2(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

fn split_bn254_to_babybear_u64(value: Bn254Scalar) -> [u64; NUM_SPLIT_LIMBS] {
    let digits = value.as_canonical_biguint().to_u64_digits();
    core::array::from_fn(|i| {
        let limb = digits.get(i).copied().unwrap_or(0);
        NativeF::from_u64(limb).as_canonical_u64()
    })
}

fn poseidon2_params() -> &'static Poseidon2Params {
    static PARAMS: OnceLock<Poseidon2Params> = OnceLock::new();
    PARAMS.get_or_init(|| {
        let mut round_constants: Vec<[Fr; POSEIDON2_WIDTH]> = RC3
            .iter()
            .map(|round| {
                round
                    .iter()
                    .copied()
                    .map(|value: ArkBn254| {
                        let bytes = value.into_bigint().to_bytes_le();
                        let big = BigUint::from_bytes_le(&bytes);
                        biguint_to_fe(&big)
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("RC3 must have width-3 round constants")
            })
            .collect();

        let rounds_f_beginning = POSEIDON2_ROUNDS_F / 2;
        let internal_end = rounds_f_beginning + POSEIDON2_ROUNDS_P;
        let internal_rc = round_constants
            .drain(rounds_f_beginning..internal_end)
            .map(|round| round[0])
            .collect::<Vec<_>>();

        Poseidon2Params {
            external_rc: round_constants,
            internal_rc,
            mat_internal_diag_m_1: MAT_DIAG3_M_1
                .iter()
                .copied()
                .map(|value| {
                    let bytes = value.into_bigint().to_bytes_le();
                    let big = BigUint::from_bytes_le(&bytes);
                    biguint_to_fe(&big)
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("MAT_DIAG3_M_1 must contain exactly three constants"),
        }
    })
}

fn split_high_bound() -> u64 {
    static SPLIT_HIGH_BOUND: OnceLock<u64> = OnceLock::new();
    *SPLIT_HIGH_BOUND.get_or_init(|| {
        let modulus = fe_to_biguint(&(Fr::from(0u64) - Fr::from(1u64))) + BigUint::from(1u64);
        let two_192 = BigUint::from(1u64) << 192;
        let max_high = (&modulus - &two_192) >> 192;
        u64::try_from(max_high)
            .expect("BN254 split high-limb bound must fit in u64 for three 64-bit limbs")
    })
}

fn decompose_packed_bn254_to_split_limbs(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    packed: AssignedValue<Fr>,
) -> [AssignedValue<Fr>; NUM_SPLIT_LIMBS] {
    let packed_big = fe_to_biguint(packed.value());
    let digits = packed_big.to_u64_digits();

    let limb_vals = [
        digits.get(0).copied().unwrap_or(0),
        digits.get(1).copied().unwrap_or(0),
        digits.get(2).copied().unwrap_or(0),
    ];
    let high_val_big = packed_big >> 192;
    let high_val = u64::try_from(high_val_big)
        .expect("packed BN254 high limb should fit in u64 under canonical decomposition");

    let limbs = limb_vals.map(|limb| {
        let cell = ctx.load_witness(Fr::from(limb));
        range.range_check(ctx, cell, 64);
        cell
    });

    let high = ctx.load_witness(Fr::from(high_val));
    range.range_check(ctx, high, bits_for_u64(split_high_bound()));
    range.check_less_than_safe(ctx, high, split_high_bound() + 1);

    let gate = range.gate();
    let pow_64 = biguint_to_fe(&(BigUint::from(1u64) << 64));
    let pow_128 = biguint_to_fe(&(BigUint::from(1u64) << 128));
    let pow_192 = biguint_to_fe(&(BigUint::from(1u64) << 192));

    let limb1 = gate.mul(ctx, limbs[1], Constant(pow_64));
    let limb2 = gate.mul(ctx, limbs[2], Constant(pow_128));
    let high_scaled = gate.mul(ctx, high, Constant(pow_192));
    let lower = gate.add(ctx, limbs[0], limb1);
    let upper = gate.add(ctx, limb2, high_scaled);
    let recomposed = gate.add(ctx, lower, upper);
    ctx.constrain_equal(&packed, &recomposed);

    limbs
}

fn reduce_assigned_limb_to_babybear(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    limb: AssignedValue<Fr>,
) -> BabyBearVar {
    let limb_u64 = fe_to_biguint(limb.value())
        .to_u64_digits()
        .first()
        .copied()
        .unwrap_or(0);
    let quotient = limb_u64 / BABY_BEAR_MODULUS_U64;
    let remainder = limb_u64 % BABY_BEAR_MODULUS_U64;

    let remainder_var = baby_bear.load_witness(ctx, range, remainder);
    let quotient_cell = ctx.load_witness(Fr::from(quotient));
    range.check_less_than_safe(ctx, quotient_cell, MAX_U64_DIV_BABY_BEAR_PLUS_ONE);

    let gate = range.gate();
    let recomposed = gate.mul_add(
        ctx,
        quotient_cell,
        Constant(Fr::from(BABY_BEAR_MODULUS_U64)),
        remainder_var.cell,
    );
    ctx.constrain_equal(&limb, &recomposed);

    remainder_var
}

pub fn split_assigned_bn254_to_babybear_limbs(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    packed: AssignedValue<Fr>,
) -> [BabyBearVar; NUM_SPLIT_LIMBS] {
    let baby_bear = BabyBearArithmeticGadgets;
    let limbs = decompose_packed_bn254_to_split_limbs(ctx, range, packed);
    core::array::from_fn(|idx| reduce_assigned_limb_to_babybear(ctx, range, &baby_bear, limbs[idx]))
}

fn poseidon2_x_pow5(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    value: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    let value_sq = gate.mul(ctx, value, value);
    let value_4 = gate.mul(ctx, value_sq, value_sq);
    gate.mul(ctx, value, value_4)
}

fn poseidon2_matmul_external(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    state: &mut [AssignedValue<Fr>; POSEIDON2_WIDTH],
) {
    let sum = gate.sum(ctx, state.iter().cloned());
    for value in state.iter_mut() {
        *value = gate.add(ctx, *value, sum);
    }
}

fn poseidon2_matmul_internal(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    state: &mut [AssignedValue<Fr>; POSEIDON2_WIDTH],
    diag_m_1: [Fr; POSEIDON2_WIDTH],
) {
    let sum = gate.sum(ctx, state.iter().cloned());
    for (idx, value) in state.iter_mut().enumerate() {
        *value = gate.mul_add(ctx, *value, Constant(diag_m_1[idx]), sum);
    }
}

pub(crate) fn poseidon2_permute_bn254_state(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    mut state: [AssignedValue<Fr>; POSEIDON2_WIDTH],
) -> [AssignedValue<Fr>; POSEIDON2_WIDTH] {
    let gate = range.gate();
    let params = poseidon2_params();
    let rounds_f_beginning = POSEIDON2_ROUNDS_F / 2;

    poseidon2_matmul_external(ctx, gate, &mut state);

    for round in 0..rounds_f_beginning {
        for (idx, value) in state.iter_mut().enumerate() {
            *value = gate.add(ctx, *value, Constant(params.external_rc[round][idx]));
            *value = poseidon2_x_pow5(ctx, gate, *value);
        }
        poseidon2_matmul_external(ctx, gate, &mut state);
    }

    for round in 0..POSEIDON2_ROUNDS_P {
        state[0] = gate.add(ctx, state[0], Constant(params.internal_rc[round]));
        state[0] = poseidon2_x_pow5(ctx, gate, state[0]);
        poseidon2_matmul_internal(ctx, gate, &mut state, params.mat_internal_diag_m_1);
    }

    for round in rounds_f_beginning..POSEIDON2_ROUNDS_F {
        for (idx, value) in state.iter_mut().enumerate() {
            *value = gate.add(ctx, *value, Constant(params.external_rc[round][idx]));
            *value = poseidon2_x_pow5(ctx, gate, *value);
        }
        poseidon2_matmul_external(ctx, gate, &mut state);
    }

    state
}

pub(crate) fn reduce_32_cells(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    values: &[AssignedValue<Fr>],
) -> AssignedValue<Fr> {
    let gate = range.gate();
    let base = Fr::from(1u64 << 32);
    let mut power = Fr::from(1u64);
    let mut acc = ctx.load_constant(Fr::from(0u64));

    for value in values {
        acc = gate.mul_add(ctx, *value, Constant(power), acc);
        power *= base;
    }
    acc
}

pub(crate) fn hash_babybear_slice_to_digest(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    values: &[BabyBearVar],
) -> AssignedValue<Fr> {
    let mut state = core::array::from_fn(|_| ctx.load_constant(Fr::from(0u64)));
    for block_chunk in values.chunks(MULTI_FIELD32_RATE) {
        for (chunk_id, chunk) in block_chunk.chunks(MULTI_FIELD32_NUM_F_ELMS).enumerate() {
            let cells = chunk.iter().map(|value| value.cell).collect::<Vec<_>>();
            state[chunk_id] = reduce_32_cells(ctx, range, &cells);
        }
        state = poseidon2_permute_bn254_state(ctx, range, state);
    }
    state[0]
}

pub(crate) fn compress_bn254_digests(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    left: AssignedValue<Fr>,
    right: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    let zero = ctx.load_constant(Fr::from(0u64));
    let state = poseidon2_permute_bn254_state(ctx, range, [left, right, zero]);
    state[0]
}

impl TranscriptGadget {
    pub fn new(ctx: &mut Context<Fr>) -> Self {
        let sponge_state = core::array::from_fn(|_| ctx.load_constant(Fr::from(0u64)));
        Self {
            sponge_state,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    pub fn load_digest_witness(ctx: &mut Context<Fr>, digest: NativeDigest) -> DigestVar {
        DigestVar {
            elems: core::array::from_fn(|i| ctx.load_witness(bn254_to_halo2(digest[i]))),
        }
    }

    fn permute_state(&mut self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) {
        self.sponge_state = poseidon2_permute_bn254_state(ctx, range, self.sponge_state);
    }

    fn reduce_32(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        values: &[BabyBearVar],
    ) -> AssignedValue<Fr> {
        let cells = values.iter().map(|value| value.cell).collect::<Vec<_>>();
        reduce_32_cells(ctx, range, &cells)
    }

    fn split_state_to_babybear(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        packed: AssignedValue<Fr>,
    ) -> [BabyBearVar; NUM_SPLIT_LIMBS] {
        let limbs = decompose_packed_bn254_to_split_limbs(ctx, range, packed);
        core::array::from_fn(|idx| {
            reduce_assigned_limb_to_babybear(ctx, range, baby_bear, limbs[idx])
        })
    }

    fn duplexing(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
    ) {
        assert!(
            self.input_buffer.len() <= NUM_SPLIT_LIMBS * POSEIDON2_RATE,
            "input buffer exceeds transcript absorb rate"
        );

        for (idx, chunk) in self.input_buffer.chunks(NUM_SPLIT_LIMBS).enumerate() {
            self.sponge_state[idx] = self.reduce_32(ctx, range, chunk);
        }
        self.input_buffer.clear();

        self.permute_state(ctx, range);

        self.output_buffer.clear();
        for packed in self.sponge_state {
            let parts = self.split_state_to_babybear(ctx, range, baby_bear, packed);
            self.output_buffer.extend(parts);
        }
    }

    pub fn observe(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        value: &BabyBearVar,
    ) {
        self.output_buffer.clear();
        self.input_buffer.push(value.clone());
        if self.input_buffer.len() == NUM_SPLIT_LIMBS * POSEIDON2_RATE {
            self.duplexing(ctx, range, baby_bear);
        }
    }

    pub fn observe_ext(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        value: &BabyBearExtVar,
    ) {
        for coeff in &value.coeffs {
            self.observe(ctx, range, baby_bear, coeff);
        }
    }

    pub fn observe_commit(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        digest: &DigestVar,
    ) {
        for packed in &digest.elems {
            let limbs = self.split_state_to_babybear(ctx, range, baby_bear, *packed);
            for limb in &limbs {
                self.observe(ctx, range, baby_bear, limb);
            }
        }
    }

    pub fn sample(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
    ) -> BabyBearVar {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing(ctx, range, baby_bear);
        }

        self.output_buffer
            .pop()
            .expect("transcript output buffer must be non-empty after duplexing")
    }

    pub fn sample_ext(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
    ) -> BabyBearExtVar {
        let coeffs = core::array::from_fn(|_| self.sample(ctx, range, baby_bear));
        BabyBearExtVar { coeffs }
    }

    pub fn sample_bits(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        bits: usize,
    ) -> AssignedValue<Fr> {
        assert!(
            bits < (u32::BITS as usize),
            "sample_bits requires bits < 32: {bits}"
        );
        assert!(
            (1u64 << bits) < BABY_BEAR_MODULUS_U64,
            "sample_bits requires (1 << bits) < modulus: bits={bits}"
        );

        let sampled = self.sample(ctx, range, baby_bear);
        if bits == 0 {
            return ctx.load_constant(Fr::from(0u64));
        }

        let (_, rem) = range.div_mod(
            ctx,
            sampled.cell,
            BigUint::from(1u64) << bits,
            BABY_BEAR_BITS,
        );
        range.range_check(ctx, rem, bits);
        rem
    }

    pub fn check_witness(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        bits: usize,
        witness: &BabyBearVar,
    ) -> AssignedValue<Fr> {
        if bits == 0 {
            return ctx.load_constant(Fr::from(1u64));
        }

        self.observe(ctx, range, baby_bear, witness);
        let sampled_bits = self.sample_bits(ctx, range, baby_bear, bits);
        range.gate().is_zero(ctx, sampled_bits)
    }

}

pub fn constrain_transcript_events(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    events: &[TranscriptEvent],
) -> TranscriptReplay {
    let baby_bear = BabyBearArithmeticGadgets;
    let gate = range.gate();
    let mut transcript = TranscriptGadget::new(ctx);
    let zero = ctx.load_constant(Fr::from(0u64));
    let one = ctx.load_constant(Fr::from(1u64));
    let mut replay = TranscriptReplay {
        events: Vec::with_capacity(events.len()),
        observes: Vec::new(),
        samples: Vec::new(),
    };

    for event in events {
        match event {
            TranscriptEvent::Observe(value) => {
                let observed = baby_bear.load_witness(ctx, range, *value);
                transcript.observe(ctx, range, &baby_bear, &observed);
                let is_sample = ctx.load_witness(Fr::from(0u64));
                gate.assert_bit(ctx, is_sample);
                ctx.constrain_equal(&is_sample, &zero);
                replay.observes.push(observed.cell);
                replay.events.push(AssignedTranscriptEvent {
                    is_sample,
                    value: observed.cell,
                });
            }
            TranscriptEvent::Sample(expected) => {
                let sampled = transcript.sample(ctx, range, &baby_bear);
                gate.assert_is_const(ctx, &sampled.cell, &Fr::from(*expected));
                let is_sample = ctx.load_witness(Fr::from(1u64));
                gate.assert_bit(ctx, is_sample);
                ctx.constrain_equal(&is_sample, &one);
                replay.samples.push(sampled.cell);
                replay.events.push(AssignedTranscriptEvent {
                    is_sample,
                    value: sampled.cell,
                });
            }
        }
    }

    replay
}

#[derive(Clone, Debug)]
pub struct AssignedTranscriptEvent {
    pub is_sample: AssignedValue<Fr>,
    pub value: AssignedValue<Fr>,
}

#[derive(Clone, Debug)]
pub struct TranscriptReplay {
    pub events: Vec<AssignedTranscriptEvent>,
    pub observes: Vec<AssignedValue<Fr>>,
    pub samples: Vec<AssignedValue<Fr>>,
}

#[cfg(test)]
mod tests;
