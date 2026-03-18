use std::sync::OnceLock;

use halo2_base::{
    gates::{range::RangeChip, GateInstructions, RangeInstructions},
    halo2_proofs::arithmetic::Field,
    utils::{biguint_to_fe, fe_to_biguint},
    AssignedValue, Context,
    QuantumCell::Constant,
};
use num_bigint::BigUint;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        default_transcript, BabyBearBn254Poseidon2Config as NativeConfig, Bn254Scalar,
        Digest as NativeDigest, Transcript as NativeTranscript, F as NativeF,
    },
    openvm_stark_backend::{
        p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64},
        FiatShamirTranscript,
    },
};

use crate::{
    field::baby_bear::{
        BabyBearChip, BabyBearExtWire, BabyBearWire, BABY_BEAR_BITS, BABY_BEAR_MODULUS_U64,
    },
    hash::poseidon2::{
        poseidon2_permute_bn254_state, reduce_32_cells, DIGEST_WIDTH, POSEIDON2_RATE,
        POSEIDON2_WIDTH,
    },
    utils::bits_for_u64,
    Fr,
};

pub(crate) const NUM_SPLIT_LIMBS: usize = 3;
const MAX_U64_DIV_BABY_BEAR_PLUS_ONE: u64 =
    ((u64::MAX as u128 / BABY_BEAR_MODULUS_U64 as u128) + 1) as u64;

#[derive(Clone, Debug)]
pub struct DigestWire {
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
    input_buffer: Vec<BabyBearWire>,
    output_buffer: Vec<BabyBearWire>,
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
        digits.first().copied().unwrap_or(0),
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
    baby_bear: &BabyBearChip<'_>,
    limb: AssignedValue<Fr>,
) -> BabyBearWire {
    let limb_u64 = fe_to_biguint(limb.value())
        .to_u64_digits()
        .first()
        .copied()
        .unwrap_or(0);
    let quotient = limb_u64 / BABY_BEAR_MODULUS_U64;
    let remainder = limb_u64 % BABY_BEAR_MODULUS_U64;

    let remainder_var = baby_bear.load_witness(ctx, NativeF::from_u64(remainder));
    let quotient_cell = ctx.load_witness(Fr::from(quotient));
    range.check_less_than_safe(ctx, quotient_cell, MAX_U64_DIV_BABY_BEAR_PLUS_ONE);

    let gate = range.gate();
    let recomposed = gate.mul_add(
        ctx,
        quotient_cell,
        Constant(Fr::from(BABY_BEAR_MODULUS_U64)),
        remainder_var.0,
    );
    ctx.constrain_equal(&limb, &recomposed);

    remainder_var
}

pub fn split_assigned_bn254_to_babybear_limbs(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    packed: AssignedValue<Fr>,
) -> [BabyBearWire; NUM_SPLIT_LIMBS] {
    let baby_bear = BabyBearChip::new(range);
    let limbs = decompose_packed_bn254_to_split_limbs(ctx, range, packed);
    core::array::from_fn(|idx| reduce_assigned_limb_to_babybear(ctx, range, &baby_bear, limbs[idx]))
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

    pub fn load_digest_witness(ctx: &mut Context<Fr>, digest: NativeDigest) -> DigestWire {
        DigestWire {
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
        values: &[BabyBearWire],
    ) -> AssignedValue<Fr> {
        let cells = values.iter().map(|value| value.0).collect::<Vec<_>>();
        reduce_32_cells(ctx, range, &cells)
    }

    fn split_state_to_babybear(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
        packed: AssignedValue<Fr>,
    ) -> [BabyBearWire; NUM_SPLIT_LIMBS] {
        let limbs = decompose_packed_bn254_to_split_limbs(ctx, range, packed);
        core::array::from_fn(|idx| {
            reduce_assigned_limb_to_babybear(ctx, range, baby_bear, limbs[idx])
        })
    }

    fn duplexing(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
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
        baby_bear: &BabyBearChip<'_>,
        value: &BabyBearWire,
    ) {
        self.output_buffer.clear();
        self.input_buffer.push(*value);
        if self.input_buffer.len() == NUM_SPLIT_LIMBS * POSEIDON2_RATE {
            self.duplexing(ctx, range, baby_bear);
        }
    }

    pub fn observe_ext(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
        value: &BabyBearExtWire,
    ) {
        for coeff in &value.0 {
            self.observe(ctx, range, baby_bear, coeff);
        }
    }

    pub fn observe_commit(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
        digest: &DigestWire,
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
        baby_bear: &BabyBearChip<'_>,
    ) -> BabyBearWire {
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
        baby_bear: &BabyBearChip<'_>,
    ) -> BabyBearExtWire {
        let coeffs = core::array::from_fn(|_| self.sample(ctx, range, baby_bear));
        BabyBearExtWire(coeffs)
    }

    pub fn sample_bits(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
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

        let (_, rem) = range.div_mod(ctx, sampled.0, BigUint::from(1u64) << bits, BABY_BEAR_BITS);
        range.range_check(ctx, rem, bits);
        rem
    }

    pub fn check_witness(
        &mut self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearChip<'_>,
        bits: usize,
        witness: &BabyBearWire,
    ) -> AssignedValue<Fr> {
        if bits == 0 {
            return ctx.load_constant(Fr::from(1u64));
        }

        self.observe(ctx, range, baby_bear, witness);
        let sampled_bits = self.sample_bits(ctx, range, baby_bear, bits);
        range.gate().is_zero(ctx, sampled_bits)
    }
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

pub fn constrain_transcript_events(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    events: &[TranscriptEvent],
) -> TranscriptReplay {
    let baby_bear = BabyBearChip::new(range);
    let gate = range.gate();
    let mut transcript = TranscriptGadget::new(ctx);
    let mut replay = TranscriptReplay {
        events: Vec::with_capacity(events.len()),
        observes: Vec::new(),
        samples: Vec::new(),
    };

    for event in events {
        match event {
            TranscriptEvent::Observe(value) => {
                let observed = baby_bear.load_witness(ctx, NativeF::from_u64(*value));
                transcript.observe(ctx, range, &baby_bear, &observed);
                let is_sample = ctx.load_zero();
                replay.observes.push(observed.0);
                replay.events.push(AssignedTranscriptEvent {
                    is_sample,
                    value: observed.0,
                });
            }
            TranscriptEvent::Sample(expected) => {
                let sampled = transcript.sample(ctx, range, &baby_bear);
                gate.assert_is_const(ctx, &sampled.0, &Fr::from(*expected));
                let is_sample = ctx.load_constant(Fr::ONE);
                replay.samples.push(sampled.0);
                replay.events.push(AssignedTranscriptEvent {
                    is_sample,
                    value: sampled.0,
                });
            }
        }
    }

    replay
}

#[cfg(test)]
mod tests;
