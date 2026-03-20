use std::sync::OnceLock;

use halo2_base::{
    gates::{range::RangeChip, GateInstructions, RangeInstructions},
    halo2_proofs::arithmetic::Field,
    utils::{biguint_to_fe, fe_to_biguint},
    AssignedValue, Context,
};
use itertools::Itertools;
use num_bigint::BigUint;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{Bn254Scalar, Digest as RootDigest},
    openvm_stark_backend::p3_field::PrimeField,
};

use crate::{
    field::baby_bear::{
        BabyBearChip, BabyBearExt4Wire, BabyBearExtWire, BabyBearWire, BABY_BEAR_BITS,
        BABY_BEAR_MODULUS_U64,
    },
    hash::{
        poseidon2::{reduce_32_cells, Poseidon2State, DIGEST_WIDTH, POSEIDON2_RATE},
        POSEIDON2_PARAMS, POSEIDON2_WIDTH,
    },
    Fr,
};

pub(crate) const NUM_SPLIT_LIMBS: usize = 3;

#[derive(Clone, Debug)]
pub struct DigestWire {
    pub elems: [AssignedValue<Fr>; DIGEST_WIDTH],
}

pub fn digest_wire_from_root(root: AssignedValue<Fr>) -> DigestWire {
    DigestWire {
        elems: core::array::from_fn(|_| root),
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
    let limb_base: BigUint = BigUint::from(1u64) << 64usize;
    let (quot0, limb0) = range.div_mod(ctx, packed, limb_base.clone(), 254);
    let (quot1, limb1) = range.div_mod(ctx, quot0, limb_base.clone(), 190);
    let (high, limb2) = range.div_mod(ctx, quot1, limb_base, 126);
    range.check_less_than_safe(ctx, high, split_high_bound() + 1);
    [limb0, limb1, limb2]
}

fn reduce_assigned_limb_to_babybear(
    ctx: &mut Context<Fr>,
    baby_bear: &BabyBearChip,
    limb: AssignedValue<Fr>,
) -> BabyBearWire {
    let range = baby_bear.range();
    let (_, remainder) = range.div_mod(ctx, limb, BigUint::from(BABY_BEAR_MODULUS_U64), 64);
    BabyBearWire {
        value: remainder,
        max_bits: BABY_BEAR_BITS,
    }
}

pub fn split_assigned_bn254_to_babybear_limbs(
    ctx: &mut Context<Fr>,
    baby_bear: &BabyBearChip,
    packed: AssignedValue<Fr>,
) -> [BabyBearWire; NUM_SPLIT_LIMBS] {
    let limbs = decompose_packed_bn254_to_split_limbs(ctx, baby_bear.range(), packed);
    core::array::from_fn(|idx| reduce_assigned_limb_to_babybear(ctx, baby_bear, limbs[idx]))
}

impl TranscriptGadget {
    pub fn new(ctx: &mut Context<Fr>) -> Self {
        let zero = ctx.load_zero();
        let sponge_state = core::array::from_fn(|_| zero);
        Self {
            sponge_state,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    pub fn load_digest_witness(ctx: &mut Context<Fr>, digest: RootDigest) -> DigestWire {
        DigestWire {
            elems: core::array::from_fn(|i| ctx.load_witness(bn254_to_halo2(digest[i]))),
        }
    }

    fn permute_state(&mut self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) {
        let gate = range.gate();
        let mut state = Poseidon2State::new(self.sponge_state);
        state.permutation(ctx, gate, &POSEIDON2_PARAMS);
        self.sponge_state = state.s;
    }

    fn reduce_32(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        values: &[BabyBearWire],
    ) -> AssignedValue<Fr> {
        reduce_32_cells(
            ctx,
            range.gate(),
            &values.iter().map(|v| v.value).collect_vec(),
        )
    }

    fn split_state_to_babybear(
        &self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
        packed: AssignedValue<Fr>,
    ) -> [BabyBearWire; NUM_SPLIT_LIMBS] {
        let limbs = decompose_packed_bn254_to_split_limbs(ctx, baby_bear.range(), packed);
        core::array::from_fn(|idx| reduce_assigned_limb_to_babybear(ctx, baby_bear, limbs[idx]))
    }

    fn duplexing(&mut self, ctx: &mut Context<Fr>, baby_bear: &BabyBearChip) {
        assert!(
            self.input_buffer.len() <= NUM_SPLIT_LIMBS * POSEIDON2_RATE,
            "input buffer exceeds transcript absorb rate"
        );

        let range = baby_bear.range();
        for (idx, chunk) in self.input_buffer.chunks(NUM_SPLIT_LIMBS).enumerate() {
            self.sponge_state[idx] = self.reduce_32(ctx, range, chunk);
        }
        self.input_buffer.clear();

        self.permute_state(ctx, range);

        self.output_buffer.clear();
        for packed in self.sponge_state {
            let parts = self.split_state_to_babybear(ctx, baby_bear, packed);
            self.output_buffer.extend(parts);
        }
    }

    pub fn observe(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
        value: &BabyBearWire,
    ) {
        self.output_buffer.clear();
        self.input_buffer.push(*value);
        if self.input_buffer.len() == NUM_SPLIT_LIMBS * POSEIDON2_RATE {
            self.duplexing(ctx, baby_bear);
        }
    }

    pub fn observe_ext(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
        value: &BabyBearExtWire,
    ) {
        for coeff in &value.0 {
            self.observe(ctx, baby_bear, coeff);
        }
    }

    pub fn observe_commit(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
        digest: &DigestWire,
    ) {
        for packed in &digest.elems {
            let limbs = self.split_state_to_babybear(ctx, baby_bear, *packed);
            for limb in &limbs {
                self.observe(ctx, baby_bear, limb);
            }
        }
    }

    pub fn sample(&mut self, ctx: &mut Context<Fr>, baby_bear: &BabyBearChip) -> BabyBearWire {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing(ctx, baby_bear);
        }

        self.output_buffer
            .pop()
            .expect("transcript output buffer must be non-empty after duplexing")
    }

    pub fn sample_ext(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
    ) -> BabyBearExtWire {
        let coeffs = core::array::from_fn(|_| self.sample(ctx, baby_bear));
        BabyBearExt4Wire(coeffs)
    }

    pub fn sample_bits(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
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

        if bits == 0 {
            return ctx.load_zero();
        }
        let sampled = self.sample(ctx, baby_bear);
        // Reduce BabyBearWire so it is constrained to be less than BabyBear modulus
        let sampled_reduced = baby_bear.reduce(ctx, sampled);
        // PERF[jpw]: we could optimize this since the divisor is a power of 2
        let range = baby_bear.range();
        let (_, rem) = range.div_mod(
            ctx,
            sampled_reduced.value,
            BigUint::from(1u64) << bits,
            BABY_BEAR_BITS,
        );
        rem
    }

    /// Asserts that the PoW witness must pass.
    pub fn check_witness(
        &mut self,
        ctx: &mut Context<Fr>,
        baby_bear: &BabyBearChip,
        bits: usize,
        witness: &BabyBearWire,
    ) {
        if bits == 0 {
            return;
        }

        self.observe(ctx, baby_bear, witness);
        let sampled_bits = self.sample_bits(ctx, baby_bear, bits);
        baby_bear
            .range()
            .gate()
            .assert_is_const(ctx, &sampled_bits, &Fr::ZERO);
    }
}

#[cfg(test)]
pub(crate) mod tests;
