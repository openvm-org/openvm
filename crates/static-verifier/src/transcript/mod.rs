use core::{array, iter};
use std::sync::OnceLock;

use halo2_base::{
    gates::{GateInstructions, RangeInstructions},
    halo2_proofs::arithmetic::Field,
    utils::{biguint_to_fe, fe_to_biguint},
    AssignedValue, Context, QuantumCell,
};
use num_bigint::BigUint;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{Bn254Scalar, Digest as RootDigest},
    openvm_stark_backend::p3_field::{Field as P3Field, PrimeField},
};

use crate::{
    field::baby_bear::{
        BabyBearChip, BabyBearExt4Wire, BabyBearExtWire, BabyBearWire, ReducedBabyBearExtWire,
        ReducedBabyBearWire, BABY_BEAR_BITS, BABY_BEAR_MODULUS_U64,
    },
    hash::{
        poseidon2::{pack_base_2_31_cells, Poseidon2State, DIGEST_WIDTH, POSEIDON2_RATE},
        POSEIDON2_PARAMS, POSEIDON2_WIDTH,
    },
    Fr,
};

/// Number of BabyBear values bit-packed into one BN254 word (base-2^31).
/// = floor(254 / 31) = 8
pub(crate) const NUM_OBS_PER_WORD: usize = 8;

/// Number of BabyBear samples extracted from one BN254 word (base-BabyBear decomposition).
/// Must match `compute_num_samples_per_elem` in `MultiFieldTranscript`.
pub(crate) const NUM_SAMPLES_PER_WORD: usize = 5;

/// Precomputed bounds for the base-BabyBear hint decomposition.
///
/// Below `k = NUM_SAMPLES_PER_WORD`.
struct BaseBabyBearDecompBounds {
    /// floor((q-1) / p^k) as a field element, for the boundary equality check.
    top_quotient_max_fe: Fr,
    /// floor((q-1) / p^k) + 1, for the range check.
    top_quotient_max_plus_one: BigUint,
    /// One more than the maximum lower part when top quotient is at its max.
    lower_max_plus_one: BigUint,
    /// p^k as a BigUint.
    pow_k: BigUint,
}

fn base_baby_bear_decomp_bounds() -> &'static BaseBabyBearDecompBounds {
    static BOUNDS: OnceLock<BaseBabyBearDecompBounds> = OnceLock::new();
    BOUNDS.get_or_init(|| {
        let p = BigUint::from(BABY_BEAR_MODULUS_U64);
        let one = BigUint::from(1u64);
        let modulus = <Bn254Scalar as P3Field>::order();
        let modulus_minus_one = &modulus - &one;
        let pow_k = p.pow(NUM_SAMPLES_PER_WORD as u32);
        let q_k_max = &modulus_minus_one / &pow_k;
        let lower_max = modulus_minus_one - &q_k_max * &pow_k;
        BaseBabyBearDecompBounds {
            top_quotient_max_fe: biguint_to_fe(&q_k_max),
            top_quotient_max_plus_one: &q_k_max + &one,
            lower_max_plus_one: lower_max + one,
            pow_k,
        }
    })
}

/// Decompose a BN254 field element into base-BabyBear digits using the
/// witness-and-verify (hint) approach.
///
/// This helper only computes the decomposition off-circuit and loads the resulting
/// raw digit cells plus top quotient as witnesses. Constraints must be enforced
/// separately before the digits are wrapped as `BabyBearWire`s.
fn load_base_baby_bear_decomposition_witness(
    ctx: &mut Context<Fr>,
    packed: AssignedValue<Fr>,
) -> ([AssignedValue<Fr>; NUM_SAMPLES_PER_WORD], AssignedValue<Fr>) {
    let p = BigUint::from(BABY_BEAR_MODULUS_U64);

    // Witness: compute digits and top quotient out-of-circuit.
    let mut value = fe_to_biguint(packed.value());
    let digit_witnesses_big: [BigUint; NUM_SAMPLES_PER_WORD] = array::from_fn(|_| {
        let digit = &value % &p;
        value /= &p;
        digit
    });
    let top_quotient_big = value;

    // Load each digit witness.
    let digit_witnesses =
        array::from_fn(|idx| ctx.load_witness(biguint_to_fe(&digit_witnesses_big[idx])));

    // Load top quotient witness.
    let top_quotient = ctx.load_witness(biguint_to_fe(&top_quotient_big));
    (digit_witnesses, top_quotient)
}

fn constrain_base_baby_bear_decomposition(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    range: &impl RangeInstructions<Fr>,
    packed: AssignedValue<Fr>,
    digit_witnesses: [AssignedValue<Fr>; NUM_SAMPLES_PER_WORD],
    top_quotient: AssignedValue<Fr>,
    bounds: &BaseBabyBearDecompBounds,
) -> [BabyBearWire; NUM_SAMPLES_PER_WORD] {
    let p = BigUint::from(BABY_BEAR_MODULUS_U64);
    let one = BigUint::from(1u64);

    for &digit in &digit_witnesses {
        range.check_less_than_safe(ctx, digit, BABY_BEAR_MODULUS_U64);
    }
    let top_quotient_valid =
        range.is_big_less_than_safe(ctx, top_quotient, bounds.top_quotient_max_plus_one.clone());
    gate.assert_is_const(ctx, &top_quotient_valid, &Fr::ONE);

    // Verify recomposition: lower = sum(digit_i * p^i)
    let lower = gate.inner_product(
        ctx,
        digit_witnesses.iter().copied(),
        iter::successors(Some(one), |power| Some(power * &p))
            .take(NUM_SAMPLES_PER_WORD)
            .map(|power| QuantumCell::Constant(biguint_to_fe(&power))),
    );

    // packed == top_quotient * p^k + lower
    let recomposed = gate.mul_add(
        ctx,
        top_quotient,
        QuantumCell::Constant(biguint_to_fe(&bounds.pow_k)),
        lower,
    );
    ctx.constrain_equal(&packed, &recomposed);

    // Boundary check: when top_quotient is at max, lower must not exceed the remainder.
    let at_top_boundary = gate.is_equal(
        ctx,
        top_quotient,
        QuantumCell::Constant(bounds.top_quotient_max_fe),
    );
    // Range-check lower against p^k (not lower_max_plus_one) because lower can be
    // any value in [0, p^k - 1] and p^k may need more bits than lower_max_plus_one.
    // Using is_big_less_than_safe with lower_max_plus_one would derive an insufficient
    // range_bits from lower_max_plus_one.bits() (e.g. 152 vs the 155 bits needed).
    let lower_range_bits =
        (bounds.pow_k.bits() as usize).div_ceil(range.lookup_bits()) * range.lookup_bits();
    range.range_check(ctx, lower, lower_range_bits);
    let lower_is_valid = range.is_less_than(
        ctx,
        lower,
        QuantumCell::Constant(biguint_to_fe(&bounds.lower_max_plus_one)),
        lower_range_bits,
    );
    let lower_is_invalid = gate.not(ctx, lower_is_valid);
    let lower_violation = gate.mul(ctx, at_top_boundary, lower_is_invalid);
    gate.assert_is_const(ctx, &lower_violation, &Fr::ZERO);

    digit_witnesses.map(|value| BabyBearWire {
        value,
        max_bits: BABY_BEAR_BITS,
    })
}

pub(crate) fn decompose_bn254_to_base_baby_bear_digits(
    ctx: &mut Context<Fr>,
    baby_bear: &BabyBearChip,
    packed: AssignedValue<Fr>,
) -> [BabyBearWire; NUM_SAMPLES_PER_WORD] {
    let bounds = base_baby_bear_decomp_bounds();
    let range = baby_bear.range();
    let gate = range.gate();
    let (digit_witnesses, top_quotient) = load_base_baby_bear_decomposition_witness(ctx, packed);
    constrain_base_baby_bear_decomposition(
        ctx,
        gate,
        range,
        packed,
        digit_witnesses,
        top_quotient,
        bounds,
    )
}

#[derive(Clone, Debug)]
pub struct DigestWire<F = AssignedValue<Fr>> {
    pub elems: [F; DIGEST_WIDTH],
}

pub fn digest_wire_from_root<F: Copy>(root: F) -> DigestWire<F> {
    DigestWire {
        elems: array::from_fn(|_| root),
    }
}

fn bn254_to_halo2(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

/// Circuit gadget that mirrors `MultiFieldTranscript<BabyBear, Bn254Scalar, _, 3, 2>`.
///
/// Uses absorb_idx/sample_idx sponge tracking, base-2^31 observe packing,
/// base-BabyBear sample decomposition, and direct digest absorption.
#[derive(Clone, Debug)]
pub struct TranscriptChip {
    baby_bear: BabyBearChip,
    sponge_state: [AssignedValue<Fr>; POSEIDON2_WIDTH],
    absorb_idx: usize,
    sample_idx: usize,
    observe_buf: Vec<ReducedBabyBearWire>,
    sample_buf: Vec<BabyBearWire>,
}

impl TranscriptChip {
    pub fn baby_bear(&self) -> &BabyBearChip {
        &self.baby_bear
    }

    pub fn new(ctx: &mut Context<Fr>, baby_bear: BabyBearChip) -> Self {
        let zero = ctx.load_zero();
        Self {
            baby_bear,
            sponge_state: array::from_fn(|_| zero),
            absorb_idx: 0,
            sample_idx: 0,
            observe_buf: Vec::with_capacity(NUM_OBS_PER_WORD),
            sample_buf: Vec::with_capacity(NUM_SAMPLES_PER_WORD),
        }
    }

    pub fn load_digest_witness(ctx: &mut Context<Fr>, digest: RootDigest) -> DigestWire {
        DigestWire {
            elems: array::from_fn(|i| ctx.load_witness(bn254_to_halo2(digest[i]))),
        }
    }

    // --- Low-level sponge (matches DuplexSponge::absorb/squeeze) ---

    fn sponge_absorb(&mut self, ctx: &mut Context<Fr>, value: AssignedValue<Fr>) {
        self.sponge_state[self.absorb_idx] = value;
        self.absorb_idx += 1;
        if self.absorb_idx == POSEIDON2_RATE {
            self.permute_state(ctx);
            self.absorb_idx = 0;
            self.sample_idx = POSEIDON2_RATE;
        }
    }

    fn sponge_squeeze(&mut self, ctx: &mut Context<Fr>) -> AssignedValue<Fr> {
        if self.absorb_idx != 0 || self.sample_idx == 0 {
            self.permute_state(ctx);
            self.absorb_idx = 0;
            self.sample_idx = POSEIDON2_RATE;
        }
        self.sample_idx -= 1;
        self.sponge_state[self.sample_idx]
    }

    fn permute_state(&mut self, ctx: &mut Context<Fr>) {
        let gate = self.baby_bear.range().gate();
        let mut state = Poseidon2State::new(self.sponge_state);
        state.permutation(ctx, gate, &POSEIDON2_PARAMS);
        self.sponge_state = state.s;
    }

    // --- Observe/sample buffer management ---

    // Rule: observe-side operations call `invalidate_samples`,
    //       sample-side operations call `flush_observe_buf`,
    //       cross-layer operations call both.

    fn invalidate_samples(&mut self) {
        self.sample_buf.clear();
    }

    fn flush_observe_buf(&mut self, ctx: &mut Context<Fr>) {
        if !self.observe_buf.is_empty() {
            let gate = self.baby_bear.range().gate();
            let packed = pack_base_2_31_cells(ctx, gate, &self.observe_buf);
            self.sponge_absorb(ctx, packed);
            self.observe_buf.clear();
        }
    }

    fn absorb_digest(&mut self, ctx: &mut Context<Fr>, digest: &DigestWire) {
        self.invalidate_samples();
        self.flush_observe_buf(ctx);
        for &elem in &digest.elems {
            self.sponge_absorb(ctx, elem);
        }
    }

    pub fn observe(&mut self, ctx: &mut Context<Fr>, value: &ReducedBabyBearWire) {
        self.invalidate_samples();
        self.observe_buf.push(*value);
        if self.observe_buf.len() == NUM_OBS_PER_WORD {
            self.flush_observe_buf(ctx);
        }
    }

    pub fn observe_ext(&mut self, ctx: &mut Context<Fr>, value: &ReducedBabyBearExtWire) {
        for coeff in value.coeffs() {
            self.observe(ctx, coeff);
        }
    }

    /// Absorb digest words directly into the sponge (lossless).
    pub fn observe_commit(&mut self, ctx: &mut Context<Fr>, digest: &DigestWire) {
        self.absorb_digest(ctx, digest);
    }

    pub fn sample(&mut self, ctx: &mut Context<Fr>) -> BabyBearWire {
        if let Some(val) = self.sample_buf.pop() {
            return val;
        }
        self.flush_observe_buf(ctx);
        let squeezed = self.sponge_squeeze(ctx);
        self.sample_buf = Vec::from(decompose_bn254_to_base_baby_bear_digits(
            ctx,
            &self.baby_bear,
            squeezed,
        ));
        // Reverse so pop() returns digits in order (b_0 first).
        self.sample_buf.reverse();
        self.sample_buf
            .pop()
            .expect("sample_buf should be non-empty")
    }

    pub fn sample_ext(&mut self, ctx: &mut Context<Fr>) -> BabyBearExtWire {
        let coeffs = array::from_fn(|_| self.sample(ctx));
        BabyBearExt4Wire(coeffs)
    }

    pub fn sample_bits(&mut self, ctx: &mut Context<Fr>, bits: usize) -> AssignedValue<Fr> {
        assert!(
            bits < (u32::BITS as usize),
            "sample_bits requires bits < 32: {bits}"
        );
        assert!(
            (1u64 << bits) < BABY_BEAR_MODULUS_U64,
            "sample_bits requires (1 << bits) < modulus: bits={bits}"
        );

        let sampled = self.sample(ctx);
        if bits == 0 {
            return ctx.load_zero();
        }
        // PERF[jpw]: we could optimize this since the divisor is a power of 2
        let range = self.baby_bear.range();
        let divisor = BigUint::from(1u64) << bits;
        let (_, rem) = range.div_mod(ctx, sampled.value, divisor, BABY_BEAR_BITS);
        rem
    }

    /// Asserts that the PoW witness must pass.
    pub fn check_witness(
        &mut self,
        ctx: &mut Context<Fr>,
        bits: usize,
        witness: &ReducedBabyBearWire,
    ) {
        if bits == 0 {
            return;
        }

        self.observe(ctx, witness);
        let sampled_bits = self.sample_bits(ctx, bits);
        self.baby_bear
            .range()
            .gate()
            .assert_is_const(ctx, &sampled_bits, &Fr::ZERO);
    }
}

#[cfg(test)]
pub(crate) mod tests;
