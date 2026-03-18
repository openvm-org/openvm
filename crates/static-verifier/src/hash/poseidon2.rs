//! Bn254 Scalar Poseidon2 permutation halo2-base implementation
use std::sync::OnceLock;

use halo2_base::{
    gates::{range::RangeChip, GateInstructions, RangeInstructions},
    utils::biguint_to_fe,
    AssignedValue, Context,
    QuantumCell::Constant,
};
use num_bigint::BigUint;
use zkhash::{
    ark_ff::{BigInteger as _, PrimeField as _},
    fields::bn256::FpBN256 as ArkBn254,
    poseidon2::poseidon2_instance_bn256::{MAT_DIAG3_M_1, RC3},
};

use crate::{field::baby_bear::BabyBearWire, Fr};

pub(crate) const POSEIDON2_WIDTH: usize = 3;
pub(crate) const POSEIDON2_RATE: usize = 2;
pub(crate) const POSEIDON2_ROUNDS_F: usize = 8;
pub(crate) const POSEIDON2_ROUNDS_P: usize = 56;
pub(crate) const MULTI_FIELD32_RATE: usize = 16;
pub(crate) const MULTI_FIELD32_NUM_F_ELMS: usize = 8;
pub(crate) const DIGEST_WIDTH: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct Poseidon2Params {
    external_rc: Vec<[Fr; POSEIDON2_WIDTH]>,
    internal_rc: Vec<Fr>,
    mat_internal_diag_m_1: [Fr; POSEIDON2_WIDTH],
}

// TODO: re-expose from stark-sdk
pub(crate) fn poseidon2_params() -> &'static Poseidon2Params {
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
    values: &[BabyBearWire],
) -> AssignedValue<Fr> {
    let mut state = core::array::from_fn(|_| ctx.load_constant(Fr::from(0u64)));
    for block_chunk in values.chunks(MULTI_FIELD32_RATE) {
        for (chunk_id, chunk) in block_chunk.chunks(MULTI_FIELD32_NUM_F_ELMS).enumerate() {
            let cells = chunk.iter().map(|value| value.0).collect::<Vec<_>>();
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
