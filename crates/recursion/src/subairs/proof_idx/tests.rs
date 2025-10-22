use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::{
    AirRef,
    engine::StarkEngine,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    p3_matrix::{Matrix, dense::RowMajorMatrix},
    prover::types::AirProofRawInput,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{
        FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
    },
    engine::StarkFriEngine,
};
use stark_backend_v2::{
    F, StarkEngineV2,
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, DeviceDataTransporterV2,
        ProvingContextV2,
    },
    test_utils::test_engine_small,
    verifier::VerifierError,
};

use super::{ProofIdxCols, ProofIdxSubAir};
use crate::utils::MAX_CONSTRAINT_DEGREE;

const WIDTH: usize = size_of::<ProofIdxCols<u8>>();

#[derive(Clone, Copy)]
struct TestAir;

impl<F: Field> BaseAirWithPublicValues<F> for TestAir {}
impl<F: Field> PartitionedBaseAir<F> for TestAir {}
impl<F: Field> BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for TestAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        let local_cols = ProofIdxCols {
            proof_idx: local[0],
            is_enabled: local[1],
            is_proof_start: local[2],
            is_proof_end: local[3],
        };
        let next_cols = ProofIdxCols {
            proof_idx: next[0],
            is_enabled: next[1],
            is_proof_start: next[2],
            is_proof_end: next[3],
        };

        ProofIdxSubAir.eval(builder, (local_cols, next_cols));
    }
}

fn generate_trace<F: Field>(rows: Vec<[u32; WIDTH]>) -> RowMajorMatrix<F> {
    let padding_proof_idx = rows
        .last()
        .map(|&[proof_idx, is_enabled, _, _]| {
            if is_enabled == 0 {
                proof_idx
            } else {
                proof_idx + 1
            }
        })
        .unwrap_or_default();

    let padded_len = rows.len().next_power_of_two().max(4);
    let mut padded = rows;
    padded.resize(padded_len, [padding_proof_idx, 0, 0, 0]);

    let values: Vec<F> = padded
        .into_iter()
        .flat_map(|r| r.map(F::from_canonical_u32))
        .collect();

    RowMajorMatrix::new(values, WIDTH)
}

// TODO(ayush): add debug method to v2 engine
fn debug_constraints(airs: &[AirRef<BabyBearPoseidon2Config>], traces: &[RowMajorMatrix<F>]) {
    // Use v1 engine for debug
    let engine_v1 = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine_v1.keygen_builder();
    for air in airs {
        keygen_builder.add_air(air.clone());
    }
    let pk_v1 = keygen_builder.generate_pk();

    let proof_inputs: Vec<_> = traces
        .iter()
        .map(|trace| AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(trace.clone().into()),
            public_values: vec![],
        })
        .collect();

    engine_v1.debug(airs, &pk_v1.per_air, &proof_inputs);
}

fn prove_and_verify(
    airs: Vec<AirRef<BabyBearPoseidon2Config>>,
    traces: Vec<RowMajorMatrix<F>>,
) -> Result<(), VerifierError> {
    debug_constraints(&airs, &traces);

    let engine = test_engine_small();
    let (pk, vk) = engine.keygen(&airs);

    let ctx: ProvingContextV2<CpuBackendV2> = ProvingContextV2::new(
        traces
            .into_iter()
            .enumerate()
            .map(|(air_idx, trace)| {
                (
                    air_idx,
                    AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
                )
            })
            .collect(),
    );

    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx: ProvingContextV2<CpuBackendV2> = ProvingContextV2::new(
        ctx.into_iter()
            .map(|(air_idx, air_ctx)| {
                (
                    air_idx,
                    AirProvingContextV2 {
                        cached_mains: vec![],
                        common_main: device.transport_matrix_to_device(&air_ctx.common_main),
                        public_values: air_ctx.public_values,
                    },
                )
            })
            .collect(),
    );

    let proof = engine.prove(&d_pk, d_ctx);
    engine.verify(&vk, &proof)
}

fn prove_and_verify_test_air(trace: RowMajorMatrix<F>) {
    disable_debug_builder();
    let airs = any_rap_arc_vec![TestAir];
    prove_and_verify(airs, vec![trace]).unwrap();
}

#[test]
fn test_max_constraint_degree() {
    let engine = test_engine_small();
    let airs = any_rap_arc_vec![TestAir];
    let (_pk, vk) = engine.keygen(&airs);

    assert!(vk.inner.max_constraint_degree <= MAX_CONSTRAINT_DEGREE);
}

#[test]
fn test_single_row_enabled() {
    let trace = generate_trace(vec![[0, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_single_row_disabled() {
    let trace = generate_trace(vec![[0, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_single_proof_multiple_rows() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_single_proof_with_padding() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_disabled_proof_multiple_rows() {
    let trace = generate_trace(vec![[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_disabled_interior_rows() {
    let trace = generate_trace(vec![[5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_two_proofs() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_two_proofs_single_row_each() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_three_proofs() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [2, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_proofs_with_padding_between() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 0, 0, 0], [2, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_disabled_to_disabled_with_increment() {
    let trace = generate_trace(vec![[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_multiple_disabled_proofs() {
    let trace = generate_trace(vec![[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_enabled_then_disabled_proofs() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_disabled_to_enabled() {
    let trace = generate_trace(vec![[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_complex_enabled_disabled_mix() {
    let trace = generate_trace(vec![
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 1, 1, 0],
        [2, 1, 0, 0],
        [2, 1, 0, 1],
        [3, 0, 0, 0],
    ]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_nonzero_proof_idx_with_padding() {
    let trace = generate_trace(vec![[5, 1, 1, 0], [5, 1, 0, 1], [6, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_padding_with_incrementing_proof_idx() {
    let trace = generate_trace(vec![
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
    ]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_all_rows_disabled() {
    let trace = generate_trace(vec![[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
fn test_first_row_nonzero_proof_idx() {
    let trace = generate_trace(vec![[2, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

// Boolean constraints
#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_is_enabled_not_boolean() {
    let trace = generate_trace(vec![[0, 2, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_is_proof_start_not_boolean() {
    let trace = generate_trace(vec![[0, 1, 2, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_is_proof_end_not_boolean() {
    let trace = generate_trace(vec![[0, 1, 1, 2]]);
    prove_and_verify_test_air(trace);
}

// Boundary constraints
#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_missing_start() {
    let trace = generate_trace(vec![[0, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_missing_end() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]);
    prove_and_verify_test_air(trace);
}

// Disabled row constraints
#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_disabled_row_nonzero_proof_start() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [0, 0, 1, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_disabled_row_nonzero_proof_end() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [0, 0, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_disabled_row_has_both_start_and_end() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [0, 0, 1, 1]]);
    prove_and_verify_test_air(trace);
}

// Transition constraints
#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_proof_idx_jumps_by_more_than_one() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [2, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_proof_idx_decreases() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [0, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_is_enabled_changes_within_proof() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_within_proof_has_end_flag() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_within_proof_has_start_flag() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_proof_boundary_current_not_end() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [1, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_proof_boundary_next_not_start() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 1, 0, 1]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_enabled_to_disabled_not_end() {
    let trace = generate_trace(vec![[0, 1, 1, 0], [1, 0, 0, 0]]);
    prove_and_verify_test_air(trace);
}

#[test]
#[should_panic(expected = "Zerocheck sum is not zero")]
fn test_fail_disabled_to_enabled_same_proof_idx() {
    let trace = generate_trace(vec![[0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]);
    prove_and_verify_test_air(trace);
}
