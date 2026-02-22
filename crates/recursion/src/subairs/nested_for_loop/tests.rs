#![cfg(debug_assertions)]

use core::borrow::Borrow;

use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::{
    any_air_arc_vec,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{
        AirProvingContext, ColMajorMatrix, CpuBackend, DeviceDataTransporter, ProvingContext,
    },
    utils::disable_debug_builder,
    verifier::VerifierError,
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};

use super::{NestedForLoopIoCols, NestedForLoopSubAir};
use crate::tests::{test_engine_small, MAX_CONSTRAINT_DEGREE};

const fn width<const DEPTH_MINUS_ONE: usize>() -> usize {
    size_of::<NestedForLoopIoCols<u8, DEPTH_MINUS_ONE>>()
}

#[derive(Clone, Copy)]
struct TestAir<const DEPTH_MINUS_ONE: usize>;

impl<F: Field, const DEPTH_MINUS_ONE: usize> BaseAirWithPublicValues<F>
    for TestAir<DEPTH_MINUS_ONE>
{
}
impl<F: Field, const DEPTH_MINUS_ONE: usize> PartitionedBaseAir<F> for TestAir<DEPTH_MINUS_ONE> {}
impl<F: Field, const DEPTH_MINUS_ONE: usize> BaseAir<F> for TestAir<DEPTH_MINUS_ONE> {
    fn width(&self) -> usize {
        width::<DEPTH_MINUS_ONE>()
    }
}

impl<AB: AirBuilder, const DEPTH_MINUS_ONE: usize> Air<AB> for TestAir<DEPTH_MINUS_ONE>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local_io: &NestedForLoopIoCols<AB::Var, DEPTH_MINUS_ONE> = (*local).borrow();
        let next_io: &NestedForLoopIoCols<AB::Var, DEPTH_MINUS_ONE> = (*next).borrow();

        NestedForLoopSubAir::<DEPTH_MINUS_ONE>.eval(
            builder,
            (local_io.clone().map_into(), next_io.clone().map_into()),
        );
    }
}

fn generate_trace<F: Field, const DEPTH_MINUS_ONE: usize, const POSEIDON2_WIDTH: usize>(
    rows: Vec<[u32; POSEIDON2_WIDTH]>,
) -> RowMajorMatrix<F> {
    let mut rows: Vec<[F; POSEIDON2_WIDTH]> =
        rows.into_iter().map(|r| r.map(F::from_u32)).collect();

    let io_width = size_of::<NestedForLoopIoCols<u8, DEPTH_MINUS_ONE>>();

    let padding_counters = rows
        .last()
        .map(|row| {
            let cols: &NestedForLoopIoCols<F, DEPTH_MINUS_ONE> = row[..io_width].borrow();
            if cols.is_enabled == F::ZERO {
                cols.counter
            } else {
                cols.counter.map(|idx| idx + F::ONE)
            }
        })
        .unwrap_or([F::ZERO; DEPTH_MINUS_ONE]);

    let mut padding_row = [F::ZERO; POSEIDON2_WIDTH];
    padding_row[1..(DEPTH_MINUS_ONE + 1)].copy_from_slice(&padding_counters[..DEPTH_MINUS_ONE]);

    let padded_len = rows.len().next_power_of_two().max(4);
    rows.resize(padded_len, padding_row);

    RowMajorMatrix::new(rows.into_flattened(), POSEIDON2_WIDTH)
}

fn prove_and_verify(
    airs: Vec<AirRef<BabyBearPoseidon2Config>>,
    traces: Vec<RowMajorMatrix<F>>,
) -> Result<(), VerifierError<<BabyBearPoseidon2Config as StarkProtocolConfig>::EF>> {
    let engine = test_engine_small();

    // Debug constraints using v2 engine
    let debug_ctx = ProvingContext::new(
        traces
            .iter()
            .enumerate()
            .map(|(air_idx, trace)| {
                (
                    air_idx,
                    AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(trace)),
                )
            })
            .collect(),
    );
    engine.debug(&airs, &debug_ctx);
    let (pk, vk) = engine.keygen(&airs);

    let ctx: ProvingContext<CpuBackend<BabyBearPoseidon2Config>> = ProvingContext::new(
        traces
            .into_iter()
            .enumerate()
            .map(|(air_idx, trace)| {
                (
                    air_idx,
                    AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
                )
            })
            .collect(),
    );

    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx: ProvingContext<CpuBackend<BabyBearPoseidon2Config>> = ProvingContext::new(
        ctx.into_iter()
            .map(|(air_idx, air_ctx)| {
                (
                    air_idx,
                    AirProvingContext {
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

fn prove_and_verify_test_air<const DEPTH_MINUS_ONE: usize>(trace: RowMajorMatrix<F>) {
    disable_debug_builder();
    let airs = any_air_arc_vec![TestAir::<DEPTH_MINUS_ONE>];
    prove_and_verify(airs, vec![trace]).unwrap();
}

// ============================================================================
// Tests for DEPTH=2: Two Nested Loops
// ============================================================================

#[test]
fn test_max_constraint_degree() {
    let engine = test_engine_small();
    let airs = any_air_arc_vec![TestAir::<1>];
    let (_pk, vk) = engine.keygen(&airs);

    assert!(vk.max_constraint_degree() <= MAX_CONSTRAINT_DEGREE);
}

#[test]
fn test_single_outer_iteration_enabled() {
    // for i in 0..1 { for j in 0..1 { ... }}
    // [is_enabled, counter[0]=i, is_first[0]]
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_single_row_disabled() {
    // All rows disabled (no active loop iterations)
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_single_outer_iteration_multiple_rows() {
    // for i in 0..1 { for j in 0..1 { ... }}
    // With multiple rows in the same outer iteration (i=0)
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 0, 0], [1, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_single_outer_iteration_with_padding() {
    // for i in 0..1 { for j in 0..1 { ... }}
    // Followed by disabled padding
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 0, 0], [0, 1, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_disabled_multiple_rows() {
    // All rows disabled
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_disabled_with_nonzero_counter() {
    // All rows disabled with non-zero loop index
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[0, 5, 0], [0, 5, 0], [0, 5, 0], [0, 5, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_two_outer_iterations() {
    // for i in 0..2 { for j in 0..M { ... }}
    // With multiple rows per outer iteration
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_two_outer_iterations_single_row_each() {
    // for i in 0..2 { for j in 0..1 { ... }}
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 1, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_three_outer_iterations() {
    // for i in 0..3 { for j in 0..M { ... }}
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 2, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic]
fn test_fail_outer_iterations_with_padding_between() {
    // for i in 0..3 { for j in 0..M { ... }}
    // Disabled padding between iterations should now fail
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [0, 1, 0], [1, 2, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_disabled_with_counter_increment() {
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_multiple_disabled_iterations() {
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_enabled_then_disabled_iterations() {
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [0, 1, 0], [0, 2, 0], [0, 3, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_disabled_then_enabled() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0], [0, 0, 0], [1, 1, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_complex_enabled_disabled_mix() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 2, 1],
        [1, 2, 0],
        [1, 2, 0],
        [0, 3, 0],
    ]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_nonzero_counter_with_padding() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 5, 1], [1, 5, 0], [0, 6, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_padding_with_incrementing_counter() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
    ]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_all_rows_disabled() {
    let trace =
        generate_trace::<_, 1, { width::<1>() }>(vec![[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
fn test_first_row_nonzero_counter() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 2, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_is_first_not_boolean() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 2]]);
    prove_and_verify_test_air::<1>(trace);
}

// Boundary constraints
#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_missing_start() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

// Disabled row constraints
#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_disabled_row_has_is_first() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [0, 0, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

// Transition constraints
#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_counter_jumps_by_more_than_one() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 2, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_counter_decreases() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 1, 1], [1, 0, 1]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_is_enabled_changes_within_iteration() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [0, 0, 0], [1, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_is_first_set_within_iteration() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 0, 1], [1, 0, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_fail_iteration_boundary_missing_is_first() {
    let trace = generate_trace::<_, 1, { width::<1>() }>(vec![[1, 0, 1], [1, 1, 0]]);
    prove_and_verify_test_air::<1>(trace);
}

// ============================================================================
// Tests for DEPTH=3: Three Nested Loops
// ============================================================================

#[test]
fn test_nested_max_constraint_degree() {
    let engine = test_engine_small();
    let airs = any_air_arc_vec![TestAir::<2>];
    let (_pk, vk) = engine.keygen(&airs);

    assert!(vk.max_constraint_degree() <= MAX_CONSTRAINT_DEGREE);
}

#[test]
fn test_three_loops_single_iteration_each() {
    // for i in 0..1 { for j in 0..1 { for k in 0..1 { ... }}}
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_multiple_middle_iterations() {
    // for i in 0..1 { for j in 0..3 { for k in 0..M { ... }}}
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [1, 0, 0, 1, 1], // i=0, j=0 start, k start
        [1, 0, 0, 0, 0], // i=0, j=0 continue, k continue
        [1, 0, 1, 0, 1], // i=0, j=1 start, k start
        [1, 0, 1, 0, 0], // i=0, j=1 continue, k continue
        [1, 0, 2, 0, 1], // i=0, j=2 start, k start
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_multiple_outer_iterations() {
    // for i in 0..2 { for j in 0..1 { for k in 0..1 { ... }}}
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [1, 0, 0, 1, 1], // i=0, j=0 start, k start
        [1, 1, 0, 1, 1], // i=1, j=0 start, k start
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_2x2_iterations() {
    // for i in 0..2 { for j in 0..2 { for k in 0..M { ... }}}
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [1, 0, 0, 1, 1], // i=0, j=0 start, k start
        [1, 0, 0, 0, 0], // i=0, j=0 continue, k continue
        [1, 0, 1, 0, 1], // i=0, j=1 start, k start
        [1, 0, 1, 0, 0], // i=0, j=1 continue, k continue
        [1, 1, 0, 1, 1], // i=1, j=0 start, k start
        [1, 1, 0, 0, 0], // i=1, j=0 continue, k continue
        [1, 1, 1, 0, 1], // i=1, j=1 start, k start
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_with_disabled_padding() {
    // for i in 0..1 { for j in 0..2 { for k in 0..M { ... }}}
    // Followed by disabled padding
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [1, 0, 0, 1, 1], // i=0, j=0 start, k start
        [1, 0, 1, 0, 1], // i=0, j=1 start, k start
        [0, 1, 0, 0, 0], // Disabled padding
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_all_disabled() {
    // All rows disabled (no active loop iterations)
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
fn test_three_loops_3x2_iterations() {
    // for i in 0..3 { for j in 0..2 { for k in 0..1 { ... }}}
    // [enabled, counter[0]=i, counter[1]=j, is_first[0]=j_is_first, is_first[1]=k_is_first]
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![
        [1, 0, 0, 1, 1], // i=0, j=0 start, k start
        [1, 0, 1, 0, 1], // i=0, j=1 start, k start
        [1, 1, 0, 1, 1], // i=1, j=0 start, k start
        [1, 1, 1, 0, 1], // i=1, j=1 start, k start
        [1, 2, 0, 1, 1], // i=2, j=0 start, k start
        [1, 2, 1, 0, 1], // i=2, j=1 start, k start
    ]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_missing_start_on_outer_start() {
    // When i starts, j must also start
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 0, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_missing_start_on_outer_increment() {
    // When i increments, j must start
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 1, 0, 0, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_start_within_iteration() {
    // j start should only happen when j increments or i increments
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 0, 0, 1, 0]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_counter_jumps() {
    // j jumps from 0 to 2
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 0, 2, 1, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_outer_counter_jumps() {
    // i jumps from 0 to 2
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 2, 0, 1, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_counter_decreases() {
    // j decreases from 1 to 0
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 1, 1, 1], [1, 0, 0, 1, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_outer_missing_start_flag() {
    // i missing start flag on first row (j is missing start flag, since there's no i start flag in
    // data)
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 0]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_outer_start_within_iteration() {
    // i start flag set when j increments but i doesn't
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 0, 1, 1, 1]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_middle_boundary_missing_start() {
    // When j increments, must have j start flag
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 0, 1, 0, 0]]);
    prove_and_verify_test_air::<2>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_three_loops_fail_outer_boundary_missing_start() {
    // When i increments, must have j start flag (there's no i start flag in columns)
    let trace = generate_trace::<_, 2, { width::<2>() }>(vec![[1, 0, 0, 1, 1], [1, 1, 0, 1, 0]]);
    prove_and_verify_test_air::<2>(trace);
}

// ============================================================================
// Tests for DEPTH=4: Four Nested Loops
// ============================================================================

#[test]
fn test_four_loops_max_constraint_degree() {
    let engine = test_engine_small();
    let airs = any_air_arc_vec![TestAir::<3>];
    let (_pk, vk) = engine.keygen(&airs);

    assert!(vk.max_constraint_degree() <= MAX_CONSTRAINT_DEGREE);
}

#[test]
fn test_four_loops_single_iteration_each() {
    // for i in 0..1 { for j in 0..1 { for k in 0..1 { for l in 0..1 { ... }}}}
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![[1, 0, 0, 0, 1, 1, 1]]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
fn test_four_loops_multiple_mid_inner_iterations() {
    // for i in 0..1 { for j in 0..1 { for k in 0..2 { for l in 0..M { ... }}}}
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1], // i=0, j=0 start, k=0 start, l start
        [1, 0, 0, 1, 0, 0, 1], // i=0, j=0 continue, k=1 start, l start
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
fn test_four_loops_mid_outer_increment_resets_mid_inner() {
    // for i in 0..1 { for j in 0..2 { for k in 0..2 { for l in 0..M { ... }}}}
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1], // i=0, j=0 start, k=0 start, l start
        [1, 0, 0, 1, 0, 0, 1], // i=0, j=0 continue, k=1 start, l start
        [1, 0, 1, 0, 0, 1, 1], // i=0, j=1 start, k=0 start, l start
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
fn test_four_loops_outer_increment_resets_all() {
    // for i in 0..2 { for j in 0..2 { for k in 0..2 { for l in 0..M { ... }}}}
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1], // i=0, j=0 start, k=0 start, l start
        [1, 0, 0, 1, 0, 0, 1], // i=0, j=0 continue, k=1 start, l start
        [1, 0, 1, 0, 0, 1, 1], // i=0, j=1 start, k=0 start (j increments, k resets)
        [1, 0, 1, 1, 0, 0, 1], // i=0, j=1 continue, k=1 start
        [1, 1, 0, 0, 1, 1, 1], /* i=1, j=0 start, k=0 start (i increments, j and k both
                                * reset) */
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
fn test_four_loops_2x2x2_iterations() {
    // for i in 0..2 { for j in 0..2 { for k in 0..2 { for l in 0..1 { ... }}}}
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1], // i=0, j=0 start, k=0 start, l start
        [1, 0, 0, 1, 0, 0, 1], // i=0, j=0 continue, k=1 start, l start
        [1, 0, 1, 0, 0, 1, 1], // i=0, j=1 start, k=0 start, l start
        [1, 0, 1, 1, 0, 0, 1], // i=0, j=1 continue, k=1 start, l start
        [1, 1, 0, 0, 1, 1, 1], // i=1, j=0 start, k=0 start, l start
        [1, 1, 0, 1, 0, 0, 1], // i=1, j=0 continue, k=1 start, l start
        [1, 1, 1, 0, 0, 1, 1], // i=1, j=1 start, k=0 start, l start
        [1, 1, 1, 1, 0, 0, 1], // i=1, j=1 continue, k=1 start, l start
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
fn test_four_loops_with_disabled_padding() {
    // for i in 0..1 { for j in 0..1 { for k in 0..2 { for l in 0..M { ... }}}}
    // Followed by disabled padding
    // [enabled, counter[0]=i, counter[1]=j, counter[2]=k, is_first[0]=j_is_first,
    // is_first[1]=k_is_first, is_first[2]=l_is_first]
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1], // i=0, j=0 start, k=0 start, l start
        [1, 0, 0, 1, 0, 0, 1], // i=0, j=0 continue, k=1 start, l start
        [0, 1, 0, 0, 0, 0, 0], // Disabled padding
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_four_loops_fail_mid_inner_missing_start_on_mid_outer_start() {
    // When j starts, k must start
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0], // j starts but k doesn't
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_four_loops_fail_mid_outer_and_mid_inner_missing_start_on_outer_start() {
    // When i starts, j and k must start
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 0, 1], // i starts but j doesn't
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_four_loops_fail_mid_inner_counter_jumps() {
    // k jumps from 0 to 2
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 2, 1, 0, 1], // k jumps from 0 to 2
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_four_loops_fail_mid_outer_counter_jumps() {
    // j jumps from 0 to 2
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1],
        [1, 0, 2, 0, 1, 1, 1], // j jumps from 0 to 2
    ]);
    prove_and_verify_test_air::<3>(trace);
}

#[test]
#[should_panic] // TODO: catch explicit error
fn test_four_loops_fail_mid_inner_start_within_iteration() {
    // k start flag when k doesn't change
    let trace = generate_trace::<_, 3, { width::<3>() }>(vec![
        [1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1], // k start flag when k doesn't change
    ]);
    prove_and_verify_test_air::<3>(trace);
}
