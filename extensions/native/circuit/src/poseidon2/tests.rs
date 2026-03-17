use std::{borrow::BorrowMut, cmp::min};

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder, VmChipTester},
    VmField,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS, Poseidon2Opcode, Poseidon2Opcode::*, VerifyBatchOpcode::VERIFY_BATCH,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{
    air_builders::debug::DebugConstraintBuilder,
    interaction::RapPhaseSeqKind,
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32, PrimeField64},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView},
        stack::VerticalPair,
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_blake3::{BabyBearBlake3Config, BabyBearBlake3Engine},
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng};

use super::air::VerifyBatchBus;
use crate::poseidon2::{
    air::NativePoseidon2Air,
    chip::{NativePoseidon2Executor, NativePoseidon2Filler},
    columns::{InsideRowSpecificCols, NativePoseidon2Cols},
    NativePoseidon2Chip, CHUNK,
};
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_cuda_backend::types::F as CudaF;
        use crate::poseidon2::{chip::NativePoseidon2RecordMut, NativePoseidon2ChipGpu};
    } else {
        use openvm_circuit::utils::air_test;
        use openvm_instructions::{program::Program, SystemOpcode};
        use openvm_native_compiler::FieldArithmeticOpcode;
        use crate::{NativeConfig, NativeCpuBuilder};
    }
}

const VERIFY_BATCH_BUS: VerifyBatchBus = VerifyBatchBus::new(7);
const MAX_INS_CAPACITY: usize = 1 << 15;
type Harness<F, const SBOX_REGISTERS: usize> = TestChipHarness<
    F,
    NativePoseidon2Executor<F, SBOX_REGISTERS>,
    NativePoseidon2Air<F, SBOX_REGISTERS>,
    NativePoseidon2Chip<F, SBOX_REGISTERS>,
>;

fn create_test_chip<F: VmField, const SBOX_REGISTERS: usize>(
    tester: &VmChipTestBuilder<F>,
) -> Harness<F, SBOX_REGISTERS> {
    let air = NativePoseidon2Air::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        VERIFY_BATCH_BUS,
        Poseidon2Config::default(),
    );
    let step = NativePoseidon2Executor::new(Poseidon2Config::default());
    let chip = NativePoseidon2Chip::new(
        NativePoseidon2Filler::new(Poseidon2Config::default()),
        tester.memory_helper(),
    );

    Harness::with_capacity(step, air, chip, MAX_INS_CAPACITY)
}

fn compute_commit<F: Field>(
    dim: &[usize],
    opened: &[Vec<F>],
    proof: &[[F; CHUNK]],
    sibling_is_on_right: &[bool],
    hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
) -> [F; CHUNK] {
    let mut log_height = dim[0] as isize;
    let mut proof_index = 0;
    let mut opened_index = 0;
    let mut root = [F::ZERO; CHUNK];
    while log_height >= 0 {
        let mut concat = vec![];
        while opened_index < opened.len() && dim[opened_index] == log_height as usize {
            concat.extend(opened[opened_index].clone());
            opened_index += 1;
        }
        if !concat.is_empty() {
            let mut left = [F::ZERO; CHUNK];
            let mut right = [F::ZERO; CHUNK];
            for i in (0..concat.len()).step_by(CHUNK) {
                left[..(min(i + CHUNK, concat.len()) - i)]
                    .copy_from_slice(&concat[i..min(i + CHUNK, concat.len())]);
                (left, right) = hash_function(left, right);
            }
            root = if log_height as usize == dim[0] {
                left
            } else {
                hash_function(root, left).0
            }
        }
        if log_height > 0 {
            let sibling = proof[proof_index];
            let (left, right) = if sibling_is_on_right[proof_index] {
                (sibling, root)
            } else {
                (root, sibling)
            };
            root = hash_function(left, right).0;
        }
        log_height -= 1;
        proof_index += 1;
    }
    root
}

type F = BabyBear;

#[derive(Debug, Clone)]
struct VerifyBatchInstance {
    dim: Vec<usize>,
    opened: Vec<Vec<F>>,
    proof: Vec<[F; CHUNK]>,
    sibling_is_on_right: Vec<bool>,
    commit: [F; CHUNK],
}

fn random_instance(
    rng: &mut StdRng,
    row_lengths: Vec<Vec<usize>>,
    opened_element_size: usize,
    hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
) -> VerifyBatchInstance {
    let mut dims = vec![];
    let mut opened = vec![];
    let mut proof = vec![];
    let mut sibling_is_on_right = vec![];
    for (log_height, row_lengths) in row_lengths.iter().enumerate() {
        for &row_length in row_lengths {
            dims.push(log_height);
            let mut opened_row = vec![];
            for _ in 0..opened_element_size * row_length {
                opened_row.push(rng.random());
            }
            opened.push(opened_row);
        }
        if log_height > 0 {
            proof.push(std::array::from_fn(|_| rng.random()));
            sibling_is_on_right.push(rng.random());
        }
    }

    dims.reverse();
    opened.reverse();
    proof.reverse();
    sibling_is_on_right.reverse();

    let commit = compute_commit(&dims, &opened, &proof, &sibling_is_on_right, hash_function);

    VerifyBatchInstance {
        dim: dims,
        opened,
        proof,
        sibling_is_on_right,
        commit,
    }
}

const SBOX_REGISTERS: usize = 1;

#[derive(Clone)]
struct Case {
    row_lengths: Vec<Vec<usize>>,
    opened_element_size: usize,
}

fn set_and_execute<const SBOX_REGISTERS: usize>(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness<BabyBear, SBOX_REGISTERS>,
    rng: &mut StdRng,
    case: Case,
) {
    let instance = random_instance(
        rng,
        case.row_lengths,
        case.opened_element_size,
        |left, right| {
            let concatenated =
                std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
            let permuted = harness.executor.subchip.permute(concatenated);
            (
                std::array::from_fn(|i| permuted[i]),
                std::array::from_fn(|i| permuted[i + CHUNK]),
            )
        },
    );
    let VerifyBatchInstance {
        dim,
        opened,
        proof,
        sibling_is_on_right,
        commit,
    } = instance;

    let dim_register = gen_pointer(rng, 1);
    let opened_register = gen_pointer(rng, 1);
    let opened_length_register = gen_pointer(rng, 1);
    let proof_id = gen_pointer(rng, 1);
    let index_register = gen_pointer(rng, 1);
    let commit_register = gen_pointer(rng, 1);

    let dim_base_pointer = gen_pointer(rng, 1);
    let opened_base_pointer = gen_pointer(rng, 2);
    let index_base_pointer = gen_pointer(rng, 1);
    let commit_pointer = gen_pointer(rng, 1);

    let address_space = AS::Native as usize;
    tester.write_usize(address_space, dim_register, [dim_base_pointer]);
    tester.write_usize(address_space, opened_register, [opened_base_pointer]);
    tester.write_usize(address_space, opened_length_register, [opened.len()]);
    tester.write_usize(address_space, proof_id, [tester.streams.hint_space.len()]);
    tester.write_usize(address_space, index_register, [index_base_pointer]);
    tester.write_usize(address_space, commit_register, [commit_pointer]);

    for (i, &dim_value) in dim.iter().enumerate() {
        tester.write_usize(address_space, dim_base_pointer + i, [dim_value]);
    }
    for (i, opened_row) in opened.iter().enumerate() {
        let row_pointer = gen_pointer(rng, 1);
        tester.write_usize(
            address_space,
            opened_base_pointer + (2 * i),
            [row_pointer, opened_row.len() / case.opened_element_size],
        );
        for (j, &opened_value) in opened_row.iter().enumerate() {
            tester.write(address_space, row_pointer + j, [opened_value]);
        }
    }
    tester
        .streams
        .hint_space
        .push(proof.iter().flatten().copied().collect());
    for (i, &bit) in sibling_is_on_right.iter().enumerate() {
        tester.write(address_space, index_base_pointer + i, [F::from_bool(bit)]);
    }
    tester.write(address_space, commit_pointer, commit);

    let opened_element_size_inv = F::from_usize(case.opened_element_size)
        .inverse()
        .as_canonical_u32() as usize;
    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            VERIFY_BATCH.global_opcode(),
            [
                dim_register,
                opened_register,
                opened_length_register,
                proof_id,
                index_register,
                commit_register,
                opened_element_size_inv,
            ],
        ),
    );
}

fn test<const N: usize>(cases: [Case; N]) {
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let mut valid_tester = VmChipTestBuilder::default_native();
    let mut valid_harness = create_test_chip::<F, SBOX_REGISTERS>(&valid_tester);
    let mut prank_tester = VmChipTestBuilder::default_native();
    let mut prank_harness = create_test_chip::<F, SBOX_REGISTERS>(&prank_tester);

    let mut rng = create_seeded_rng();
    for case in cases {
        set_and_execute(
            &mut valid_tester,
            &mut valid_harness,
            &mut rng,
            case.clone(),
        );
        set_and_execute(&mut prank_tester, &mut prank_harness, &mut rng, case);
    }

    let valid_tester = valid_tester.build().load(valid_harness).finalize();
    valid_tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    let p2_chip = Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
    let inner_trace = p2_chip.generate_trace(vec![[F::ZERO; 2 * CHUNK]]);
    let inner_width = p2_chip.air.width();

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).expect("row exists").to_vec();
        trace_row[..inner_width].copy_from_slice(&inner_trace.values);
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    let prank_tester = prank_tester
        .build()
        .load_and_prank_trace(prank_harness, modify_trace)
        .finalize();

    // Run a test after pranking the poseidon2 stuff
    prank_tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
}

fn assert_poseidon2_local_air_rejects_trace(
    harness: &Harness<F, SBOX_REGISTERS>,
    trace: RowMajorMatrix<F>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let empty: &[F] = &[];
        let trace_view = trace.as_view();
        for row_idx in 0..trace.height() {
            let next_row_idx = (row_idx + 1) % trace.height();
            let local_row = trace_view.row_slice(row_idx).expect("row exists");
            let next_row = trace_view.row_slice(next_row_idx).expect("row exists");
            let mut builder = DebugConstraintBuilder::<BabyBearBlake3Config> {
                air_name: "NativePoseidon2Air",
                row_index: row_idx,
                preprocessed: VerticalPair::new(
                    RowMajorMatrixView::new_row(empty),
                    RowMajorMatrixView::new_row(empty),
                ),
                partitioned_main: vec![VerticalPair::new(
                    RowMajorMatrixView::new_row(local_row.as_ref()),
                    RowMajorMatrixView::new_row(next_row.as_ref()),
                )],
                after_challenge: vec![],
                challenges: &[],
                public_values: &[],
                exposed_values_after_challenge: &[],
                is_first_row: if row_idx == 0 { F::ONE } else { F::ZERO },
                is_last_row: if row_idx + 1 == trace.height() {
                    F::ONE
                } else {
                    F::ZERO
                },
                is_transition: if row_idx + 1 == trace.height() {
                    F::ZERO
                } else {
                    F::ONE
                },
                rap_phase_seq_kind: RapPhaseSeqKind::FriLogUp,
                has_common_main: true,
            };
            harness.air.eval(&mut builder);
        }
    }));

    assert!(
        result.is_err(),
        "expected local Poseidon2 AIR constraints to reject the trace"
    );
}

#[test]
fn verify_batch_test_felt() {
    test([Case {
        row_lengths: vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]],
        opened_element_size: 1,
    }]);
}

#[test]
fn verify_batch_test_felt_multiple() {
    test([
        Case {
            row_lengths: vec![vec![1, 1, 1, 2, 3], vec![9], vec![8]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![8], vec![7], vec![6]],
            opened_element_size: 1,
        },
    ])
}

#[test]
fn verify_batch_test_ext() {
    test([Case {
        row_lengths: vec![vec![3], vec![], vec![1, 2, 1], vec![4]],
        opened_element_size: 4,
    }]);
}

#[test]
fn verify_batch_test_ext_multiple() {
    test([
        Case {
            row_lengths: vec![vec![1, 1, 1], vec![3], vec![2]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![4], vec![3], vec![2]],
            opened_element_size: 4,
        },
    ])
}

#[test]
fn verify_batch_test_felt_and_ext() {
    test([
        Case {
            row_lengths: vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![1, 1, 1], vec![3], vec![2]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![8], vec![7], vec![6]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![4], vec![3], vec![2]],
            opened_element_size: 4,
        },
    ])
}

#[test]
fn verify_batch_rejects_unanchored_inside_row_start() {
    let tester = VmChipTestBuilder::default_native();
    let harness = create_test_chip::<F, SBOX_REGISTERS>(&tester);

    let air_width = harness.air.width();
    let inner_width = harness.executor.subchip.air.width();
    let mut trace = RowMajorMatrix::new(F::zero_vec(2 * air_width), air_width);

    // Keep the Poseidon2 witness valid and only prank the verify-batch columns.
    // Row 0 is disabled; row 1 is treated as the first row of an inside-row block.
    let left_inputs: [F; CHUNK] = std::array::from_fn(|i| F::from_usize(i + 1));
    let inside_state = std::array::from_fn(|i| if i < CHUNK { left_inputs[i] } else { F::ZERO });
    let inner_trace = harness
        .executor
        .subchip
        .generate_trace(vec![[F::ZERO; 2 * CHUNK], inside_state]);

    trace.row_mut(0)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(0).expect("row exists").as_ref());
    trace.row_mut(1)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(1).expect("row exists").as_ref());

    let row_1: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = trace.row_mut(1).borrow_mut();
    // This row starts an inside-row block, so cell 0 must be anchored to
    // opened_values[initial_opened_index][0].
    row_1.inside_row = F::ONE;
    row_1.end_inside_row = F::ONE;
    row_1.very_first_timestamp = F::from_u32(10);
    row_1.start_timestamp = F::from_u32(10);
    row_1.opened_element_size_inv = F::ONE;
    row_1.initial_opened_index = F::ZERO;

    let inside_specific: &mut InsideRowSpecificCols<F> =
        row_1.specific[..InsideRowSpecificCols::<u8>::width()].borrow_mut();
    let shared_mem_helper = tester.memory_helper();
    let mem_helper = shared_mem_helper.as_borrowed();
    let row_pointer_base = 100usize;
    let row_end = row_pointer_base + CHUNK;
    for i in 0..CHUNK {
        let cell = &mut inside_specific.cells[i];
        // Malicious witness: the block claims to start at opened_index = 0, but cell 0 is not
        // marked first-in-row, so the old AIR let it hash another contiguous memory segment.
        cell.opened_index = F::ZERO;
        cell.row_pointer = F::from_usize(row_pointer_base + i);
        cell.row_end = F::from_usize(row_end);
        cell.is_first_in_row = F::ZERO;
        mem_helper.fill(0, 11 + (2 * i) as u32, cell.read.as_mut());
    }

    assert_poseidon2_local_air_rejects_trace(&harness, trace);
}

#[test]
fn verify_batch_rejects_opened_index_advance_before_row_end() {
    let tester = VmChipTestBuilder::default_native();
    let harness = create_test_chip::<F, SBOX_REGISTERS>(&tester);

    let air_width = harness.air.width();
    let inner_width = harness.executor.subchip.air.width();
    let mut trace = RowMajorMatrix::new(F::zero_vec(4 * air_width), air_width);

    // Build two consecutive inside-row rows with a valid Poseidon transition. Only the
    // verify-batch row boundary bookkeeping is falsified.
    let row_1_values: [F; CHUNK] = std::array::from_fn(|i| F::from_usize(i + 1));
    let mut row_1_state = [F::ZERO; 2 * CHUNK];
    row_1_state[..CHUNK].copy_from_slice(&row_1_values);

    let row_1_output = harness.executor.subchip.permute(row_1_state);

    let row_2_values: [F; CHUNK] = std::array::from_fn(|i| F::from_usize(100 + i));
    let mut row_2_state = [F::ZERO; 2 * CHUNK];
    // The right half continues honestly across rows. Only the left half fakes the early jump.
    row_2_state[..CHUNK].copy_from_slice(&row_2_values);
    row_2_state[CHUNK..].copy_from_slice(&row_1_output[CHUNK..]);

    let inner_trace = harness.executor.subchip.generate_trace(vec![
        [F::ZERO; 2 * CHUNK],
        row_1_state,
        row_2_state,
        [F::ZERO; 2 * CHUNK],
    ]);

    trace.row_mut(0)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(0).expect("row exists").as_ref());
    trace.row_mut(1)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(1).expect("row exists").as_ref());
    trace.row_mut(2)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(2).expect("row exists").as_ref());
    trace.row_mut(3)[..inner_width]
        .copy_from_slice(inner_trace.row_slice(3).expect("row exists").as_ref());

    let shared_mem_helper = tester.memory_helper();
    let mem_helper = shared_mem_helper.as_borrowed();
    let opened_base_pointer = F::from_u32(200);
    let very_first_timestamp = F::from_u32(10);
    let row_1_start_timestamp_u32 = 10;
    let row_2_start_timestamp_u32 = row_1_start_timestamp_u32 + (2 * CHUNK) as u32;

    {
        let row_1: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = trace.row_mut(1).borrow_mut();
        row_1.inside_row = F::ONE;
        row_1.end_inside_row = F::ZERO;
        row_1.very_first_timestamp = very_first_timestamp;
        row_1.start_timestamp = F::from_u32(row_1_start_timestamp_u32);
        row_1.opened_element_size_inv = F::ONE;
        row_1.initial_opened_index = F::ZERO;
        row_1.opened_base_pointer = opened_base_pointer;

        let inside_specific: &mut InsideRowSpecificCols<F> =
            row_1.specific[..InsideRowSpecificCols::<u8>::width()].borrow_mut();
        let row_pointer_base = 100usize;
        // This row still has one unread element after CHUNK cells, so the next trace row must
        // continue the same opened row.
        let row_end = row_pointer_base + CHUNK + 1;
        for i in 0..CHUNK {
            let cell = &mut inside_specific.cells[i];
            cell.opened_index = F::ZERO;
            cell.row_pointer = F::from_usize(row_pointer_base + i);
            cell.row_end = F::from_usize(row_end);
            cell.is_first_in_row = if i == 0 { F::ONE } else { F::ZERO };
            if i == 0 {
                mem_helper.fill(
                    0,
                    row_1_start_timestamp_u32,
                    cell.read_row_pointer_and_length.as_mut(),
                );
            }
            mem_helper.fill(
                0,
                row_1_start_timestamp_u32 + (2 * i) as u32 + 1,
                cell.read.as_mut(),
            );
        }
    }

    {
        let row_2: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = trace.row_mut(2).borrow_mut();
        // Malicious witness: row 2 starts a new opened row even though row 1 has not reached
        // row_end, which skips the final unread element of opened row 0.
        row_2.inside_row = F::ONE;
        row_2.end_inside_row = F::ONE;
        row_2.very_first_timestamp = very_first_timestamp;
        row_2.start_timestamp = F::from_u32(row_2_start_timestamp_u32);
        row_2.opened_element_size_inv = F::ONE;
        row_2.initial_opened_index = F::ZERO;
        row_2.opened_base_pointer = opened_base_pointer;

        let inside_specific: &mut InsideRowSpecificCols<F> =
            row_2.specific[..InsideRowSpecificCols::<u8>::width()].borrow_mut();
        let row_pointer_base = 300usize;
        let row_end = row_pointer_base + CHUNK;
        for i in 0..CHUNK {
            let cell = &mut inside_specific.cells[i];
            // Pretend row 2 begins at opened row 1 immediately.
            cell.opened_index = F::ONE;
            cell.row_pointer = F::from_usize(row_pointer_base + i);
            cell.row_end = F::from_usize(row_end);
            cell.is_first_in_row = if i == 0 { F::ONE } else { F::ZERO };
            if i == 0 {
                mem_helper.fill(
                    0,
                    row_2_start_timestamp_u32,
                    cell.read_row_pointer_and_length.as_mut(),
                );
            }
            mem_helper.fill(
                0,
                row_2_start_timestamp_u32 + (2 * i) as u32 + 1,
                cell.read.as_mut(),
            );
        }
    }

    assert_poseidon2_local_air_rejects_trace(&harness, trace);
}

/// Create random instructions for the poseidon2 chip.
fn random_instructions(num_ops: usize) -> Vec<Instruction<BabyBear>> {
    let mut rng = create_seeded_rng();
    (0..num_ops)
        .map(|_| {
            let [a, b, c] = std::array::from_fn(|_| BabyBear::from_usize(gen_pointer(&mut rng, 1)));
            Instruction {
                opcode: if rng.random_bool(0.5) {
                    PERM_POS2
                } else {
                    COMP_POS2
                }
                .global_opcode(),
                a,
                b,
                c,
                d: BabyBear::from_usize(4),
                e: BabyBear::from_usize(4),
                f: BabyBear::ZERO,
                g: BabyBear::ZERO,
            }
        })
        .collect()
}

fn tester_with_random_poseidon2_ops(num_ops: usize) -> VmChipTester<BabyBearBlake3Config> {
    let elem_range = || 1..=100;

    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip::<F, SBOX_REGISTERS>(&tester);

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = Poseidon2Opcode::from_usize(
            instruction
                .opcode
                .local_opcode_idx(Poseidon2Opcode::CLASS_OFFSET),
        );
        let [a, b, c, d, e] = [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = gen_pointer(&mut rng, CHUNK) / 2;
        let lhs = gen_pointer(&mut rng, CHUNK) / 2;
        let rhs = gen_pointer(&mut rng, CHUNK) / 2;

        let data: [_; 2 * CHUNK] =
            std::array::from_fn(|_| BabyBear::from_usize(rng.random_range(elem_range())));

        let hash = harness.executor.subchip.permute(data);

        tester.write(d, a, [BabyBear::from_usize(dst)]);
        tester.write(d, b, [BabyBear::from_usize(lhs)]);
        if opcode == COMP_POS2 {
            tester.write(d, c, [BabyBear::from_usize(rhs)]);
        }

        let data_left: [_; CHUNK] = std::array::from_fn(|i| data[i]);
        let data_right: [_; CHUNK] = std::array::from_fn(|i| data[CHUNK + i]);
        match opcode {
            COMP_POS2 => {
                tester.write(e, lhs, data_left);
                tester.write(e, rhs, data_right);
            }
            PERM_POS2 => {
                tester.write(e, lhs, data_left);
                tester.write(e, lhs + CHUNK, data_right);
            }
        }

        tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

        match opcode {
            COMP_POS2 => {
                let expected: [_; CHUNK] = std::array::from_fn(|i| hash[i]);
                let actual = tester.read::<{ CHUNK }>(e, dst);
                assert_eq!(expected, actual);
            }
            PERM_POS2 => {
                let actual_0 = tester.read::<{ CHUNK }>(e, dst);
                let actual_1 = tester.read::<{ CHUNK }>(e, dst + CHUNK);
                let actual = [actual_0, actual_1].concat();
                assert_eq!(&hash, &actual[..]);
            }
        }
    }
    tester.build().load(harness).finalize()
}

fn get_engine() -> BabyBearBlake3Engine {
    BabyBearBlake3Engine::new(FriParameters::new_for_testing(3))
}

#[test]
fn verify_batch_chip_simple_1() {
    let tester = tester_with_random_poseidon2_ops(1);
    tester.test(get_engine).expect("Verification failed");
}

#[test]
fn verify_batch_chip_simple_3() {
    let tester = tester_with_random_poseidon2_ops(3);
    tester.test(get_engine).expect("Verification failed");
}

#[test]
fn verify_batch_chip_simple_50() {
    let tester = tester_with_random_poseidon2_ops(50);
    tester.test(get_engine).expect("Verification failed");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_vm_compress_poseidon2_as4() {
    let mut rng = create_seeded_rng();

    let mut instructions = vec![];

    let lhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [lhs_ptr + i]_4 <- rnd()
        instructions.push(Instruction::large_from_isize(
            FieldArithmeticOpcode::ADD.global_opcode(),
            lhs_ptr + i,
            rng.random_range(1usize..(1usize << 20)) as isize,
            0,
            4,
            0,
            0,
            0,
        ));
    }
    let rhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [rhs_ptr + i]_4 <- rnd()
        instructions.push(Instruction::large_from_isize(
            FieldArithmeticOpcode::ADD.global_opcode(),
            rhs_ptr + i,
            rng.random_range(1usize..(1usize << 20)) as isize,
            0,
            4,
            0,
            0,
            0,
        ));
    }
    let dst_ptr = gen_pointer(&mut rng, CHUNK) as isize;

    // [11]_4 <- lhs_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        11,
        lhs_ptr,
        0,
        4,
        0,
        0,
        0,
    ));

    // [22]_4 <- rhs_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        22,
        rhs_ptr,
        0,
        4,
        0,
        0,
        0,
    ));
    // [33]_4 <- dst_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        33,
        0,
        dst_ptr,
        4,
        0,
        0,
        0,
    ));

    instructions.push(Instruction::from_isize(
        COMP_POS2.global_opcode(),
        33,
        11,
        22,
        4,
        4,
    ));
    instructions.push(Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test(
        NativeCpuBuilder,
        NativeConfig::aggregation(0, 3),
        program.clone(),
    );
    air_test(
        NativeCpuBuilder,
        NativeConfig::aggregation(0, 7),
        program.clone(),
    );
}

// CUDA-specific tests
#[cfg(feature = "cuda")]
mod cuda_tests {
    use std::array::from_fn;

    use openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness};
    use test_case::test_case;

    use super::*;

    const MAX_INS_CAPACITY_GPU: usize = 128;
    const SBOX_REGISTERS_GPU: usize = 1;

    fn create_gpu_test_harness(
        tester: &GpuChipTestBuilder,
        config: Poseidon2Config<CudaF>,
    ) -> GpuTestChipHarness<
        CudaF,
        NativePoseidon2Executor<CudaF, SBOX_REGISTERS_GPU>,
        NativePoseidon2Air<CudaF, SBOX_REGISTERS_GPU>,
        NativePoseidon2ChipGpu<SBOX_REGISTERS_GPU>,
        NativePoseidon2Chip<CudaF, SBOX_REGISTERS_GPU>,
    > {
        let air = NativePoseidon2Air::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            VerifyBatchBus::new(7),
            config,
        );
        let executor = NativePoseidon2Executor::new(config);

        let cpu_chip = NativePoseidon2Chip::new(
            NativePoseidon2Filler::new(config),
            tester.dummy_memory_helper(),
        );

        let gpu_chip =
            NativePoseidon2ChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY_GPU)
    }

    #[test_case(Poseidon2Opcode::PERM_POS2)]
    #[test_case(Poseidon2Opcode::COMP_POS2)]
    fn test_cuda_poseidon2_chip_gpu(opcode: Poseidon2Opcode) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_gpu_test_harness(&tester, Poseidon2Config::default());

        for _ in 0..100 {
            let instruction = Instruction {
                opcode: opcode.global_opcode(),
                a: CudaF::from_usize(gen_pointer(&mut rng, 1)),
                b: CudaF::from_usize(gen_pointer(&mut rng, 1)),
                c: CudaF::from_usize(gen_pointer(&mut rng, 1)),
                d: CudaF::from_usize(4),
                e: CudaF::from_usize(4),
                f: CudaF::ZERO,
                g: CudaF::ZERO,
            };

            let dst = gen_pointer(&mut rng, CHUNK) / 2;
            let lhs = gen_pointer(&mut rng, CHUNK) / 2;
            let rhs = gen_pointer(&mut rng, CHUNK) / 2;

            let [a, b, c, d, e] = [
                instruction.a,
                instruction.b,
                instruction.c,
                instruction.d,
                instruction.e,
            ]
            .map(|elem| elem.as_canonical_u32() as usize);

            tester.write::<1>(d, a, [CudaF::from_usize(dst)]);
            tester.write::<1>(d, b, [CudaF::from_usize(lhs)]);
            if opcode == Poseidon2Opcode::COMP_POS2 {
                tester.write::<1>(d, c, [CudaF::from_usize(rhs)]);
            }

            let data_left: [_; CHUNK] = from_fn(|_| CudaF::from_usize(rng.random_range(1..=100)));
            let data_right: [_; CHUNK] = from_fn(|_| CudaF::from_usize(rng.random_range(1..=100)));
            match opcode {
                Poseidon2Opcode::COMP_POS2 => {
                    tester.write::<CHUNK>(e, lhs, data_left);
                    tester.write::<CHUNK>(e, rhs, data_right);
                }
                Poseidon2Opcode::PERM_POS2 => {
                    tester.write::<CHUNK>(e, lhs, data_left);
                    tester.write::<CHUNK>(e, lhs + CHUNK, data_right);
                }
            }

            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }

        type Record<'a> = NativePoseidon2RecordMut<'a, CudaF, SBOX_REGISTERS_GPU>;
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_cuda_verify_batch() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        const ADDRESS_SPACE: usize = AS::Native as usize;

        let config = Poseidon2Config::default();
        let hasher = Poseidon2SubChip::<CudaF, SBOX_REGISTERS_GPU>::new(config.constants);

        let mut harness = create_gpu_test_harness(&tester, config);

        let cases: [(Vec<Vec<usize>>, usize); 5] = [
            (vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]], 1),
            (vec![vec![1, 1, 1], vec![3], vec![2]], 4),
            (vec![vec![8], vec![7], vec![6]], 1),
            (vec![vec![], vec![], vec![], vec![1]], 4),
            (vec![vec![4], vec![3], vec![2]], 4),
        ];

        for (row_lengths, opened_element_size) in cases {
            let instance =
                random_instance(&mut rng, row_lengths, opened_element_size, |left, right| {
                    let concatenated =
                        std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
                    let permuted = hasher.permute(concatenated);
                    (
                        std::array::from_fn(|i| permuted[i]),
                        std::array::from_fn(|i| permuted[i + CHUNK]),
                    )
                });

            let VerifyBatchInstance {
                dim,
                opened,
                proof,
                sibling_is_on_right,
                commit,
            } = instance;

            let dim_register = gen_pointer(&mut rng, 1);
            let opened_register = gen_pointer(&mut rng, 1);
            let opened_length_register = gen_pointer(&mut rng, 1);
            let proof_id = gen_pointer(&mut rng, 1);
            let index_register = gen_pointer(&mut rng, 1);
            let commit_register = gen_pointer(&mut rng, 1);

            let dim_base_pointer = gen_pointer(&mut rng, 1);
            let opened_base_pointer = gen_pointer(&mut rng, 2);
            let index_base_pointer = gen_pointer(&mut rng, 1);
            let commit_pointer = gen_pointer(&mut rng, 1);

            tester.write_usize(ADDRESS_SPACE, dim_register, [dim_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, opened_register, [opened_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, opened_length_register, [opened.len()]);
            tester.write_usize(ADDRESS_SPACE, proof_id, [tester.streams.hint_space.len()]);
            tester.write_usize(ADDRESS_SPACE, index_register, [index_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, commit_register, [commit_pointer]);

            for (i, &dim_value) in dim.iter().enumerate() {
                tester.write_usize(ADDRESS_SPACE, dim_base_pointer + i, [dim_value]);
            }
            for (i, opened_row) in opened.iter().enumerate() {
                let row_pointer = gen_pointer(&mut rng, 1);
                tester.write_usize(
                    ADDRESS_SPACE,
                    opened_base_pointer + (2 * i),
                    [row_pointer, opened_row.len() / opened_element_size],
                );
                for (j, &opened_value) in opened_row.iter().enumerate() {
                    tester.write(ADDRESS_SPACE, row_pointer + j, [opened_value]);
                }
            }

            tester
                .streams
                .hint_space
                .push(proof.iter().flatten().copied().collect());
            for (i, &bit) in sibling_is_on_right.iter().enumerate() {
                tester.write(
                    ADDRESS_SPACE,
                    index_base_pointer + i,
                    [CudaF::from_bool(bit)],
                );
            }
            tester.write(ADDRESS_SPACE, commit_pointer, commit);

            let opened_element_size_inv = CudaF::from_usize(opened_element_size)
                .inverse()
                .as_canonical_u32() as usize;
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &Instruction::from_usize(
                    VERIFY_BATCH.global_opcode(),
                    [
                        dim_register,
                        opened_register,
                        opened_length_register,
                        proof_id,
                        index_register,
                        commit_register,
                        opened_element_size_inv,
                    ],
                ),
            );
        }

        type Record<'a> = NativePoseidon2RecordMut<'a, CudaF, SBOX_REGISTERS_GPU>;
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
