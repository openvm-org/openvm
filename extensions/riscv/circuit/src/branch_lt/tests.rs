#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::i32_to_f,
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_riscv_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits,
                RvrPreflightTranscript,
            },
            MatrixRecordArena, VmExecutor,
        },
        system::{
            cuda::memory::MemoryInventoryGPU,
            memory::online::{AddressMap, GuestMemory, TracingMemory},
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::Chip,
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        program::Program,
        riscv::RV64_REGISTER_AS,
        SystemOpcode,
    },
    openvm_stark_backend::prover::ColMajorMatrix,
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BranchAdapterRecord, BranchLessThanCoreRecord, Rv64BranchLessThanChipGpu,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{run_cmp, Rv64BranchLessThanChip};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64BranchAdapterAir,
        Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller, RV64_REGISTER_NUM_LIMBS,
        RV_B_TYPE_IMM_BITS, U16_BITS,
    },
    branch_lt::BranchLessThanCoreCols,
    test_utils::{rv64_marker_bytes_to_u16_marker, rv64_msb_byte_prank_to_u16_limb},
    BranchLessThanCoreAir, BranchLessThanFiller, Rv64BranchLessThanAir, Rv64BranchLessThanExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
type Harness = TestChipHarness<
    F,
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanAir,
    Rv64BranchLessThanChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BranchLessThanAir,
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanChip<F>,
) {
    let air = Rv64BranchLessThanAir::new(
        Rv64BranchAdapterAir::new(execution_bridge, memory_bridge),
        BranchLessThanCoreAir::new(range_checker_chip.bus(), BranchLessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BranchLessThanExecutor::new(
        Rv64BranchAdapterExecutor::new(),
        BranchLessThanOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BranchLessThanChip::new(
        BranchLessThanFiller::new(
            Rv64BranchAdapterFiller,
            range_checker_chip,
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BranchLessThanOpcode,
    a: Option<[u16; BLOCK_FE_WIDTH]>,
    b: Option<[u16; BLOCK_FE_WIDTH]>,
    imm: Option<i32>,
) {
    let a = a.unwrap_or(array::from_fn(|_| rng.random_range(0..=u16::MAX)));
    let b = b.unwrap_or(if rng.random_bool(0.5) {
        a
    } else {
        array::from_fn(|_| rng.random_range(0..=u16::MAX))
    });

    let imm = imm.unwrap_or(rng.random_range((-ABS_MAX_IMM)..ABS_MAX_IMM));
    let rs1 = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2 = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let a_bytes: [F; RV64_REGISTER_NUM_LIMBS] = rv64_u16_block_to_bytes(a).map(F::from_u8);
    let b_bytes: [F; RV64_REGISTER_NUM_LIMBS] = rv64_u16_block_to_bytes(b).map(F::from_u8);
    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs1, a_bytes);
    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs2, b_bytes);

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::from_isize(
            opcode.global_opcode(),
            rs1 as isize,
            rs2 as isize,
            imm as isize,
            1,
            1,
        ),
        rng.random_range(imm.unsigned_abs()..(1 << (PC_BITS - 1))),
    );

    let (cmp_result, _, _, _) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(opcode.local_usize() as u8, &a, &b);
    let from_pc = tester.last_from_pc().as_canonical_u32() as i32;
    let to_pc = tester.last_to_pc().as_canonical_u32() as i32;
    let pc_inc = if cmp_result { imm } else { 4 };

    assert_eq!(to_pc, from_pc + pc_inc);
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(BranchLessThanOpcode::BLT, 100)]
#[test_case(BranchLessThanOpcode::BLTU, 100)]
#[test_case(BranchLessThanOpcode::BGE, 100)]
#[test_case(BranchLessThanOpcode::BGEU, 100)]
fn rand_branch_lt_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    // Test special case where b = c
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(rv64_bytes_to_u16_block([
            101, 128, 202, 255, 255, 255, 255, 255,
        ])),
        Some(rv64_bytes_to_u16_block([
            101, 128, 202, 255, 255, 255, 255, 255,
        ])),
        Some(24),
    );
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(rv64_bytes_to_u16_block([36, 0, 0, 0, 0, 0, 0, 0])),
        Some(rv64_bytes_to_u16_block([36, 0, 0, 0, 0, 0, 0, 0])),
        Some(24),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct BranchLessThanPrankValues<const NUM_LIMBS: usize> {
    pub a_msb: Option<i32>,
    pub b_msb: Option<i32>,
    pub diff_marker: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_branch_lt_test(
    opcode: BranchLessThanOpcode,
    a: [u16; BLOCK_FE_WIDTH],
    b: [u16; BLOCK_FE_WIDTH],
    prank_cmp_result: bool,
    prank_vals: BranchLessThanPrankValues<BLOCK_FE_WIDTH>,
    _interaction_error: bool,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(a),
        Some(b),
        Some(imm),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let ge_opcode = opcode == BranchLessThanOpcode::BGE || opcode == BranchLessThanOpcode::BGEU;

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BranchLessThanCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(a_msb) = prank_vals.a_msb {
            cols.a_msb_f = i32_to_f(a_msb);
        }
        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(prank_cmp_result);
        cols.cmp_lt = F::from_bool(ge_opcode ^ prank_cmp_result);

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_blt_wrong_lt_cmp_negative_test() {
    let a = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let b = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let prank_vals = Default::default();
    // Canonical (a<b) cmp_result is true for BLT/BLTU and false for BGE/BGEU; prank to opposite.
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_wrong_ge_cmp_negative_test() {
    let a = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let b = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let prank_vals = Default::default();
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv64_blt_wrong_eq_cmp_negative_test() {
    let a = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let b = a;
    let prank_vals = Default::default();
    // Canonical (a==b) cmp_result is false for BLT/BLTU and true for BGE/BGEU; prank to opposite.
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv64_blt_fake_diff_val_negative_test() {
    let a = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let b = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let prank_vals = BranchLessThanPrankValues {
        diff_val: Some(F::NEG_ONE.as_canonical_u32()),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

#[test]
fn rv64_blt_zero_diff_val_negative_test() {
    let a = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let b = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 1, 0])),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

#[test]
fn rv64_blt_fake_diff_marker_negative_test() {
    let a = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let b = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([1, 0, 0, 0, 0, 0, 0, 0])),
        diff_val: Some(72),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_zero_diff_marker_negative_test() {
    let a = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let b = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 0])),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_signed_wrong_a_msb_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 34, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(rv64_msb_byte_prank_to_u16_limb(a_bytes, 206)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_signed_wrong_a_msb_sign_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 34, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(rv64_msb_byte_prank_to_u16_limb(a_bytes, 205)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
}

#[test]
fn rv64_blt_signed_wrong_b_msb_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 36, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b_bytes, 206)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
}

#[test]
fn rv64_blt_signed_wrong_b_msb_sign_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 36, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b_bytes, 205)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, true);
}

#[test]
fn rv64_blt_unsigned_wrong_a_msb_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 36, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(rv64_msb_byte_prank_to_u16_limb(a_bytes, 204)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv64_blt_unsigned_wrong_a_msb_sign_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 36, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(rv64_msb_byte_prank_to_u16_limb(a_bytes, -51)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, true);
}

#[test]
fn rv64_blt_unsigned_wrong_b_msb_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 34, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b_bytes, 206)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_unsigned_wrong_b_msb_sign_negative_test() {
    let a_bytes = [145, 56, 89, 100, 5, 34, 25, 205];
    let b_bytes = [73, 56, 89, 100, 5, 35, 25, 205];
    let a = rv64_bytes_to_u16_block(a_bytes);
    let b = rv64_bytes_to_u16_block(b_bytes);
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b_bytes, -51)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_harness(&mut tester);

    let x = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    set_and_execute(
        &mut tester,
        &mut chip.executor,
        &mut chip.arena,
        &mut rng,
        BranchLessThanOpcode::BLT,
        Some(x),
        Some(x),
        Some(8),
    );

    set_and_execute(
        &mut tester,
        &mut chip.executor,
        &mut chip.arena,
        &mut rng,
        BranchLessThanOpcode::BGE,
        Some(x),
        Some(x),
        Some(8),
    );
}

#[test]
fn run_cmp_unsigned_sanity_test() {
    let x = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let y = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let diff_u16_idx = 5 / 2; // old byte-limb diff index, packed into u16 cells
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BLTU as u8, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, diff_u16_idx);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BGEU as u8, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, diff_u16_idx);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_cmp_same_sign_sanity_test() {
    let x = rv64_bytes_to_u16_block([145, 56, 89, 100, 5, 34, 25, 205]);
    let y = rv64_bytes_to_u16_block([73, 56, 89, 100, 5, 35, 25, 205]);
    let diff_u16_idx = 5 / 2; // old byte-limb diff index, packed into u16 cells
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BLT as u8, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, diff_u16_idx);
    assert!(x_sign); // negative
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BGE as u8, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, diff_u16_idx);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_diff_sign_sanity_test() {
    let x = rv64_bytes_to_u16_block([0x2d, 0x23, 0x19, 0x37, 0, 0, 0, 0x37]);
    let y = rv64_bytes_to_u16_block([0xad, 0x22, 0x19, 0xcd, 0xff, 0xff, 0xff, 0xcd]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BLT as u8, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH - 1);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BGE as u8, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH - 1);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_eq_sanity_test() {
    let x = rv64_bytes_to_u16_block([0x2d, 0x23, 0x19, 0x37, 0, 0, 0, 0x37]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BLT as u8, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BLTU as u8, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BGE as u8, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(BranchLessThanOpcode::BGEU as u8, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH);
    assert_eq!(x_sign, y_sign);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanAir,
    Rv64BranchLessThanChipGpu,
    Rv64BranchLessThanChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip =
        Rv64BranchLessThanChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BranchLessThanOpcode::BLT, 100)]
#[test_case(BranchLessThanOpcode::BLTU, 100)]
#[test_case(BranchLessThanOpcode::BGE, 100)]
#[test_case(BranchLessThanOpcode::BGEU, 100)]
fn test_cuda_rand_branch_lt_tracegen(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
    let mut rng = create_seeded_rng();

    let mut harness = create_cuda_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BranchAdapterRecord,
        &'a mut BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_branch_lt_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let branch = |opcode: BranchLessThanOpcode, rs1: usize, rs2: usize, immediate: isize| {
        Instruction::<F>::from_isize(
            opcode.global_opcode(),
            reg(rs1) as isize,
            reg(rs2) as isize,
            immediate,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        )
    };
    let instructions = [
        // Taken, and skips the unexecuted row at PC=4.
        branch(BranchLessThanOpcode::BLT, 1, 2, 8),
        branch(BranchLessThanOpcode::BLT, 0, 0, 4),
        // Signed and unsigned interpretations of the same source words disagree.
        branch(BranchLessThanOpcode::BLTU, 1, 2, 4),
        branch(BranchLessThanOpcode::BGE, 1, 2, 4),
        branch(BranchLessThanOpcode::BGEU, 1, 2, 4),
        // Exercise x0 on either side and aliased reads, including a non-x0 predecessor chain.
        branch(BranchLessThanOpcode::BLT, 0, 2, 4),
        branch(BranchLessThanOpcode::BLTU, 2, 0, 4),
        branch(BranchLessThanOpcode::BGE, 0, 0, 4),
        branch(BranchLessThanOpcode::BGEU, 1, 1, 4),
        // The negative field-encoded target is not selected, so this falls through.
        branch(BranchLessThanOpcode::BGE, 3, 2, -4),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let initial_registers = [(1usize, u64::MAX), (2, 1), (3, i64::MIN as u64)];
    let init_memory: SparseMemoryImage = initial_registers
        .into_iter()
        .flat_map(|(register, value)| {
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(move |(offset, byte)| {
                    ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)
                })
        })
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(10, 18))
        .unwrap();

    let mut tester = GpuChipTestBuilder::default();
    let mut initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    initial_image.memory.set_from_sparse(&init_memory);
    tester.memory.memory = TracingMemory::from_image(initial_image);
    let device_ctx = tester.range_checker().device_ctx.clone();
    let hasher_chip = tester.memory.hasher_chip.clone().unwrap();
    tester.memory.inventory = MemoryInventoryGPU::new(
        tester.memory.config.clone(),
        hasher_chip,
        device_ctx.clone(),
    );
    tester
        .memory
        .inventory
        .set_initial_memory(&tester.memory.memory.data().memory);
    let mut harness = create_cuda_harness(&tester);
    for &instruction_index in &[0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instructions[instruction_index],
            instruction_index as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BranchAdapterRecord,
        &'a mut BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchLessThanOpcode::BLT.global_opcode())
            .len(),
        2
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchLessThanOpcode::BLTU.global_opcode())
            .len(),
        2
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchLessThanOpcode::BGE.global_opcode())
            .len(),
        3
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchLessThanOpcode::BGEU.global_opcode())
            .len(),
        2
    );
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let raw_count = |count: &F| {
        const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<u32>()) };
        // The CUDA range-check buffer is typed as `F` for shared ownership but kernels update it as
        // an atomic `u32` histogram.
        unsafe { *(std::ptr::from_ref(count).cast::<u32>()) }
    };

    // A minimal terminated transcript starts at PC=4, takes a field-encoded -4 BGE target to the
    // TERMINATE at PC=0, and uses two aliased x0 reads. This exercises the selected negative
    // target, not only a negative immediate on a fallthrough row.
    let negative_program = Program::from_instructions(&[
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        branch(BranchLessThanOpcode::BGE, 0, 0, -4),
    ]);
    let alias_timestamp = execution
        .transcript
        .program_log
        .iter()
        .find(|event| event.pc == 28)
        .unwrap()
        .timestamp;
    let alias_read_index = execution
        .transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == alias_timestamp)
        .unwrap();
    let mut negative_from = execution.transcript.program_log[0];
    negative_from.pc = 4;
    negative_from.timestamp = 1;
    let mut negative_terminate = execution.transcript.program_log[0];
    negative_terminate.pc = 0;
    negative_terminate.timestamp = 3;
    let mut negative_first_read = execution.transcript.memory_log[alias_read_index];
    negative_first_read.timestamp = 1;
    let mut negative_second_read = execution.transcript.memory_log[alias_read_index + 1];
    negative_second_read.timestamp = 2;
    let negative_transcript = RvrPreflightTranscript {
        program_log: vec![negative_from, negative_terminate, negative_terminate],
        memory_log: vec![negative_first_read, negative_second_read],
        initial_write_log: Vec::new(),
    };
    let d_negative_program =
        GpuRvrProgram::upload(&negative_program, &memory_config, device_ctx).unwrap();
    let (d_negative, d_negative_plan) = d_negative_program
        .upload_transcript(&negative_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let negative_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    Rv64BranchLessThanChipGpu::new(negative_range_checker.clone(), tester.timestamp_max_bits())
        .generate_proving_ctx_from_rvr(&d_negative_program, &d_negative, &d_negative_plan)
        .unwrap();
    assert_eq!(d_negative.error_code().unwrap(), 0);
    assert_eq!(
        negative_range_checker
            .count
            .to_host_on(device_ctx)
            .unwrap()
            .iter()
            .map(raw_count)
            .sum::<u32>(),
        6
    );

    let run_corrupt =
        |transcript: RvrPreflightTranscript, expected_error: u32, expected_lookup_count: u32| {
            let corrupt_range_checker = Arc::new(
                openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
                    openvm_circuit::arch::testing::default_var_range_checker_bus(),
                    device_ctx.clone(),
                ),
            );
            let corrupt_chip = Rv64BranchLessThanChipGpu::new(
                corrupt_range_checker.clone(),
                tester.timestamp_max_bits(),
            );
            let (d_corrupt, d_corrupt_plan) = d_program
                .upload_transcript(&transcript, RvrPreflightEndpoint::Terminated)
                .unwrap();
            corrupt_chip
                .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
                .unwrap();
            assert_eq!(d_corrupt.error_code().unwrap(), expected_error);
            assert_eq!(
                corrupt_range_checker
                    .count
                    .to_host_on(device_ctx)
                    .unwrap()
                    .iter()
                    .map(raw_count)
                    .sum::<u32>(),
                expected_lookup_count,
                "the rejected row must not update the shared lookup histogram"
            );
        };

    let final_row_timestamp = execution
        .transcript
        .program_log
        .iter()
        .find(|event| event.pc == 36)
        .unwrap()
        .timestamp;
    let final_unique_read = execution
        .transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == final_row_timestamp)
        .unwrap();

    // Change the final BGE's unique rs1 value from i64::MIN to two. The logged row falls through,
    // but the corrupted comparison takes the field-encoded -4 target. Using the final unique read
    // avoids also corrupting a later event's predecessor chain.
    let mut target_corrupt = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    target_corrupt.memory_log[final_unique_read].value = [2, 0, 0, 0];
    run_corrupt(target_corrupt, 18, 54);

    let mut u16_corrupt = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    u16_corrupt.memory_log[final_unique_read].value[0] = 1 << 16;
    run_corrupt(u16_corrupt, 17, 54);

    let mut schedule_corrupt = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    // Point the read at an otherwise-unused, canonically aligned register block. This reaches the
    // branch kernel's exact schedule check instead of being rejected earlier by generic postflight
    // address validation or perturbing another row's predecessor chain.
    schedule_corrupt.memory_log[0].pointer = (reg(31) / 2) as u32;
    run_corrupt(schedule_corrupt, 16, 54);

    // On the non-x0 rs1 == rs2 row, change only the second read while preserving BGEU=true.
    // Target validation therefore passes, but the second read no longer matches its immediate
    // predecessor.
    let mut predecessor_corrupt = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let alias_timestamp = predecessor_corrupt
        .program_log
        .iter()
        .find(|event| event.pc == 32)
        .unwrap()
        .timestamp;
    let second_alias_read = predecessor_corrupt
        .memory_log
        .iter()
        .position(|event| event.timestamp == alias_timestamp + 1)
        .unwrap();
    predecessor_corrupt.memory_log[second_alias_read].value[0] ^= 1;
    run_corrupt(predecessor_corrupt, 19, 55);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_chip =
        Rv64BranchLessThanChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits());
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );
    // Four timestamp lookups and two MSB lookups per row, plus one non-zero-difference lookup for
    // each of the seven unequal rows.
    assert_eq!(replay_counts.iter().map(raw_count).sum::<u32>(), 9 * 6 + 7);

    let expected_trace = <Rv64BranchLessThanChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<SC>,
    >>::generate_proving_ctx(&harness.cpu_chip, harness.matrix_arena)
    .common_main;
    let replay_trace = transport_matrix_d2h_row_major(&replay_ctx.common_main, device_ctx).unwrap();
    let canonical_rows = |matrix: &RowMajorMatrix<F>| {
        let mut rows = (0..matrix.height())
            .map(|row| matrix.row_slice(row).unwrap().to_vec())
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row[1].as_canonical_u32());
        rows
    };
    assert_eq!(
        canonical_rows(&expected_trace),
        canonical_rows(&replay_trace)
    );
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR BLT/BLTU/BGE/BGEU transcript replay proof failed");
}
