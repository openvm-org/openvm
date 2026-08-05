use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::LessThanImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64LessThanImmChipGpu,
    openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    trace::generate_trace_from_postflight, LessThanImmCoreAir, LessThanImmCoreCols,
    LessThanImmFiller, Rv64LessThanImmAir, Rv64LessThanImmChip, Rv64LessThanImmExecutor,
};
use crate::{
    adapters::{Rv64BaseAluImmU16AdapterAir, U16_BITS},
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness =
    TestChipHarness<F, Rv64LessThanImmExecutor, Rv64LessThanImmAir, Rv64LessThanImmChip<F>>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = Rv64LessThanImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThanImmExecutor::new(LessThanImmOpcode::CLASS_OFFSET);
    let chip = Rv64LessThanImmChip::new(
        LessThanImmFiller::new(range_checker),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, 64, generate_trace_from_postflight)
}

fn encode_i12(imm: i16) -> usize {
    debug_assert!((-2048..=2047).contains(&imm));
    (imm as i32 as u32 & 0x00ff_ffff) as usize
}

fn expected(opcode: LessThanImmOpcode, source: u64, imm: i16) -> bool {
    match opcode {
        LessThanImmOpcode::SLTI => (source as i64) < i64::from(imm),
        LessThanImmOpcode::SLTIU => source < (imm as i64 as u64),
    }
}

#[test]
fn rv64_less_than_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for opcode in [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU] {
        for source in [0, 1, i64::MAX as u64, 1u64 << 63, u64::MAX] {
            for imm in [-2048, -1, 0, 1, 2047] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

                let mut result = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
                result[0] = F::from_bool(expected(opcode, source, imm));
                assert_eq!(result, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd));
            }
        }
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

type CoreCols = LessThanImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS>;

/// Executes `num_ops` SLTI instructions with immediate `imm` against a zero source register, then
/// rewrites the core columns of `row` via `prank` and expects verification to fail.
///
/// The trace is padded to the next power of two, so a `row` at or past `num_ops` is a padding row.
fn run_negative_prank(num_ops: usize, imm: i16, row: usize, prank: impl Fn(&mut CoreCols)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    for _ in 0..num_ops {
        let (instruction, _) = rv64_rand_write_register_or_imm(
            &mut tester,
            0u64.to_le_bytes(),
            [0; RV64_REGISTER_NUM_LIMBS],
            Some(encode_i12(imm)),
            LessThanImmOpcode::SLTI.global_opcode().as_usize(),
            &mut rng,
        );
        tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);
    }

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let width = trace.width();
        let mut values = trace.values.clone();
        assert!(values.len() / width > row, "trace has no row {row}");
        let cols: &mut CoreCols = values[row * width..(row + 1) * width]
            .split_at_mut(adapter_width)
            .1
            .borrow_mut();
        prank(cols);
        *trace = RowMajorMatrix::new(values, width);
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked trace should fail verification");
}

/// Pranks the sole executed row, which is valid.
fn prank_valid_row(imm: i16, prank: impl Fn(&mut CoreCols)) {
    run_negative_prank(1, imm, 0, prank);
}

/// Pranks a padding row: three ops pad the height to four, so row 3 is unused.
fn prank_padding_row(prank: impl Fn(&mut CoreCols)) {
    run_negative_prank(3, -1, 3, prank);
}

#[test]
fn rv64_less_than_immediate_result_negative() {
    // 0 < 1 holds, so the honest cmp_result is 1 and clearing it must be rejected.
    prank_valid_row(1, |cols| cols.cmp_result = F::ZERO);
}

#[test]
fn rv64_less_than_immediate_opcode_mode_negative_tests() {
    // 1 claims SLTIU on a SLTI row; 3 is out of range; 0 marks the row invalid, so its
    // execution-bus interaction goes missing.
    for mode in [1, 3, 0] {
        prank_valid_row(-1, |cols| cols.opcode_mode = F::from_u32(mode));
    }
}

#[test]
fn rv64_less_than_immediate_padding_imm_sign_negative_test() {
    // With `is_signed = -1`, the AIR computes
    //   c_msb_f = imm_sign * ((2^16 - 1) - (-1) * 2^16) = imm_sign * (2^17 - 1).
    const C_MSB: u32 = (1u32 << (U16_BITS + 1)) - 1;
    // Low limb of a sign-extended immediate whose low 11 bits are zero.
    const C_LOW: u32 = 0xF800;

    prank_padding_row(|cols| {
        cols.imm_sign = F::ONE;
        // Match the sign-extended immediate limb for limb so every raw_diff vanishes.
        cols.b[0] = F::from_u32(C_LOW);
        for limb in &mut cols.b[1..BLOCK_FE_WIDTH - 1] {
            *limb = F::from_u16(u16::MAX);
        }
        // The top limb also feeds `b_diff = b[last] - b_msb_f`, which must be 0 or 2^16. Setting
        // both to C_MSB keeps b_diff = 0 and makes the top raw_diff zero.
        cols.b[BLOCK_FE_WIDTH - 1] = F::from_u32(C_MSB);
        cols.b_msb_f = F::from_u32(C_MSB);
    });
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64LessThanImmExecutor,
    Rv64LessThanImmAir,
    Rv64LessThanImmChipGpu,
    Rv64LessThanImmChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = Rv64LessThanImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThanImmExecutor::new(LessThanImmOpcode::CLASS_OFFSET);
    let cpu_chip = Rv64LessThanImmChip::new(
        LessThanImmFiller::new(range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LessThanImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64).with_trace_generators(
        generate_trace_from_postflight,
        |chip, program, transcript, plan| {
            chip.generate_proving_ctx_from_postflight(program, transcript, plan)
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_less_than_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);

    for opcode in [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU] {
        for (source, imm) in [
            (0, -2048),
            (1, -1),
            (i64::MAX as u64, 0),
            (1u64 << 63, 1),
            (u64::MAX, 2047),
        ] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                source.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(encode_i12(imm)),
                opcode.global_opcode().as_usize(),
                &mut rng,
            );
            tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);
        }
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
