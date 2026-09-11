use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
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
    crate::LessThanImmChipGpu,
    openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    trace::generate_trace_from_postflight, LessThanImmAir, LessThanImmChip, LessThanImmCoreAir,
    LessThanImmCoreCols, LessThanImmExecutor, LessThanImmFiller,
};
use crate::{
    adapters::{BaseAluImmU16AdapterAir, U16_BITS},
    test_utils::rand_write_register_or_imm,
};

type F = BabyBear;
type Harness = TestChipHarness<F, LessThanImmExecutor, LessThanImmAir, LessThanImmChip<F>>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = LessThanImmAir::new(
        BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = LessThanImmExecutor::new(LessThanImmOpcode::CLASS_OFFSET);
    let chip = LessThanImmChip::new(
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
fn less_than_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for opcode in [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU] {
        for source in [0, 1, i64::MAX as u64, 1u64 << 63, u64::MAX] {
            for imm in [-2048, -1, 0, 1, 2047] {
                let (instruction, rd) = rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

                let mut result = [F::ZERO; REGISTER_NUM_LIMBS];
                result[0] = F::from_bool(expected(opcode, source, imm));
                assert_eq!(result, tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd));
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

#[test]
fn less_than_immediate_result_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    let (instruction, _) = rand_write_register_or_imm(
        &mut tester,
        0u64.to_le_bytes(),
        [0; REGISTER_NUM_LIMBS],
        Some(encode_i12(1)),
        LessThanImmOpcode::SLTI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut LessThanImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.cmp_result = F::ZERO;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("altered comparison result should fail");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    LessThanImmExecutor,
    LessThanImmAir,
    LessThanImmChipGpu,
    LessThanImmChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = LessThanImmAir::new(
        BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = LessThanImmExecutor::new(LessThanImmOpcode::CLASS_OFFSET);
    let cpu_chip = LessThanImmChip::new(
        LessThanImmFiller::new(range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = LessThanImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
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
            let (instruction, _) = rand_write_register_or_imm(
                &mut tester,
                source.to_le_bytes(),
                [0; REGISTER_NUM_LIMBS],
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
