use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::ShiftLogicalImmChipGpu,
    openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    trace::generate_trace_from_postflight, ShiftLogicalImmAir, ShiftLogicalImmChip,
    ShiftLogicalImmCoreAir, ShiftLogicalImmCoreCols, ShiftLogicalImmExecutor,
    ShiftLogicalImmFiller,
};
use crate::{
    adapters::{BaseAluImmU16AdapterAir, U16_BITS},
    test_utils::rand_write_register_or_imm,
};

type F = BabyBear;
type Harness =
    TestChipHarness<F, ShiftLogicalImmExecutor, ShiftLogicalImmAir, ShiftLogicalImmChip<F>>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = ShiftLogicalImmAir::new(
        BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftLogicalImmCoreAir::new(range_checker.bus(), ShiftImmOpcode::CLASS_OFFSET),
    );
    let executor = ShiftLogicalImmExecutor::new(ShiftImmOpcode::CLASS_OFFSET);
    let chip = ShiftLogicalImmChip::new(
        ShiftLogicalImmFiller::new(range_checker),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, 32, generate_trace_from_postflight)
}

#[test]
fn shift_logical_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for opcode in [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI] {
        for source in [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210] {
            for shamt in [0usize, 1, 15, 16, 31, 32, 63] {
                let (instruction, rd) = rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; REGISTER_NUM_LIMBS],
                    Some(shamt),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

                let result = match opcode {
                    ShiftImmOpcode::SLLI => source << shamt,
                    ShiftImmOpcode::SRLI => source >> shamt,
                    ShiftImmOpcode::SRAI => unreachable!(),
                };
                assert_eq!(
                    result.to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd),
                );
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
fn shift_logical_immediate_marker_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    let (instruction, _) = rand_write_register_or_imm(
        &mut tester,
        1u64.to_le_bytes(),
        [0; REGISTER_NUM_LIMBS],
        Some(1),
        ShiftImmOpcode::SLLI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftLogicalImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.bit_shift_marker = [F::ZERO; U16_BITS];
        cols.bit_shift_marker[2] = F::ONE;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("altered shift marker should fail");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    ShiftLogicalImmExecutor,
    ShiftLogicalImmAir,
    ShiftLogicalImmChipGpu,
    ShiftLogicalImmChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = ShiftLogicalImmAir::new(
        BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftLogicalImmCoreAir::new(range_checker.bus(), ShiftImmOpcode::CLASS_OFFSET),
    );
    let executor = ShiftLogicalImmExecutor::new(ShiftImmOpcode::CLASS_OFFSET);
    let cpu_chip = ShiftLogicalImmChip::new(
        ShiftLogicalImmFiller::new(range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = ShiftLogicalImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32).with_trace_generators(
        generate_trace_from_postflight,
        |chip, program, transcript, plan| {
            chip.generate_proving_ctx_from_postflight(program, transcript, plan)
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_shift_logical_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);

    for opcode in [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI] {
        for shamt in [0usize, 1, 15, 16, 31, 32, 63] {
            let (instruction, _) = rand_write_register_or_imm(
                &mut tester,
                0x0123_4567_89ab_cdefu64.to_le_bytes(),
                [0; REGISTER_NUM_LIMBS],
                Some(shamt),
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

mod word {
    use openvm_circuit::arch::testing::{TestBuilder, TestChipHarness, VmChipTestBuilder};
    use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
    use openvm_riscv_transpiler::ShiftWImmOpcode;
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    use {
        crate::ShiftWLogicalImmChipGpu,
        openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
        openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        std::sync::Arc,
    };

    use crate::{
        adapters::BaseAluWImmU16AdapterAir,
        shift_logical_imm::{
            trace::generate_word_trace_from_postflight, ShiftLogicalImmCoreAir,
            ShiftLogicalImmFiller, ShiftWLogicalImmAir, ShiftWLogicalImmChip,
            ShiftWLogicalImmExecutor,
        },
        test_utils::rand_write_register_or_imm,
    };

    type F = BabyBear;
    type Harness =
        TestChipHarness<F, ShiftWLogicalImmExecutor, ShiftWLogicalImmAir, ShiftWLogicalImmChip<F>>;

    fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
        let range_checker = tester.range_checker();
        let air = ShiftWLogicalImmAir::new(
            BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftLogicalImmCoreAir::new(range_checker.bus(), ShiftWImmOpcode::CLASS_OFFSET),
        );
        let executor = ShiftWLogicalImmExecutor::new(ShiftWImmOpcode::CLASS_OFFSET);
        let chip = ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(range_checker),
            tester.memory_helper(),
        );
        Harness::with_capacity(executor, air, chip, 32, generate_word_trace_from_postflight)
    }

    #[test]
    fn shift_w_logical_immediate_boundaries() {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let mut harness = create_harness(&tester);

        for opcode in [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW] {
            for source in [0xa5a5_a5a5_1234_5678u64, 0x5a5a_5a5a_8765_4321] {
                for shamt in [0usize, 1, 15, 16, 31] {
                    let (instruction, rd) = rand_write_register_or_imm(
                        &mut tester,
                        source.to_le_bytes(),
                        [0; REGISTER_NUM_LIMBS],
                        Some(shamt),
                        opcode.global_opcode().as_usize(),
                        &mut rng,
                    );
                    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

                    let word = source as u32;
                    let result = match opcode {
                        ShiftWImmOpcode::SLLIW => word << shamt,
                        ShiftWImmOpcode::SRLIW => word >> shamt,
                        ShiftWImmOpcode::SRAIW => unreachable!(),
                    };
                    let expected = (result as i32 as i64 as u64).to_le_bytes().map(F::from_u8);
                    assert_eq!(
                        expected,
                        tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd),
                        "{opcode:?} source={source:#018x} shamt={shamt}",
                    );
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

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness = GpuTestChipHarness<
        F,
        ShiftWLogicalImmExecutor,
        ShiftWLogicalImmAir,
        ShiftWLogicalImmChipGpu,
        ShiftWLogicalImmChip<F>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
        let range_checker = Arc::new(VariableRangeCheckerChip::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
        ));
        let air = ShiftWLogicalImmAir::new(
            BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftLogicalImmCoreAir::new(range_checker.bus(), ShiftWImmOpcode::CLASS_OFFSET),
        );
        let executor = ShiftWLogicalImmExecutor::new(ShiftWImmOpcode::CLASS_OFFSET);
        let cpu_chip = ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(range_checker),
            tester.dummy_memory_helper(),
        );
        let gpu_chip =
            ShiftWLogicalImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32)
            .with_trace_generators(
                generate_word_trace_from_postflight,
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_cuda_shift_w_logical_immediate_boundaries_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_cuda_harness(&tester);
        let source = 0xa5a5_a5a5_1234_5678u64.to_le_bytes();

        for opcode in [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW] {
            for shamt in [0usize, 1, 15, 16, 31] {
                let (instruction, _) = rand_write_register_or_imm(
                    &mut tester,
                    source,
                    [0; REGISTER_NUM_LIMBS],
                    Some(shamt),
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
}
