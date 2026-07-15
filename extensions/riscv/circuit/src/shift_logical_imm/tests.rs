use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BaseAluImmU16AdapterRecord, Rv64ShiftLogicalImmChipGpu,
        ShiftLogicalImmCoreRecord,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    Rv64ShiftLogicalImmAir, Rv64ShiftLogicalImmChip, Rv64ShiftLogicalImmExecutor,
    ShiftLogicalImmCoreAir, ShiftLogicalImmCoreCols, ShiftLogicalImmFiller,
};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, U16_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv64ShiftLogicalImmExecutor,
    Rv64ShiftLogicalImmAir,
    Rv64ShiftLogicalImmChip<F>,
>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = Rv64ShiftLogicalImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftLogicalImmCoreAir::new(
            range_checker.bus(),
            ShiftImmOpcode::CLASS_OFFSET,
            ShiftImmOpcode::SLLI as usize,
            ShiftImmOpcode::SRLI as usize,
        ),
    );
    let executor = Rv64ShiftLogicalImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        ShiftImmOpcode::CLASS_OFFSET,
        ShiftImmOpcode::SLLI as usize,
        ShiftImmOpcode::SRLI as usize,
    );
    let chip = Rv64ShiftLogicalImmChip::new(
        ShiftLogicalImmFiller::new(
            Rv64BaseAluImmU16AdapterFiller::new(),
            range_checker,
            ShiftImmOpcode::SLLI as usize,
        ),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, 32)
}

#[test]
fn rv64_shift_logical_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for opcode in [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI] {
        for source in [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210] {
            for shamt in [0usize, 1, 15, 16, 31, 32, 63] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(shamt),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

                let result = match opcode {
                    ShiftImmOpcode::SLLI => source << shamt,
                    ShiftImmOpcode::SRLI => source >> shamt,
                    ShiftImmOpcode::SRAI => unreachable!(),
                };
                assert_eq!(
                    result.to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
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
fn rv64_shift_logical_immediate_marker_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    let (instruction, _) = rv64_rand_write_register_or_imm(
        &mut tester,
        1u64.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(1),
        ShiftImmOpcode::SLLI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

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

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64ShiftLogicalImmExecutor,
    Rv64ShiftLogicalImmAir,
    Rv64ShiftLogicalImmChipGpu,
    Rv64ShiftLogicalImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = Rv64ShiftLogicalImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftLogicalImmCoreAir::new(
            range_checker.bus(),
            ShiftImmOpcode::CLASS_OFFSET,
            ShiftImmOpcode::SLLI as usize,
            ShiftImmOpcode::SRLI as usize,
        ),
    );
    let executor = Rv64ShiftLogicalImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        ShiftImmOpcode::CLASS_OFFSET,
        ShiftImmOpcode::SLLI as usize,
        ShiftImmOpcode::SRLI as usize,
    );
    let cpu_chip = Rv64ShiftLogicalImmChip::new(
        ShiftLogicalImmFiller::new(
            Rv64BaseAluImmU16AdapterFiller::new(),
            range_checker,
            ShiftImmOpcode::SLLI as usize,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip =
        Rv64ShiftLogicalImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_shift_logical_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);

    for opcode in [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI] {
        for shamt in [0usize, 1, 15, 16, 31, 32, 63] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                0x0123_4567_89ab_cdefu64.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(shamt),
                opcode.global_opcode().as_usize(),
                &mut rng,
            );
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut ShiftLogicalImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

mod word {
    use openvm_circuit::arch::testing::{TestBuilder, TestChipHarness, VmChipTestBuilder};
    use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
    use openvm_riscv_transpiler::ShiftWImmOpcode;
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    #[cfg(feature = "cuda")]
    use {
        crate::{
            adapters::{Rv64BaseAluWImmU16AdapterRecord, RV64_WORD_U16_LIMBS, U16_BITS},
            Rv64ShiftWLogicalImmChipGpu, ShiftLogicalImmCoreRecord,
        },
        openvm_circuit::arch::{
            testing::{GpuChipTestBuilder, GpuTestChipHarness},
            EmptyAdapterCoreLayout,
        },
        openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        std::sync::Arc,
    };

    use super::super::{
        Rv64ShiftWLogicalImmAir, Rv64ShiftWLogicalImmChip, Rv64ShiftWLogicalImmExecutor,
        ShiftLogicalImmCoreAir, ShiftLogicalImmFiller,
    };
    use crate::{
        adapters::{
            Rv64BaseAluWImmU16AdapterAir, Rv64BaseAluWImmU16AdapterExecutor,
            Rv64BaseAluWImmU16AdapterFiller,
        },
        test_utils::rv64_rand_write_register_or_imm,
    };

    type F = BabyBear;
    type Harness = TestChipHarness<
        F,
        Rv64ShiftWLogicalImmExecutor,
        Rv64ShiftWLogicalImmAir,
        Rv64ShiftWLogicalImmChip<F>,
    >;

    fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
        let range_checker = tester.range_checker();
        let air = Rv64ShiftWLogicalImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftLogicalImmCoreAir::new(
                range_checker.bus(),
                ShiftWImmOpcode::CLASS_OFFSET,
                ShiftWImmOpcode::SLLIW as usize,
                ShiftWImmOpcode::SRLIW as usize,
            ),
        );
        let executor = Rv64ShiftWLogicalImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
            ShiftWImmOpcode::SLLIW as usize,
            ShiftWImmOpcode::SRLIW as usize,
        );
        let chip = Rv64ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker,
                ShiftWImmOpcode::SLLIW as usize,
            ),
            tester.memory_helper(),
        );
        Harness::with_capacity(executor, air, chip, 32)
    }

    #[test]
    fn rv64_shift_w_logical_immediate_boundaries() {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let mut harness = create_harness(&tester);

        for opcode in [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW] {
            for source in [0xa5a5_a5a5_1234_5678u64, 0x5a5a_5a5a_8765_4321] {
                for shamt in [0usize, 1, 15, 16, 31] {
                    let (instruction, rd) = rv64_rand_write_register_or_imm(
                        &mut tester,
                        source.to_le_bytes(),
                        [0; RV64_REGISTER_NUM_LIMBS],
                        Some(shamt),
                        opcode.global_opcode().as_usize(),
                        &mut rng,
                    );
                    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

                    let word = source as u32;
                    let result = match opcode {
                        ShiftWImmOpcode::SLLIW => word << shamt,
                        ShiftWImmOpcode::SRLIW => word >> shamt,
                        ShiftWImmOpcode::SRAIW => unreachable!(),
                    };
                    let expected = (result as i32 as i64 as u64).to_le_bytes().map(F::from_u8);
                    assert_eq!(
                        expected,
                        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
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

    #[cfg(feature = "cuda")]
    type GpuHarness = GpuTestChipHarness<
        F,
        Rv64ShiftWLogicalImmExecutor,
        Rv64ShiftWLogicalImmAir,
        Rv64ShiftWLogicalImmChipGpu,
        Rv64ShiftWLogicalImmChip<F>,
    >;

    #[cfg(feature = "cuda")]
    fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
        let range_checker = Arc::new(VariableRangeCheckerChip::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
        ));
        let air = Rv64ShiftWLogicalImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftLogicalImmCoreAir::new(
                range_checker.bus(),
                ShiftWImmOpcode::CLASS_OFFSET,
                ShiftWImmOpcode::SLLIW as usize,
                ShiftWImmOpcode::SRLIW as usize,
            ),
        );
        let executor = Rv64ShiftWLogicalImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
            ShiftWImmOpcode::SLLIW as usize,
            ShiftWImmOpcode::SRLIW as usize,
        );
        let cpu_chip = Rv64ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker,
                ShiftWImmOpcode::SLLIW as usize,
            ),
            tester.dummy_memory_helper(),
        );
        let gpu_chip =
            Rv64ShiftWLogicalImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_shift_w_logical_immediate_boundaries_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_cuda_harness(&tester);
        let source = 0xa5a5_a5a5_1234_5678u64.to_le_bytes();

        for opcode in [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW] {
            for shamt in [0usize, 1, 15, 16, 31] {
                let (instruction, _) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source,
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(shamt),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(
                    &mut harness.executor,
                    &mut harness.dense_arena,
                    &instruction,
                );
            }
        }

        type Record<'a> = (
            &'a mut Rv64BaseAluWImmU16AdapterRecord,
            &'a mut ShiftLogicalImmCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv64BaseAluWImmU16AdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
