use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        ExecutionBridge,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::BaseAluImmOpcode;
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
        adapters::Rv64BaseAluImmAdapterRecord, BitwiseLogicImmCoreRecord,
        Rv64BitwiseLogicImmChipGpu,
    },
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{
    BitwiseLogicImmCoreAir, BitwiseLogicImmCoreCols, BitwiseLogicImmFiller, Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChip, Rv64BitwiseLogicImmExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluImmAdapterAir, Rv64BaseAluImmAdapterExecutor, Rv64BaseAluImmAdapterFiller,
        RV64_BYTE_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmChip<F>,
) {
    let air = Rv64BitwiseLogicImmAir::new(
        Rv64BaseAluImmAdapterAir::new(execution_bridge, memory_bridge),
        BitwiseLogicImmCoreAir::new(bitwise_chip.bus(), BaseAluImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BitwiseLogicImmExecutor::new(
        Rv64BaseAluImmAdapterExecutor::new(),
        BaseAluImmOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BitwiseLogicImmChip::new(
        BitwiseLogicImmFiller::new(Rv64BaseAluImmAdapterFiller::new(), bitwise_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let chip = Arc::new(BitwiseOperationLookupChip::new(
        BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        chip.clone(),
        tester.memory_helper(),
    );
    (
        Harness::with_capacity(executor, air, cpu_chip, 64),
        (chip.air, chip),
    )
}

fn encode_i12(imm: i16) -> usize {
    debug_assert!((-2048..=2047).contains(&imm));
    (imm as i32 as u32 & 0x00ff_ffff) as usize
}

fn expected(opcode: BaseAluImmOpcode, source: u64, imm: i16) -> u64 {
    let imm = imm as i64 as u64;
    match opcode {
        BaseAluImmOpcode::XORI => source ^ imm,
        BaseAluImmOpcode::ORI => source | imm,
        BaseAluImmOpcode::ANDI => source & imm,
        BaseAluImmOpcode::ADDI => unreachable!(),
    }
}

#[test]
fn rv64_bitwise_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    for opcode in [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ] {
        for source in [0, 0x0123_4567_89ab_cdef, u64::MAX] {
            for imm in [-2048, -1, 0, 2047] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.arena, &instruction);
                assert_eq!(
                    expected(opcode, source, imm).to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
                );
            }
        }
    }

    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[test]
fn rv64_bitwise_immediate_binding_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);
    let (instruction, _) = rv64_rand_write_register_or_imm(
        &mut tester,
        0x0123_4567_89ab_cdefu64.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(encode_i12(-1)),
        BaseAluImmOpcode::XORI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BitwiseLogicImmCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.imm_sign = F::ZERO;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("altered immediate witness should fail");
}

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChipGpu,
    Rv64BitwiseLogicImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_bitwise = Arc::new(BitwiseOperationLookupChip::new(default_bitwise_lookup_bus()));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BitwiseLogicImmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        #[cfg(feature = "rvr")]
        Default::default(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitwise_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_harness(&tester);

    for opcode in [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ] {
        for imm in [-2048, -1, 0, 2047] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                0x0123_4567_89ab_cdefu64.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(encode_i12(imm)),
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
        &'a mut Rv64BaseAluImmAdapterRecord,
        &'a mut BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
