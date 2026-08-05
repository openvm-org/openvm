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
use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::BitwiseLogicImmChipGpu,
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use super::{
    trace::generate_trace_from_postflight, BitwiseLogicImmAir, BitwiseLogicImmChip,
    BitwiseLogicImmCoreAir, BitwiseLogicImmCoreCols, BitwiseLogicImmExecutor,
    BitwiseLogicImmFiller,
};
use crate::{
    adapters::{BaseAluImmAdapterAir, BYTE_BITS},
    test_utils::rand_write_register_or_imm,
};

type F = BabyBear;
type Harness =
    TestChipHarness<F, BitwiseLogicImmExecutor, BitwiseLogicImmAir, BitwiseLogicImmChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    BitwiseLogicImmAir,
    BitwiseLogicImmExecutor,
    BitwiseLogicImmChip<F>,
) {
    let air = BitwiseLogicImmAir::new(
        BaseAluImmAdapterAir::new(execution_bridge, memory_bridge),
        BitwiseLogicImmCoreAir::new(bitwise_chip.bus(), BaseAluImmOpcode::CLASS_OFFSET),
    );
    let executor = BitwiseLogicImmExecutor::new(BaseAluImmOpcode::CLASS_OFFSET);
    let chip = BitwiseLogicImmChip::new(BitwiseLogicImmFiller::new(bitwise_chip), memory_helper);
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
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
        Harness::with_capacity(executor, air, cpu_chip, 64, generate_trace_from_postflight),
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
fn bitwise_immediate_boundaries() {
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
                let (instruction, rd) = rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);
                assert_eq!(
                    expected(opcode, source, imm).to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd),
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
fn bitwise_immediate_binding_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);
    let (instruction, _) = rand_write_register_or_imm(
        &mut tester,
        0x0123_4567_89ab_cdefu64.to_le_bytes(),
        [0; REGISTER_NUM_LIMBS],
        Some(encode_i12(-1)),
        BaseAluImmOpcode::XORI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BitwiseLogicImmCoreCols<F, REGISTER_NUM_LIMBS, BYTE_BITS> =
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

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    BitwiseLogicImmExecutor,
    BitwiseLogicImmAir,
    BitwiseLogicImmChipGpu,
    BitwiseLogicImmChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_bitwise = Arc::new(BitwiseOperationLookupChip::new(default_bitwise_lookup_bus()));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = BitwiseLogicImmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64).with_trace_generators(
        generate_trace_from_postflight,
        |chip, program, transcript, plan| {
            chip.generate_proving_ctx_from_postflight(program, transcript, plan)
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
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
            let (instruction, _) = rand_write_register_or_imm(
                &mut tester,
                0x0123_4567_89ab_cdefu64.to_le_bytes(),
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
