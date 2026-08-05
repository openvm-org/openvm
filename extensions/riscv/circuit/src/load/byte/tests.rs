use std::{borrow::BorrowMut, sync::Arc};

#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    MemoryConfig,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::LoadStoreOpcode::{self, LOADBU};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::utils::create_seeded_rng;

use crate::{
    adapters::{bytes_to_u16_block, LoadByteAdapterAir, LoadByteAdapterFiller, BYTE_BITS},
    load::{
        byte::trace::generate_trace_from_postflight, common::load_write_data, LoadByteAir,
        LoadByteChip, LoadByteCoreAir, LoadByteCoreCols, LoadByteExecutor, LoadByteFiller,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{load::LoadByteChipGpu, test_utils::memory::dummy_range_checker};

type ByteHarness = TestChipHarness<F, LoadByteExecutor, LoadByteAir, LoadByteChip<F>>;

fn create_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    ByteHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(bitwise_bus));
    let air = LoadByteAir::new(
        LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = LoadByteChip::<F>::new(
        LoadByteFiller::new(
            LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        ByteHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            generate_trace_from_postflight,
        ),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_load_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            LOADBU,
            None,
            None,
            None,
            None,
        );
    }
    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_load_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, _) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADBU,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
        None,
    );
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_load_address_underflow_test() {
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, _) = create_byte_harness(&mut tester);
    let rs1_ptr = 8;
    tester.write_bytes(REGISTER_AS as usize, rs1_ptr, [F::ZERO; 8]);

    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &Instruction::from_usize(
            LOADBU.global_opcode(),
            [
                0,
                rs1_ptr,
                u16::MAX as usize,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
                0,
                1,
            ],
        ),
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = [
        bytes_to_u16_block([131, 74, 186, 29, 138, 45, 202, 76]),
        bytes_to_u16_block([0; 8]),
    ];
    for (shift, expected) in [
        (0, [131, 0, 0, 0, 0, 0, 0, 0]),
        (1, [74, 0, 0, 0, 0, 0, 0, 0]),
        (2, [186, 0, 0, 0, 0, 0, 0, 0]),
        (3, [29, 0, 0, 0, 0, 0, 0, 0]),
        (4, [138, 0, 0, 0, 0, 0, 0, 0]),
        (5, [45, 0, 0, 0, 0, 0, 0, 0]),
        (6, [202, 0, 0, 0, 0, 0, 0, 0]),
        (7, [76, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(
            load_write_data(LOADBU, read_data, shift),
            bytes_to_u16_block(expected)
        );
    }
}

#[test]
fn negative_split_opcode_role_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADBU,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut LoadByteCoreCols<F> = core_row.borrow_mut();
        core.selector[0] += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked byte load trace should fail");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuByteHarness =
    GpuTestChipHarness<F, LoadByteExecutor, LoadByteAir, LoadByteChipGpu, LoadByteChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = LoadByteAir::new(
        LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = LoadByteChip::<F>::new(
        LoadByteFiller::new(
            LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = LoadByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
        .with_trace_generators(
            generate_trace_from_postflight,
            |chip, program, transcript, plan| {
                chip.generate_proving_ctx_from_postflight(program, transcript, plan)
            },
        )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_rand_load_byte_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            LOADBU,
            None,
            None,
            None,
            Some(MEMORY_AS as usize),
        );
    }
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
