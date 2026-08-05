use std::sync::Arc;

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
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use super::{
    trace::generate_trace_from_postflight, RevealAdapterAir, RevealAir, RevealChip,
    RevealCoreAir, RevealExecutor, RevealFiller,
};
use crate::{
    adapters::BYTE_BITS,
    test_utils::memory::{F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{reveal::RevealChipGpu, test_utils::memory::dummy_range_checker};

type RevealHarness = TestChipHarness<F, RevealExecutor, RevealAir, RevealChip<F>>;

fn reveal_memory_config() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    config
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn reveal_gpu_memory_config() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << config.pointer_max_bits;
    config
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    RevealHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_bus));
    let air = RevealAir::new(
        RevealAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        RevealCoreAir::new(bitwise_chip.bus()),
    );
    let executor = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
    let chip = RevealChip::new(
        RevealFiller::new(tester.address_bits(), range_checker, bitwise_chip.clone()),
        tester.memory_helper(),
    );
    (
        RevealHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            generate_trace_from_postflight,
        ),
        (bitwise_chip.air, bitwise_chip),
    )
}

fn reveal_instruction(src_ptr: usize, base_ptr: usize, imm: i16) -> Instruction<F> {
    Instruction::from_usize(
        RevealOpcode::REVEAL.global_opcode(),
        [
            src_ptr,
            base_ptr,
            imm as u16 as usize,
            REGISTER_AS as usize,
            PUBLIC_VALUES_AS as usize,
            1,
            usize::from(imm.is_negative()),
        ],
    )
}

fn field_block(bytes: &[u8]) -> [F; 8] {
    <[u8; 8]>::try_from(bytes).unwrap().map(F::from_u8)
}

#[test]
fn reveal_preserves_legacy_unaligned_overwrite_semantics() {
    let mut tester = VmChipTestBuilder::from_config(reveal_memory_config());
    let (mut harness, bitwise) = create_harness(&mut tester);
    let base_ptr = REGISTER_NUM_LIMBS;
    let first_src = 2 * REGISTER_NUM_LIMBS;
    let second_src = 3 * REGISTER_NUM_LIMBS;
    let address = 39usize;
    let initial = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f,
    ];
    tester.write_bytes(PUBLIC_VALUES_AS as usize, 32, field_block(&initial[..8]));
    tester.write_bytes(PUBLIC_VALUES_AS as usize, 40, field_block(&initial[8..]));
    tester.write_bytes(
        REGISTER_AS as usize,
        base_ptr,
        (address as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        REGISTER_AS as usize,
        first_src,
        0x1122_3344_5566_7788u64.to_le_bytes().map(F::from_u8),
    );
    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &reveal_instruction(first_src, base_ptr, 0),
    );
    let mut expected = initial;
    expected[7..15].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(
        tester.read_bytes::<8>(PUBLIC_VALUES_AS as usize, 32),
        field_block(&expected[..8])
    );
    assert_eq!(
        tester.read_bytes::<8>(PUBLIC_VALUES_AS as usize, 40),
        field_block(&expected[8..])
    );

    tester.write_bytes(
        REGISTER_AS as usize,
        second_src,
        0xaabb_ccdd_eeff_0123u64.to_le_bytes().map(F::from_u8),
    );
    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &reveal_instruction(second_src, base_ptr, 0),
    );
    expected[7..15].copy_from_slice(&0xaabb_ccdd_eeff_0123u64.to_le_bytes());
    assert_eq!(
        tester.read_bytes::<8>(PUBLIC_VALUES_AS as usize, 32),
        field_block(&expected[..8])
    );
    assert_eq!(
        tester.read_bytes::<8>(PUBLIC_VALUES_AS as usize, 40),
        field_block(&expected[8..])
    );

    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuRevealHarness =
    GpuTestChipHarness<F, RevealExecutor, RevealAir, RevealChipGpu, RevealChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_gpu_harness(tester: &GpuChipTestBuilder) -> GpuRevealHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = RevealAir::new(
        RevealAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        RevealCoreAir::new(bitwise_chip.bus()),
    );
    let executor = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
    let cpu_chip = RevealChip::new(
        RevealFiller::new(tester.address_bits(), range_checker, bitwise_chip),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = RevealChipGpu::new(
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
fn test_cuda_reveal_tracegen_crossing_and_non_crossing() {
    let mut tester =
        GpuChipTestBuilder::new(reveal_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_gpu_harness(&tester);
    let base_ptr = REGISTER_NUM_LIMBS;
    let src_ptr = 2 * REGISTER_NUM_LIMBS;

    for address in [32u64, 39] {
        tester.write_bytes(
            REGISTER_AS as usize,
            base_ptr,
            address.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes(
            REGISTER_AS as usize,
            src_ptr,
            (address ^ 0xaabb_ccdd_eeff_0123)
                .to_le_bytes()
                .map(F::from_u8),
        );
        tester.execute(
            &mut harness.executor,
            &mut harness.preflight,
            &reveal_instruction(src_ptr, base_ptr, 0),
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
