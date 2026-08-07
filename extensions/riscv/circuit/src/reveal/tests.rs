#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::testing::{
    default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    MemoryConfig,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use super::{
    trace::generate_trace_from_postflight, RevealAir, RevealChip, RevealExecutor, RevealFiller,
};
use crate::test_utils::memory::{F, MAX_INS_CAPACITY};
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

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> RevealHarness {
    let range_checker = tester.range_checker();
    let air = RevealAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        range_checker.bus(),
        tester.address_bits(),
    );
    let executor = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
    let chip = RevealChip::new(
        RevealFiller::new(tester.address_bits(), range_checker),
        tester.memory_helper(),
    );
    RevealHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        generate_trace_from_postflight,
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

fn field_block(bytes: &[u8]) -> [F; REGISTER_NUM_LIMBS] {
    <[u8; REGISTER_NUM_LIMBS]>::try_from(bytes)
        .unwrap()
        .map(F::from_u8)
}

fn write_public_value_word(tester: &mut VmChipTestBuilder<F>, ptr: usize, bytes: [u8; 8]) {
    for (block, chunk) in bytes.chunks_exact(4).enumerate() {
        tester.write(
            PUBLIC_VALUES_AS as usize,
            ptr + 4 * block,
            <[u8; 4]>::try_from(chunk).unwrap().map(F::from_u8),
        );
    }
}

fn read_public_value_word(tester: &mut VmChipTestBuilder<F>, ptr: usize) -> [F; 8] {
    let mut values = [F::ZERO; 8];
    for block in 0..2 {
        values[4 * block..4 * (block + 1)]
            .copy_from_slice(&tester.read::<4>(PUBLIC_VALUES_AS as usize, ptr + 4 * block));
    }
    values
}

#[test]
fn reveal_writes_and_overwrites_aligned_public_value() {
    let mut tester = VmChipTestBuilder::from_config(reveal_memory_config());
    let mut harness = create_harness(&mut tester);
    let base_ptr = REGISTER_NUM_LIMBS;
    let first_src = 2 * REGISTER_NUM_LIMBS;
    let second_src = 3 * REGISTER_NUM_LIMBS;
    let address = 40usize;
    let initial = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f,
    ];
    write_public_value_word(&mut tester, 32, initial[..8].try_into().unwrap());
    write_public_value_word(&mut tester, 40, initial[8..].try_into().unwrap());
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
    expected[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(
        read_public_value_word(&mut tester, 32),
        field_block(&expected[..8])
    );
    assert_eq!(
        read_public_value_word(&mut tester, 40),
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
    expected[8..16].copy_from_slice(&0xaabb_ccdd_eeff_0123u64.to_le_bytes());
    assert_eq!(
        read_public_value_word(&mut tester, 32),
        field_block(&expected[..8])
    );
    assert_eq!(
        read_public_value_word(&mut tester, 40),
        field_block(&expected[8..])
    );

    tester
        .build()
        .load(harness)
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
    let air = RevealAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        range_checker.bus(),
        tester.address_bits(),
    );
    let executor = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
    let cpu_chip = RevealChip::new(
        RevealFiller::new(tester.address_bits(), range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = RevealChipGpu::new(
        tester.range_checker(),
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
fn test_cuda_reveal_tracegen_aligned() {
    let mut tester =
        GpuChipTestBuilder::new(reveal_gpu_memory_config(), default_var_range_checker_bus());
    let mut harness = create_gpu_harness(&tester);
    let base_ptr = REGISTER_NUM_LIMBS;
    let src_ptr = 2 * REGISTER_NUM_LIMBS;

    for address in [32u64, 40] {
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
