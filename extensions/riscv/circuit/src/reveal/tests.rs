use openvm_circuit::arch::testing::{
    memory::gen_register_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::RngCore;

use super::{
    trace::generate_trace_from_postflight, RevealAir, RevealChip, RevealExecutor, RevealFiller,
};
use crate::test_utils::memory::{store_memory_config, F, MAX_INS_CAPACITY};

type Harness = TestChipHarness<F, RevealExecutor, RevealAir, RevealChip<F>>;

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let timestamp_max_bits = store_memory_config().timestamp_max_bits;
    let air = RevealAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        tester.system_port().public_values_bus,
        tester.range_checker().bus(),
        timestamp_max_bits,
    );
    let executor = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
    let chip = RevealChip::new(
        RevealFiller::new(tester.range_checker(), timestamp_max_bits),
        tester.memory_helper(),
    );
    Harness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        generate_trace_from_postflight,
    )
}

#[test]
fn reveal_appends_public_value() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_harness(&mut tester);

    let src_ptr = gen_register_pointer(&mut rng, REGISTER_NUM_LIMBS);
    let value = rng.next_u64().to_le_bytes();
    tester.write_bytes(REGISTER_AS as usize, src_ptr, value.map(F::from_u8));
    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &Instruction::from_usize(RevealOpcode::REVEAL.global_opcode(), [src_ptr]),
    );

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
#[should_panic(expected = "PublicValuesCapacityExceeded")]
fn reveal_rejects_capacity_overflow() {
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    tester.public_values = openvm_circuit::arch::PublicValuesState::new(1);
    let mut harness = create_harness(&mut tester);
    let src_ptr = REGISTER_NUM_LIMBS;
    tester.write_bytes(
        REGISTER_AS as usize,
        src_ptr,
        0u64.to_le_bytes().map(F::from_u8),
    );
    let instruction = Instruction::from_usize(RevealOpcode::REVEAL.global_opcode(), [src_ptr]);
    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);
    tester.execute(&mut harness.executor, &mut harness.preflight, &instruction);
}

#[test]
#[should_panic(expected = "statically valid")]
fn reveal_rejects_nonzero_unused_operand() {
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_harness(&mut tester);
    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &Instruction::from_usize(
            RevealOpcode::REVEAL.global_opcode(),
            [REGISTER_NUM_LIMBS, 1],
        ),
    );
}
