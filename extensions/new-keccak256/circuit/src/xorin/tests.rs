use openvm_circuit::{arch::{ExecutionBridge, PreflightExecutor, testing::{BITWISE_OP_LOOKUP_BUS, TestBuilder, TestChipHarness, VmChipTestBuilder}}, system::memory::{SharedMemoryHelper, offline_checker::MemoryBridge}, utils::get_random_message};
use openvm_circuit_primitives::bitwise_op_lookup::{BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip, SharedBitwiseOperationLookupChip};
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_new_keccak256_transpiler::Rv32NewKeccakOpcode;
use std::sync::Arc;

use crate::xorin::{XorinVmChip, XorinVmExecutor, XorinVmFiller, air::XorinVmAir};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use openvm_circuit::arch::Arena;
use rand::{Rng, rngs::StdRng};
use openvm_stark_backend::p3_field::FieldAlgebra;

type F = BabyBear;
type Harness = TestChipHarness<F, XorinVmExecutor, XorinVmAir, XorinVmChip<F>>;
use openvm_new_keccak256_transpiler::Rv32NewKeccakOpcode::XORIN;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (XorinVmAir, XorinVmExecutor, XorinVmChip<F>) {
    let air = XorinVmAir::new(
        execution_bridge, 
        memory_bridge, 
        bitwise_chip.bus(), 
        address_bits,
        Rv32NewKeccakOpcode::CLASS_OFFSET
    );

    let executor = XorinVmExecutor::new(Rv32NewKeccakOpcode::CLASS_OFFSET, address_bits);
    let chip = XorinVmChip::new(
        XorinVmFiller::new(bitwise_chip, address_bits),
        memory_helper
    );
    (air, executor, chip)
}

fn create_test_harness(
    tester: &mut VmChipTestBuilder<F>
) -> (
    Harness, 
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    )
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus));

    let (air, executor, chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits()
    );

    const MAX_TRACE_ROWS: usize = 4096; 

    let harness = Harness::with_capacity(executor, air, chip, MAX_TRACE_ROWS);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>> (
    tester: &mut impl TestBuilder<F>, 
    executor: &mut E, 
    arena: &mut RA, 
    rng: &mut StdRng, 
    opcode: Rv32NewKeccakOpcode,
) {
    const LEN: usize = 4;
    let rand_buffer = get_random_message(rng, LEN);
    let mut rand_buffer_arr = [0u8; LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    let rand_input = get_random_message(rng, LEN);
    let mut rand_input_arr = [0u8; LEN];
    rand_input_arr.copy_from_slice(&rand_input);

    println!("rand_buffer_arr {:?}", rand_buffer_arr);
    println!("rand_input_arr {:?}", rand_input_arr);

    use openvm_circuit::arch::testing::memory::gen_pointer;
    let rd = gen_pointer(rng, LEN);
    let rs1 = gen_pointer(rng, LEN);
    let rs2 = gen_pointer(rng, LEN);

    let buffer_ptr = gen_pointer(rng, LEN);
    let input_ptr = gen_pointer(rng, LEN);
    let length = LEN;

    println!("buffer_ptr {}", buffer_ptr);
    println!("input_ptr {}", input_ptr);
    
    tester.write(2, buffer_ptr, rand_buffer_arr.map(F::from_canonical_u8));
    tester.write(2, input_ptr, rand_input_arr.map(F::from_canonical_u8));

    tester.write(1, rd, buffer_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs1, input_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, length.to_le_bytes().map(F::from_canonical_u8));

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2])
    );

    let mut expected_output = [0u8; LEN];
    for i in 0..LEN {
        expected_output[i] = rand_buffer_arr[i] ^ rand_input_arr[i];
    }   

    println!("preparing to read");
    let mut actual_output: [F; 4] = tester.read(2, buffer_ptr);

    for i in 0..LEN {
        assert_eq!(F::from_canonical_u8(expected_output[i]), actual_output[i]);
    }
}   

#[test]
fn test_new_keccak() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena, 
        &mut rng, 
        XORIN
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");    

}