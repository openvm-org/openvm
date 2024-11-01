use std::sync::Arc;

use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use ax_stark_sdk::utils::create_seeded_rng;
use axvm_instructions::{
    riscv::RV32_CELL_BITS, BaseAluOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{Rv32BaseAlu256Chip, Rv32LessThan256Chip, Rv32Multiplication256Chip, Rv32Shift256Chip};
use crate::{
    arch::{
        testing::VmChipTestBuilder, InstructionExecutor, BITWISE_OP_LOOKUP_BUS,
        RANGE_TUPLE_CHECKER_BUS,
    },
    rv32im::{
        adapters::{Rv32HeapAdapterChip, INT256_NUM_LIMBS},
        BaseAluCoreChip, LessThanCoreChip, MultiplicationCoreChip, ShiftCoreChip,
    },
    utils::{generate_long_number, rv32_write_heap_default},
};

type F = BabyBear;

fn run_int_256_rand_execute<E: InstructionExecutor<F>>(
    opcode: usize,
    executor: &mut E,
    tester: &mut VmChipTestBuilder<F>,
    num_ops: usize,
) {
    let mut rng = create_seeded_rng();
    for _ in 0..num_ops {
        let b = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let c = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let instruction = rv32_write_heap_default(
            tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            opcode,
        );
        tester.execute(executor, instruction);
    }
}

fn run_alu_256_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let mut tester = VmChipTestBuilder::default();

    let mut chip = Rv32BaseAlu256Chip::<F>::new(
        Rv32HeapAdapterChip::<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        BaseAluCoreChip::new(bitwise_chip.clone(), 0),
        tester.memory_controller(),
    );

    run_int_256_rand_execute(opcode as usize, &mut chip, &mut tester, num_ops);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_256_add_rand_test() {
    run_alu_256_rand_test(BaseAluOpcode::ADD, 24);
}

#[test]
fn alu_256_sub_rand_test() {
    run_alu_256_rand_test(BaseAluOpcode::SUB, 24);
}

#[test]
fn alu_256_xor_rand_test() {
    run_alu_256_rand_test(BaseAluOpcode::XOR, 24);
}

#[test]
fn alu_256_or_rand_test() {
    run_alu_256_rand_test(BaseAluOpcode::OR, 24);
}

#[test]
fn alu_256_and_rand_test() {
    run_alu_256_rand_test(BaseAluOpcode::AND, 24);
}

fn run_lt_256_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32LessThan256Chip::<F>::new(
        Rv32HeapAdapterChip::<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        LessThanCoreChip::new(bitwise_chip.clone(), 0),
        tester.memory_controller(),
    );

    run_int_256_rand_execute(opcode as usize, &mut chip, &mut tester, num_ops);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn lt_256_slt_rand_test() {
    run_lt_256_rand_test(LessThanOpcode::SLT, 24);
}

#[test]
fn lt_256_sltu_rand_test() {
    run_lt_256_rand_test(LessThanOpcode::SLTU, 24);
}

fn run_mul_256_rand_test(num_ops: usize) {
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [
            1 << RV32_CELL_BITS,
            (INT256_NUM_LIMBS * (1 << RV32_CELL_BITS)) as u32,
        ],
    );
    let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32Multiplication256Chip::<F>::new(
        Rv32HeapAdapterChip::<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        MultiplicationCoreChip::new(range_tuple_checker.clone(), 0),
        tester.memory_controller(),
    );

    run_int_256_rand_execute(MulOpcode::MUL as usize, &mut chip, &mut tester, num_ops);
    let tester = tester
        .build()
        .load(chip)
        .load(range_tuple_checker)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn mul_256_rand_test() {
    run_mul_256_rand_test(24);
}

fn run_shift_256_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32Shift256Chip::<F>::new(
        Rv32HeapAdapterChip::<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        ShiftCoreChip::new(
            bitwise_chip.clone(),
            tester.memory_controller().borrow().range_checker.clone(),
            0,
        ),
        tester.memory_controller(),
    );

    run_int_256_rand_execute(opcode as usize, &mut chip, &mut tester, num_ops);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_256_sll_rand_test() {
    run_shift_256_rand_test(ShiftOpcode::SLL, 24);
}

#[test]
fn shift_256_srl_rand_test() {
    run_shift_256_rand_test(ShiftOpcode::SRL, 24);
}

#[test]
fn shift_256_sra_rand_test() {
    run_shift_256_rand_test(ShiftOpcode::SRA, 24);
}
