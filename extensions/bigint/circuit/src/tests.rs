use std::sync::Arc;

use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use ax_stark_backend::p3_field::{AbstractField, PrimeField32};
use ax_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use axvm_circuit::{
    arch::{
        testing::VmChipTestBuilder, InstructionExecutor, BITWISE_OP_LOOKUP_BUS,
        RANGE_TUPLE_CHECKER_BUS,
    },
    utils::generate_long_number,
};
use axvm_instructions::{
    program::PC_BITS, riscv::RV32_CELL_BITS, BaseAluOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode, UsizeOpcode,
};
use axvm_rv32_adapters::{
    rv32_heap_branch_default, rv32_write_heap_default, Rv32HeapAdapterChip,
    Rv32HeapBranchAdapterChip,
};
use axvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV_B_TYPE_IMM_BITS},
    BaseAluCoreChip, BranchEqualCoreChip, BranchLessThanCoreChip, LessThanCoreChip,
    MultiplicationCoreChip, ShiftCoreChip,
};
use rand::Rng;

use super::{
    Rv32BaseAlu256Chip, Rv32BranchEqual256Chip, Rv32BranchLessThan256Chip, Rv32LessThan256Chip,
    Rv32Multiplication256Chip, Rv32Shift256Chip,
};

type F = BabyBear;

#[allow(clippy::type_complexity)]
fn run_int_256_rand_execute<E: InstructionExecutor<F>>(
    opcode: usize,
    num_ops: usize,
    executor: &mut E,
    tester: &mut VmChipTestBuilder<F>,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_LIMBS], &[u32; INT256_NUM_LIMBS]) -> bool>,
) {
    const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);

    let mut rng = create_seeded_rng();
    let branch = branch_fn.is_some();

    for _ in 0..num_ops {
        let b = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let c = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        if branch {
            let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
            let instruction = rv32_heap_branch_default(
                tester,
                vec![b.map(F::from_canonical_u32)],
                vec![c.map(F::from_canonical_u32)],
                imm as isize,
                opcode,
            );

            tester.execute_with_pc(
                executor,
                instruction,
                rng.gen_range((ABS_MAX_BRANCH as u32)..(1 << (PC_BITS - 1))),
            );

            let cmp_result = branch_fn.unwrap()(opcode, &b, &c);
            let from_pc = tester.execution.last_from_pc().as_canonical_u32() as i32;
            let to_pc = tester.execution.last_to_pc().as_canonical_u32() as i32;
            assert_eq!(to_pc, from_pc + if cmp_result { imm } else { 4 });
        } else {
            let instruction = rv32_write_heap_default(
                tester,
                vec![b.map(F::from_canonical_u32)],
                vec![c.map(F::from_canonical_u32)],
                opcode,
            );
            tester.execute(executor, instruction);
        }
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

    run_int_256_rand_execute(opcode as usize, num_ops, &mut chip, &mut tester, None);
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

    run_int_256_rand_execute(opcode as usize, num_ops, &mut chip, &mut tester, None);
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

    run_int_256_rand_execute(
        MulOpcode::MUL as usize,
        num_ops,
        &mut chip,
        &mut tester,
        None,
    );
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

    run_int_256_rand_execute(opcode as usize, num_ops, &mut chip, &mut tester, None);
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

fn run_beq_256_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let mut chip = Rv32BranchEqual256Chip::<F>::new(
        Rv32HeapBranchAdapterChip::<F, 2, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        BranchEqualCoreChip::new(0, 4),
        tester.memory_controller(),
    );

    let branch_fn = |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| {
        x.iter()
            .zip(y.iter())
            .fold(true, |acc, (x, y)| acc && (x == y))
            ^ (opcode == BranchEqualOpcode::BNE as usize)
    };

    run_int_256_rand_execute(
        opcode as usize,
        num_ops,
        &mut chip,
        &mut tester,
        Some(branch_fn),
    );
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn beq_256_beq_rand_test() {
    run_beq_256_rand_test(BranchEqualOpcode::BEQ, 24);
}

#[test]
fn beq_256_bne_rand_test() {
    run_beq_256_rand_test(BranchEqualOpcode::BNE, 24);
}

fn run_blt_256_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32BranchLessThan256Chip::<F>::new(
        Rv32HeapBranchAdapterChip::<F, 2, INT256_NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        ),
        BranchLessThanCoreChip::new(bitwise_chip.clone(), 0),
        tester.memory_controller(),
    );

    let branch_fn = |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| {
        let opcode = BranchLessThanOpcode::from_usize(opcode);
        let (is_ge, is_signed) = match opcode {
            BranchLessThanOpcode::BLT => (false, true),
            BranchLessThanOpcode::BLTU => (false, false),
            BranchLessThanOpcode::BGE => (true, true),
            BranchLessThanOpcode::BGEU => (true, false),
        };
        let x_sign = x[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) != 0 && is_signed;
        let y_sign = y[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) != 0 && is_signed;
        for (x, y) in x.iter().rev().zip(y.iter().rev()) {
            if x != y {
                return (x < y) ^ x_sign ^ y_sign ^ is_ge;
            }
        }
        is_ge
    };

    run_int_256_rand_execute(
        opcode as usize,
        num_ops,
        &mut chip,
        &mut tester,
        Some(branch_fn),
    );
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn blt_256_blt_rand_test() {
    run_blt_256_rand_test(BranchLessThanOpcode::BLT, 24);
}

#[test]
fn blt_256_bltu_rand_test() {
    run_blt_256_rand_test(BranchLessThanOpcode::BLTU, 24);
}

#[test]
fn blt_256_bge_rand_test() {
    run_blt_256_rand_test(BranchLessThanOpcode::BGE, 24);
}

#[test]
fn blt_256_bgeu_rand_test() {
    run_blt_256_rand_test(BranchLessThanOpcode::BGEU, 24);
}
