use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::{CPUChip, /*ARITHMETIC_BUS, MEMORY_BUS,*/ READ_INSTRUCTION_BUS};

use crate::cpu::{OpCode::*, trace::Instruction};
use crate::program::columns::ProgramPreprocessedCols;

use super::ProgramAir;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = ProgramPreprocessedCols::<usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ProgramPreprocessedCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn interaction_test(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
) {
    let cpu_chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = cpu_chip.generate_trace(program.clone());

    let air = ProgramAir { program };
    let trace = air.generate_trace(&execution);

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in execution.program.iter().enumerate() {
        program_rows.extend(vec![
            execution.execution_frequencies[pc],
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.as_b,
            instruction.as_c,
        ]);
    }
    let counter_trace = RowMajorMatrix::new(program_rows, 8);

    run_simple_test_no_pis(
        vec![&air, &counter_air],
        vec![trace, counter_trace],
    )
    .expect("Verification failed");
}

// integration test
/*fn air_test(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
) {
    let cpu_chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = cpu_chip.generate_trace(program.clone());

    let air = ProgramAir { program };
    let trace = air.generate_trace(&execution);

    let memory_air = DummyInteractionAir::new(5, false, MEMORY_BUS);
    let mut memory_rows = vec![];
    for memory_access in execution.memory_accesses.iter() {
        memory_rows.extend(vec![
            BabyBear::one(),
            BabyBear::from_canonical_usize(memory_access.clock),
            BabyBear::from_bool(memory_access.is_write),
            memory_access.address_space,
            memory_access.address,
            memory_access.value,
        ]);
    }
    while !(memory_rows.len() / 6).is_power_of_two() {
        memory_rows.push(BabyBear::zero());
    }
    let memory_trace = RowMajorMatrix::new(memory_rows, 6);

    let arithmetic_air = DummyInteractionAir::new(4, false, ARITHMETIC_BUS);
    let mut arithmetic_rows = vec![];
    for arithmetic_op in execution.arithmetic_ops.iter() {
        arithmetic_rows.extend(vec![
            BabyBear::one(),
            BabyBear::from_canonical_usize(arithmetic_op.opcode as usize),
            arithmetic_op.operand1,
            arithmetic_op.operand2,
            arithmetic_op.result,
        ]);
    }
    while !(arithmetic_rows.len() / 5).is_power_of_two() {
        arithmetic_rows.push(BabyBear::zero());
    }
    let arithmetic_trace = RowMajorMatrix::new(arithmetic_rows, 5);

    run_simple_test_no_pis(
        vec![&cpu_chip.air, &air, &memory_air, &arithmetic_air],
        vec![execution.trace(), trace, memory_trace, arithmetic_trace],
    )
    .expect("Verification failed");
}*/

#[test]
fn test_cpu() {
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();

    let neg = BabyBear::neg_one();
    let neg_two = neg * two;
    let neg_four = neg * BabyBear::from_canonical_u32(4);

    let n = 20;
    let nf = AbstractField::from_canonical_u64(n);

    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction { opcode: STOREW, op_a: nf, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[0]_1 <- word[n]_0
        Instruction { opcode: STOREW, op_a: nf, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[1]_1 <- word[1]_1
        Instruction { opcode: STOREW, op_a: one, op_b: one, op_c: zero, as_b: zero, as_c: one },
        // if word[0]_1 == 0 then pc -= 4
        Instruction { opcode: BEQ, op_a: zero, op_b: zero, op_c: neg_four, as_b: one, as_c: zero },
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction { opcode: FSUB, op_a: zero, op_b: zero, op_c: one, as_b: one, as_c: one },
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction { opcode: JAL, op_a: two, op_b: neg_two, op_c: zero, as_b: one, as_c: zero },
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction { opcode: JAL, op_a: two, op_b: neg_two, op_c: zero, as_b: one, as_c: zero },
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction { opcode: JAL, op_a: two, op_b: neg_two, op_c: zero, as_b: one, as_c: zero },
    ];

    interaction_test(true, program.clone());
    //air_test(true, program.clone());
}

#[test]
fn test_cpu_without_field_arithmetic() {
    //std::env::set_var("RUST_BACKTRACE", "full");
    let field_arithmetic_enabled = false;

    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();
    let four = BabyBear::from_canonical_u32(4);
    let five = BabyBear::from_canonical_u32(5);

    let neg = BabyBear::neg_one();
    let neg_two = neg * two;
    let neg_five = neg * BabyBear::from_canonical_u32(5);

    let program = vec![
        // word[0]_1 <- word[5]_0
        Instruction { opcode: STOREW, op_a: five, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[0]_1 <- word[5]_0
        Instruction { opcode: STOREW, op_a: five, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // if word[0]_1 != 4 then pc += 2
        Instruction { opcode: BNE, op_a: zero, op_b: four, op_c: two, as_b: one, as_c: zero },
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction { opcode: JAL, op_a: two, op_b: neg_two, op_c: zero, as_b: one, as_c: zero },
        // if word[0]_1 == 5 then pc -= 5
        Instruction { opcode: BEQ, op_a: zero, op_b: five, op_c: neg_five, as_b: one, as_c: zero },
        // if word[0]_1 == 5 then pc -= 5
        Instruction { opcode: BEQ, op_a: zero, op_b: five, op_c: neg_five, as_b: one, as_c: zero },
        // if word[0]_1 == 5 then pc -= 5
        Instruction { opcode: BEQ, op_a: zero, op_b: five, op_c: neg_five, as_b: one, as_c: zero },
        // if word[0]_1 == 5 then pc -= 5
        Instruction { opcode: BEQ, op_a: zero, op_b: five, op_c: neg_five, as_b: one, as_c: zero },
    ];

    interaction_test(field_arithmetic_enabled, program.clone());
    //air_test(field_arithmetic_enabled, program.clone());
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_program_negative() {
    let program = vec![
        Instruction { opcode: STOREW, op_a: BabyBear::neg_one(), op_b: BabyBear::zero(), op_c: BabyBear::zero(), as_b: BabyBear::zero(), as_c: BabyBear::one() },
        Instruction { opcode: LOADW, op_a: BabyBear::neg_one(), op_b: BabyBear::zero(), op_c: BabyBear::zero(), as_b: BabyBear::one(), as_c: BabyBear::one() },
    ];

    let cpu_chip = CPUChip::new(true);
    let execution = cpu_chip.generate_trace(program.clone());

    let air = ProgramAir { program };
    let trace = air.generate_trace(&execution);

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in execution.program.iter().enumerate() {
        program_rows.extend(vec![
            execution.execution_frequencies[pc],
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.as_b,
            instruction.as_c,
        ]);
    }
    let mut counter_trace = RowMajorMatrix::new(program_rows, 8);
    counter_trace.row_mut(1)[1] = BabyBear::zero();

    run_simple_test_no_pis(
        vec![&air, &counter_air],
        vec![trace, counter_trace],
    )
    .expect("Incorrect failure mode");
}