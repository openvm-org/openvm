use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::columns::{CPUCols, CPUIOCols};
use crate::cpu::{CPUChip, CPUOptions};

use super::{ARITHMETIC_BUS, MEMORY_BUS, READ_INSTRUCTION_BUS};
use super::{OpCode::*, trace::{ArithmeticOperation, Instruction, MemoryAccess}};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = CPUCols::<usize>::get_width(CPUOptions { field_arithmetic_enabled: true });
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = CPUCols::<usize>::from_slice(&all_cols, CPUOptions { field_arithmetic_enabled: true });
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn program_execution_test<F: PrimeField64>(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<F>>,
    expected_execution: Vec<usize>,
    expected_memory_log: Vec<MemoryAccess<F>>,
    expected_arithmetic_operations: Vec<ArithmeticOperation<F>>,
) {
    let chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = chip.generate_trace(program.clone());

    assert_eq!(execution.program, program);
    assert_eq!(execution.memory_accesses, expected_memory_log);
    assert_eq!(execution.arithmetic_ops, expected_arithmetic_operations);

    assert_eq!(execution.trace_rows.len(), expected_execution.len());
    for (i, row) in execution.trace_rows.iter().enumerate() {
        let pc = expected_execution[i];
        let expected_io = CPUIOCols {
            clock_cycle: F::from_canonical_u64(i as u64),
            pc: F::from_canonical_u64(pc as u64),
            opcode: F::from_canonical_u64(program[pc].opcode as u64),
            op_a: program[pc].op_a,
            op_b: program[pc].op_b,
            op_c: program[pc].op_c,
            as_b: program[pc].as_b,
            as_c: program[pc].as_c,
        };
        assert_eq!(row.io, expected_io);
    }
}

fn air_test(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
) {
    let chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = chip.generate_trace(program);

    let program_air = DummyInteractionAir::new(7, false, READ_INSTRUCTION_BUS);
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
    while !(program_rows.len() / 8).is_power_of_two() {
        program_rows.push(BabyBear::zero());
    }
    let program_trace = RowMajorMatrix::new(program_rows, 8);

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
        vec![&chip.air, &program_air, &memory_air, &arithmetic_air],
        vec![execution.trace(), program_trace, memory_trace, arithmetic_trace],
    )
    .expect("Verification failed");
}

fn air_test_change_pc(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
    change_row: usize,
    change_value: usize,
    should_fail: bool,
) {
    let chip = CPUChip::new(is_field_arithmetic_enabled);
    let mut execution = chip.generate_trace(program);

    let mut trace = execution.trace();
    let options = CPUOptions { field_arithmetic_enabled: is_field_arithmetic_enabled };
    let all_cols = (0..CPUCols::<BabyBear>::get_width(options)).collect::<Vec<usize>>();
    let cols_numbered = CPUCols::<usize>::from_slice(&all_cols, options);
    let pc_col = cols_numbered.io.pc;
    let old_value = trace.row_mut(change_row)[pc_col].as_canonical_u64() as usize;
    trace.row_mut(change_row)[pc_col] = BabyBear::from_canonical_usize(change_value);

    execution.execution_frequencies[old_value] -= BabyBear::one();
    execution.execution_frequencies[change_value] += BabyBear::one();

    let program_air = DummyInteractionAir::new(7, false, READ_INSTRUCTION_BUS);
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
    while !(program_rows.len() / 8).is_power_of_two() {
        program_rows.push(BabyBear::zero());
    }
    let program_trace = RowMajorMatrix::new(program_rows, 8);

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

    println!("here");

    let test_result = run_simple_test_no_pis(
        vec![&chip.air, &program_air, &memory_air, &arithmetic_air],
        vec![trace, program_trace, memory_trace, arithmetic_trace],
    );

    println!("bing: {:?}", test_result);

    if should_fail {
        assert_eq!(
            test_result,
            Err(VerificationError::OodEvaluationMismatch),
            "Expected verification to fail, but it passed"
        );
    } else {
        test_result.expect("Verification failed");
    }
}

#[test]
fn test_cpu() {
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();
    let six = BabyBear::from_canonical_u32(6);

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
    ];

    let mut expected_execution: Vec<usize> = vec![0, 1, 2, 3];
    for _ in 0..n {
        expected_execution.push(4);
        expected_execution.push(5);
        expected_execution.push(3);
    }

    let mut expected_memory_log = vec![
        MemoryAccess { clock: 0, is_write: true, address_space: one, address: zero, value: nf },
        MemoryAccess { clock: 1, is_write: true, address_space: one, address: zero, value: nf },
        MemoryAccess { clock: 2, is_write: true, address_space: one, address: one, value: one },
        MemoryAccess { clock: 3, is_write: false, address_space: one, address: zero, value: nf },
    ];
    for t in 0..n {
        let tf = BabyBear::from_canonical_u64(t);
        let clock = (4 + (3 * t)) as usize;
        expected_memory_log.extend(vec![
            MemoryAccess { clock, is_write: false, address_space: one, address: zero, value: nf - tf },
            MemoryAccess { clock, is_write: false, address_space: one, address: one, value: one },
            MemoryAccess { clock, is_write: true, address_space: one, address: zero, value: nf - tf - one },
            MemoryAccess { clock: clock + 1, is_write: true, address_space: one, address: two, value: six },
            MemoryAccess { clock: clock + 2, is_write: false, address_space: one, address: zero, value: nf - tf - one },
        ]);
    }

    let mut expected_arithmetic_operations = vec![];
    for t in 0..n {
        let tf = BabyBear::from_canonical_u64(t);
        expected_arithmetic_operations.push(
            ArithmeticOperation { opcode: FSUB, operand1: nf - tf, operand2: one, result: nf - tf - one },
        );
    }

    program_execution_test::<BabyBear>(true, program.clone(), expected_execution, expected_memory_log, expected_arithmetic_operations);
    air_test(true, program);
}

#[test]
fn test_cpu_without_field_arithmetic() {
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
    ];

    let expected_execution: Vec<usize> = vec![0, 1, 2, 4];

    let expected_memory_log = vec![
        MemoryAccess { clock: 0, is_write: true, address_space: one, address: zero, value: five },
        MemoryAccess { clock: 1, is_write: true, address_space: one, address: zero, value: five },
        MemoryAccess { clock: 2, is_write: false, address_space: one, address: zero, value: five },
        MemoryAccess { clock: 3, is_write: false, address_space: one, address: zero, value: five },
    ];

    program_execution_test::<BabyBear>(field_arithmetic_enabled, program.clone(), expected_execution, expected_memory_log, vec![]);
    air_test(field_arithmetic_enabled, program);
}

#[test]
#[should_panic]
fn test_cpu_negative() {
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();
    let four = BabyBear::from_canonical_u32(4);
    let six = BabyBear::from_canonical_u32(6);

    let neg = BabyBear::neg_one();
    let neg_four = neg * four;
    let neg_five = neg * BabyBear::from_canonical_u32(5);

    let program = vec![
        // word[0]_1 <- word[6]_0
        Instruction { opcode: STOREW, op_a: six, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[0]_1 <- word[6]_0
        Instruction { opcode: STOREW, op_a: six, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // if word[0]_1 == 4 then pc += 2
        Instruction { opcode: BEQ, op_a: zero, op_b: four, op_c: two, as_b: one, as_c: zero },
        // if word[0]_1 != 0 then pc -= 4
        Instruction { opcode: BNE, op_a: zero, op_b: zero, op_c: neg_four, as_b: one, as_c: zero },
        // if word[0]_1 != 0 then pc -= 5
        Instruction { opcode: BNE, op_a: zero, op_b: zero, op_c: neg_five, as_b: one, as_c: zero },
    ];

    air_test_change_pc(true, program, 3, 4, true);
}

#[test]
fn test_cpu_negative_assure() {
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();
    let four = BabyBear::from_canonical_u32(4);
    let six = BabyBear::from_canonical_u32(6);

    let neg = BabyBear::neg_one();
    let neg_four = neg * four;
    let neg_five = neg * BabyBear::from_canonical_u32(5);

    let program = vec![
        // word[0]_1 <- word[6]_0
        Instruction { opcode: STOREW, op_a: six, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[0]_1 <- word[6]_0
        Instruction { opcode: STOREW, op_a: six, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // if word[0]_1 == 4 then pc += 2
        Instruction { opcode: BEQ, op_a: zero, op_b: four, op_c: two, as_b: one, as_c: zero },
        // if word[0]_1 != 0 then pc -= 4
        Instruction { opcode: BNE, op_a: zero, op_b: zero, op_c: neg_four, as_b: one, as_c: zero },
        // if word[0]_1 != 0 then pc -= 5
        Instruction { opcode: BNE, op_a: zero, op_b: zero, op_c: neg_five, as_b: one, as_c: zero },
    ];

    air_test_change_pc(true, program, 3, 3, false);
}