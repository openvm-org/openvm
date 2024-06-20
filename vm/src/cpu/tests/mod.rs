use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::columns::{CPUCols, CPUIOCols};
use crate::cpu::{CPUChip, CPUOptions};
use crate::memory::OpType;

use super::{
    trace::{ArithmeticOperation, Instruction, MemoryAccess},
    OpCode::*,
};
use super::{ARITHMETIC_BUS, MEMORY_BUS, READ_INSTRUCTION_BUS};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = CPUCols::<usize>::get_width(CPUOptions {
        field_arithmetic_enabled: true,
    });
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = CPUCols::<usize>::from_slice(
        &all_cols,
        CPUOptions {
            field_arithmetic_enabled: true,
        },
    );
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn program_execution_test<F: PrimeField64>(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<F>>,
    mut expected_execution: Vec<usize>,
    expected_memory_log: Vec<MemoryAccess<F>>,
    expected_arithmetic_operations: Vec<ArithmeticOperation<F>>,
) {
    let chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = chip.generate_trace(program.clone());

    assert_eq!(execution.program, program);
    assert_eq!(execution.memory_accesses, expected_memory_log);
    assert_eq!(execution.arithmetic_ops, expected_arithmetic_operations);

    while !expected_execution.len().is_power_of_two() {
        expected_execution.push(*expected_execution.last().unwrap());
    }

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

    let mut execution_frequency_check = execution.execution_frequencies.clone();
    for row in execution.trace_rows {
        let pc = row.io.pc.as_canonical_u64() as usize;
        execution_frequency_check[pc] += F::neg_one();
    }
    for frequency in execution_frequency_check.iter() {
        assert_eq!(*frequency, F::zero());
    }
}

fn air_test(is_field_arithmetic_enabled: bool, program: Vec<Instruction<BabyBear>>) {
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
            BabyBear::from_bool(memory_access.op_type == OpType::Write),
            memory_access.address_space,
            memory_access.address,
            memory_access.data,
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
        vec![
            execution.trace(),
            program_trace,
            memory_trace,
            arithmetic_trace,
        ],
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
    let options = CPUOptions {
        field_arithmetic_enabled: is_field_arithmetic_enabled,
    };
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
            BabyBear::from_bool(memory_access.op_type == OpType::Write),
            memory_access.address_space,
            memory_access.address,
            memory_access.data,
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
fn test_cpu_1() {
    let n = 2;

    /*
    Instruction 0 assigns word[0]_1 to n.
    Instruction 1 assigns word[1]_1 to 1 for use in later arithmetic operations.
    Instruction 5 terminates
    The remainder is a loop that decrements word[0]_1 until it reaches 0, then terminates.
    Instruction 2 checks if word[0]_1 is 0 yet, and if so sets pc to 5 in order to terminate
    Instruction 3 decrements word[0]_1 (using word[1]_1)
    Instruction 4 uses JAL as a simple jump to go back to instruction 3 (repeating the loop).
     */
    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction::from_isize(STOREW, n, 0, 0, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::from_isize(STOREW, 1, 1, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(FSUB, 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let mut expected_execution: Vec<usize> = vec![0, 1, 2];
    for _ in 0..n {
        expected_execution.push(3);
        expected_execution.push(4);
        expected_execution.push(2);
    }
    expected_execution.push(5);

    let mut expected_memory_log = vec![
        MemoryAccess::from_isize(0, OpType::Write, 1, 0, n),
        MemoryAccess::from_isize(1, OpType::Write, 1, 1, 1),
        MemoryAccess::from_isize(2, OpType::Read, 1, 0, n),
    ];
    for t in 0..n {
        let clock = 3 + (3 * t);
        expected_memory_log.extend(vec![
            MemoryAccess::from_isize(clock, OpType::Read, 1, 0, n - t),
            MemoryAccess::from_isize(clock, OpType::Read, 1, 1, 1),
            MemoryAccess::from_isize(clock, OpType::Write, 1, 0, n - t - 1),
            MemoryAccess::from_isize(clock + 1, OpType::Write, 1, 2, 5),
            MemoryAccess::from_isize(clock + 2, OpType::Read, 1, 0, n - t - 1),
        ]);
    }

    let mut expected_arithmetic_operations = vec![];
    for t in 0..n {
        expected_arithmetic_operations.push(ArithmeticOperation::from_isize(
            FSUB,
            n - t,
            1,
            n - t - 1,
        ));
    }

    program_execution_test::<BabyBear>(
        true,
        program.clone(),
        expected_execution,
        expected_memory_log,
        expected_arithmetic_operations,
    );
    air_test(true, program);
}

#[test]
fn test_cpu_without_field_arithmetic() {
    let field_arithmetic_enabled = false;

    /*
    Instruction 0 assigns word[0]_1 to 5.
    Instruction 1 checks if word[0]_1 is *not* 4, and if so jumps to instruction 4.
    Instruction 2 is never run.
    Instruction 3 terminates.
    Instruction 4 checks if word[0]_1 is 5, and if so jumps to instruction 3 to terminate.
     */
    let program = vec![
        // word[0]_1 <- word[5]_0
        Instruction::from_isize(STOREW, 5, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE, 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ, 0, 5, -1, 1, 0),
    ];

    let expected_execution: Vec<usize> = vec![0, 1, 4, 3];

    let expected_memory_log = vec![
        MemoryAccess::from_isize(0, OpType::Write, 1, 0, 5),
        MemoryAccess::from_isize(1, OpType::Read, 1, 0, 5),
        MemoryAccess::from_isize(2, OpType::Read, 1, 0, 5),
    ];

    program_execution_test::<BabyBear>(
        field_arithmetic_enabled,
        program.clone(),
        expected_execution,
        expected_memory_log,
        vec![],
    );
    air_test(field_arithmetic_enabled, program);
}

#[test]
#[should_panic]
fn test_cpu_negative() {
    /*
    Instruction 0 assigns word[0]_1 to 6.
    Instruction 1 checks if word[0]_1 is 4, and if so jumps to instruction 3 (but this doesn't happen)
    Instruction 2 checks if word[0]_1 is 0, and if not jumps to instruction 4 to terminate
    Instruction 3 checks if word[0]_1 is 0, and if not jumps to instruction 4 to terminate (identical to instruction 2) (note: would go to instruction 4 either way)
    Instruction 4 terminates
     */
    let program = vec![
        // word[0]_1 <- word[6]_0
        Instruction::from_isize(STOREW, 6, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BEQ, 0, 4, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 2
        Instruction::from_isize(BNE, 0, 0, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 1
        Instruction::from_isize(BNE, 0, 0, 1, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    air_test_change_pc(true, program, 2, 3, true);
}

#[test]
fn test_cpu_negative_assure() {
    //Same program as test_cpu_negative.
    let program = vec![
        // word[0]_1 <- word[6]_0
        Instruction::from_isize(STOREW, 6, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BEQ, 0, 4, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 2
        Instruction::from_isize(BNE, 0, 0, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 1
        Instruction::from_isize(BNE, 0, 0, 1, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    air_test_change_pc(true, program, 2, 2, false);
}
