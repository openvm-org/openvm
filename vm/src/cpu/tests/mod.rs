use afs_chips::is_zero::IsZeroAir;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::columns::{CpuCols, CpuIoCols};
use crate::cpu::{CpuAir, CpuOptions};
use crate::memory::OpType;

use super::columns::MemoryAccessCols;
use super::trace::ProgramExecution;
use super::{
    trace::{ArithmeticOperation, Instruction, MemoryAccess},
    OpCode::*,
};
use super::{ARITHMETIC_BUS, MEMORY_BUS, READ_INSTRUCTION_BUS};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = CpuCols::<usize>::get_width(CpuOptions {
        field_arithmetic_enabled: true,
    });
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = CpuCols::<usize>::from_slice(
        &all_cols,
        CpuOptions {
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
    field_arithmetic_enabled: bool,
    program: Vec<Instruction<F>>,
    mut expected_execution: Vec<usize>,
    expected_memory_log: Vec<MemoryAccess<F>>,
    expected_arithmetic_operations: Vec<ArithmeticOperation<F>>,
) {
    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled,
    });
    let execution = air.generate_program_execution(program.clone());

    assert_eq!(execution.program, program);
    assert_eq!(execution.memory_accesses, expected_memory_log);
    assert_eq!(execution.arithmetic_ops, expected_arithmetic_operations);

    while !expected_execution.len().is_power_of_two() {
        expected_execution.push(*expected_execution.last().unwrap());
    }

    assert_eq!(execution.trace_rows.len(), expected_execution.len());
    for (i, row) in execution.trace_rows.iter().enumerate() {
        let pc = expected_execution[i];
        let expected_io = CpuIoCols {
            clock_cycle: F::from_canonical_u64(i as u64),
            pc: F::from_canonical_u64(pc as u64),
            opcode: F::from_canonical_u64(program[pc].opcode as u64),
            op_a: program[pc].op_a,
            op_b: program[pc].op_b,
            op_c: program[pc].op_c,
            d: program[pc].d,
            e: program[pc].e,
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

fn air_test(field_arithmetic_enabled: bool, program: Vec<Instruction<BabyBear>>) {
    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled,
    });
    let execution = air.generate_program_execution(program);
    air_test_custom_execution(field_arithmetic_enabled, execution);
}

fn air_test_change_pc(
    is_field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
    change_row: usize,
    change_value: usize,
    should_fail: bool,
) {
    let chip = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: is_field_arithmetic_enabled,
    });
    let mut execution = chip.generate_program_execution(program);

    let old_value = execution.trace_rows[change_row].io.pc.as_canonical_u64() as usize;
    execution.trace_rows[change_row].io.pc = BabyBear::from_canonical_usize(change_value);

    execution.execution_frequencies[old_value] -= BabyBear::one();
    execution.execution_frequencies[change_value] += BabyBear::one();

    air_test_custom_execution_with_failure(is_field_arithmetic_enabled, execution, should_fail);
}

fn air_test_custom_execution(
    is_field_arithmetic_enabled: bool,
    execution: ProgramExecution<BabyBear>,
) {
    air_test_custom_execution_with_failure(is_field_arithmetic_enabled, execution, false);
}

fn air_test_custom_execution_with_failure(
    is_field_arithmetic_enabled: bool,
    execution: ProgramExecution<BabyBear>,
    should_fail: bool,
) {
    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: is_field_arithmetic_enabled,
    });
    let trace = execution.trace();

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
            instruction.d,
            instruction.e,
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
            BabyBear::from_canonical_usize(memory_access.timestamp),
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

    let test_result = if is_field_arithmetic_enabled {
        run_simple_test_no_pis(
            vec![&air, &program_air, &memory_air, &arithmetic_air],
            vec![trace, program_trace, memory_trace, arithmetic_trace],
        )
    } else {
        run_simple_test_no_pis(
            vec![&air, &program_air, &memory_air],
            vec![trace, program_trace, memory_trace],
        )
    };

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
    Instruction 4 terminates
    The remainder is a loop that decrements word[0]_1 until it reaches 0, then terminates.
    Instruction 1 checks if word[0]_1 is 0 yet, and if so sets pc to 5 in order to terminate
    Instruction 2 decrements word[0]_1 (using word[1]_1)
    Instruction 3 uses JAL as a simple jump to go back to instruction 1 (repeating the loop).
     */
    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction::from_isize(STOREW, n, 0, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::from_isize(FSUB, 0, 0, 1, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let mut expected_execution: Vec<usize> = vec![0, 1];
    for _ in 0..n {
        expected_execution.push(2);
        expected_execution.push(3);
        expected_execution.push(1);
    }
    expected_execution.push(4);

    let mut expected_memory_log = vec![
        MemoryAccess::from_isize(2, OpType::Write, 1, 0, n),
        MemoryAccess::from_isize(3, OpType::Read, 1, 0, n),
    ];
    for t in 0..n {
        let clock = 2 + (3 * t);
        expected_memory_log.extend(vec![
            MemoryAccess::from_isize(3 * clock, OpType::Read, 1, 0, n - t),
            MemoryAccess::from_isize((3 * clock) + 2, OpType::Write, 1, 0, n - t - 1),
            MemoryAccess::from_isize((3 * (clock + 1)) + 2, OpType::Write, 1, 2, 4),
            MemoryAccess::from_isize(3 * (clock + 2), OpType::Read, 1, 0, n - t - 1),
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
        MemoryAccess::from_isize(2, OpType::Write, 1, 0, 5),
        MemoryAccess::from_isize(3, OpType::Read, 1, 0, 5),
        MemoryAccess::from_isize(6, OpType::Read, 1, 0, 5),
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
fn test_cpu_negative_wrong_pc() {
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
fn test_cpu_negative_wrong_pc_check() {
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

#[test]
#[should_panic(
    expected = "assertion `left == right` failed: constraints had nonzero value on row 0"
)]
fn test_cpu_negative_hasnt_terminated() {
    let program = vec![
        // word[0]_1 <- word[6]_0
        Instruction::from_isize(STOREW, 6, 0, 0, 0, 1),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];
    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: true,
    });
    let mut execution = air.generate_program_execution(program);
    execution.trace_rows.remove(execution.trace_rows.len() - 1);
    execution.execution_frequencies[1] = AbstractField::zero();

    air_test_custom_execution(true, execution);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_cpu_negative_secret_write() {
    let program = vec![
        // if word[0]_0 == word[0]_[0] then pc += 1
        Instruction::from_isize(BEQ, 0, 0, 1, 0, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: true,
    });
    let mut execution = air.generate_program_execution(program);

    let is_zero_air = IsZeroAir;
    let mut is_zero_trace = is_zero_air
        .generate_trace(vec![AbstractField::one()])
        .clone();
    let is_zero_aux = is_zero_trace.row_mut(0)[2];

    execution.trace_rows[0].aux.write = MemoryAccessCols {
        enabled: AbstractField::one(),
        address_space: AbstractField::one(),
        is_immediate: AbstractField::zero(),
        is_zero_aux,
        address: AbstractField::zero(),
        data: AbstractField::from_canonical_usize(115),
    };

    execution
        .memory_accesses
        .push(MemoryAccess::from_isize(0, OpType::Write, 1, 0, 115));

    air_test_custom_execution(true, execution);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_cpu_negative_disable_write() {
    let program = vec![
        // if word[0]_0 == word[0]_[0] then pc += 1
        Instruction::from_isize(STOREW, 113, 0, 0, 0, 1),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: true,
    });
    let mut execution = air.generate_program_execution(program);

    execution.trace_rows[0].aux.write.enabled = AbstractField::zero();

    execution.memory_accesses.remove(0);

    air_test_custom_execution(true, execution);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_cpu_negative_disable_read() {
    let program = vec![
        // if word[0]_0 == word[0]_[0] then pc += 1
        Instruction::from_isize(LOADW, 0, 0, 0, 1, 1),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let air = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: true,
    });
    let mut execution = air.generate_program_execution(program);

    execution.trace_rows[0].aux.read1.enabled = AbstractField::zero();

    execution.memory_accesses.remove(0);

    air_test_custom_execution(true, execution);
}
