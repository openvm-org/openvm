use std::sync::Arc;


use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;

use crate::cpu::columns::{CPUCols, CPUIOCols};
use crate::cpu::{CPUChip, CPUOptions};

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

#[test]
fn test_program_execution() {
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    let two = BabyBear::two();
    let five = BabyBear::from_canonical_u32(5);

    let neg = BabyBear::neg_one();
    let neg_two = neg * two;
    let neg_three = neg * BabyBear::from_canonical_u32(3);

    let n = 1600;
    let nf = AbstractField::from_canonical_u64(n);

    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction { opcode: STOREW, op_a: nf, op_b: zero, op_c: zero, as_b: zero, as_c: one },
        // word[1]_1 <- word[1]_1
        Instruction { opcode: STOREW, op_a: one, op_b: one, op_c: zero, as_b: zero, as_c: one },
        // if word[0]_1 == 0 then pc -= 3
        Instruction { opcode: BEQ, op_a: zero, op_b: zero, op_c: neg_three, as_b: one, as_c: zero },
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction { opcode: FSUB, op_a: zero, op_b: zero, op_c: one, as_b: one, as_c: one },
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction { opcode: JAL, op_a: two, op_b: neg_two, op_c: zero, as_b: one, as_c: zero },
    ];

    let mut expected_execution: Vec<usize> = vec![0, 1, 2];
    for _ in 0..n {
        expected_execution.push(3);
        expected_execution.push(4);
        expected_execution.push(2);
    }

    let mut expected_memory_log = vec![
        MemoryAccess { clock: 0, is_write: true, address_space: one, address: zero, value: nf },
        MemoryAccess { clock: 1, is_write: true, address_space: one, address: one, value: one },
        MemoryAccess { clock: 2, is_write: false, address_space: one, address: zero, value: nf },
    ];
    for t in 0..n {
        let tf = BabyBear::from_canonical_u64(t);
        let clock = (3 + (3 * t)) as usize;
        expected_memory_log.extend(vec![
            MemoryAccess { clock, is_write: false, address_space: one, address: zero, value: nf - tf },
            MemoryAccess { clock, is_write: false, address_space: one, address: one, value: one },
            MemoryAccess { clock, is_write: true, address_space: one, address: zero, value: nf - tf - one },
            MemoryAccess { clock: clock + 1, is_write: true, address_space: one, address: two, value: five },
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

    let chip = CPUChip::new(true);
    let execution = chip.generate_trace(program.clone());

    assert_eq!(execution.program, program);
    assert_eq!(execution.memory_accesses, expected_memory_log);
    assert_eq!(execution.arithmetic_ops, expected_arithmetic_operations);

    assert_eq!(execution.trace_rows.len(), expected_execution.len());
    for (i, row) in execution.trace_rows.iter().enumerate() {
        let pc = expected_execution[i];
        let expected_io = CPUIOCols {
            clock_cycle: BabyBear::from_canonical_u64(i as u64),
            pc: BabyBear::from_canonical_u64(pc as u64),
            opcode: BabyBear::from_canonical_u64(program[pc].opcode as u64),
            op_a: program[pc].op_a,
            op_b: program[pc].op_b,
            op_c: program[pc].op_c,
            as_b: program[pc].as_b,
            as_c: program[pc].as_c,
        };
        assert_eq!(row.io, expected_io);
    }
}

/*#[test]
fn test_is_less_than_chip_lt() {
    let bus_index: usize = 0;
    let limb_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanChip::new(bus_index, range_max, limb_bits, decomp, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_negative() {
    let bus_index: usize = 0;
    let limb_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanChip::new(bus_index, range_max, limb_bits, decomp, range_checker);
    let mut trace = chip.generate_trace(vec![(446, 553)]);
    let range_trace = chip.range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &chip.range_checker.air],
            vec![trace, range_trace],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
*/