use std::iter;

use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};

use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test,
    interaction::dummy_interaction_air::DummyInteractionAir,
};

use crate::{
    cpu::{
        ARITHMETIC_BUS,
        columns::{CpuCols, CpuIoCols},
        CpuChip, CpuOptions, READ_INSTRUCTION_BUS, trace::Instruction,
    },
    field_arithmetic::FieldArithmeticOperation,
    memory::manager::interface::MemoryInterface,
    program::Program,
    vm::{
        config::{DEFAULT_MAX_SEGMENT_LEN, MemoryConfig, VmConfig},
        ExecutionSegment, VirtualMachine,
    },
};

use super::{CpuAir, Opcode::*};

const TEST_NUM_WORDS: usize = 1;
const TEST_WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 16;
const DECOMP: usize = 8;

fn make_vm<const NUM_WORDS: usize, const WORD_SIZE: usize>(
    program: Program<BabyBear>,
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
) -> VirtualMachine<NUM_WORDS, WORD_SIZE, BabyBear> {
    VirtualMachine::<NUM_WORDS, WORD_SIZE, BabyBear>::new(
        VmConfig {
            field_arithmetic_enabled,
            field_extension_enabled,
            compress_poseidon2_enabled: false,
            perm_poseidon2_enabled: false,
            memory_config: MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
            num_public_values: 4,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            ..Default::default()
        },
        program,
        vec![],
    )
}

#[test]
fn test_flatten_fromslice_roundtrip() {
    let options = CpuOptions {
        field_arithmetic_enabled: true,
        field_extension_enabled: false,
        compress_poseidon2_enabled: false,
        perm_poseidon2_enabled: false,
        num_public_values: 4,
        is_less_than_enabled: false,
        modular_arithmetic_enabled: false,
    };
    let clk_max_bits = 30;
    let decomp = 8;
    let cpu_air = CpuAir::new(options, clk_max_bits, decomp);
    let num_cols = CpuCols::<TEST_WORD_SIZE, usize>::get_width(&cpu_air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = CpuCols::<TEST_WORD_SIZE, usize>::from_slice(&all_cols, &cpu_air);
    let flattened = cols_numbered.flatten(options);

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }
}

/*fn test<const WORD_SIZE: usize>(
    field_arithmetic_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
    mut expected_execution: Vec<usize>,
    expected_memory_log: Vec<MemoryAccess<WORD_SIZE, BabyBear>>,
    expected_arithmetic_operations: Vec<ArithmeticOperation<BabyBear>>,
) {
    program_execution_test(
        field_arithmetic_enabled,
        program,
        expected_execution,
        expected_memory_log,
        expected_arithmetic_operations,
    );
    let mut expected_execution_frequencies = expected_execution.clone();
    for i in 0..expected_execution.len() {
        expected_execution_frequencies[i] += 1;
    }
    air_test(
        field_arithmetic_enabled,
        program,
        expected_execution_frequencies,
        expected_memory_log,
        expected_arithmetic_operations,
    );
}*/

fn execution_test<const NUM_WORDS: usize, const WORD_SIZE: usize>(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Program<BabyBear>,
    mut expected_execution: Vec<usize>,
    expected_arithmetic_operations: Vec<FieldArithmeticOperation<BabyBear>>,
) {
    let mut vm = make_vm::<NUM_WORDS, WORD_SIZE>(
        program.clone(),
        field_arithmetic_enabled,
        field_extension_enabled,
    );
    assert_eq!(vm.segments.len(), 1);
    let segment = &mut vm.segments[0];
    CpuChip::execute(segment).unwrap();
    let mut trace = CpuChip::generate_trace(segment);

    assert_eq!(
        segment.field_arithmetic_chip.operations,
        expected_arithmetic_operations
    );

    while !expected_execution.len().is_power_of_two() {
        expected_execution.push(*expected_execution.last().unwrap());
    }

    println!("expected execution");
    assert_eq!(trace.height(), expected_execution.len());
    for (i, &pc) in expected_execution.iter().enumerate() {
        let cols =
            CpuCols::<WORD_SIZE, BabyBear>::from_slice(trace.row_mut(i), &segment.cpu_chip.air);
        let expected_io = CpuIoCols {
            // don't check timestamp
            timestamp: cols.io.timestamp,
            pc: BabyBear::from_canonical_u64(pc as u64),
            opcode: BabyBear::from_canonical_u64(program.instructions[pc].opcode as u64),
            op_a: program.instructions[pc].op_a,
            op_b: program.instructions[pc].op_b,
            op_c: program.instructions[pc].op_c,
            d: program.instructions[pc].d,
            e: program.instructions[pc].e,
            op_f: program.instructions[pc].op_f,
            op_g: program.instructions[pc].op_g,
        };
        assert_eq!(cols.io, expected_io);
    }

    let mut execution_frequency_check = segment.program_chip.execution_frequencies.clone();
    for pc in expected_execution {
        execution_frequency_check[pc] -= 1;
    }
    for frequency in execution_frequency_check.iter() {
        assert_eq!(*frequency, 0);
    }
}

fn air_test<const NUM_WORDS: usize, const WORD_SIZE: usize>(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Program<BabyBear>,
) {
    air_test_change::<NUM_WORDS, WORD_SIZE, _>(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
        false,
        |_, _| {},
    );
}

fn air_test_change_pc<const NUM_WORDS: usize, const WORD_SIZE: usize>(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Program<BabyBear>,
    should_fail: bool,
    change_row: usize,
    new: usize,
) {
    air_test_change::<NUM_WORDS, WORD_SIZE, _>(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
        should_fail,
        |rows, segment| {
            let old = rows[change_row].io.pc.as_canonical_u64() as usize;
            rows[change_row].io.pc = BabyBear::from_canonical_usize(new);
            segment.program_chip.execution_frequencies[new] += 1;
            segment.program_chip.execution_frequencies[old] -= 1;
        },
    );
}

fn air_test_change<
    const NUM_WORDS: usize,
    const WORD_SIZE: usize,
    F: Fn(
        &mut Vec<CpuCols<WORD_SIZE, BabyBear>>,
        &mut ExecutionSegment<NUM_WORDS, WORD_SIZE, BabyBear>,
    ),
>(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Program<BabyBear>,
    should_fail: bool,
    change: F,
) {
    let mut vm = make_vm::<NUM_WORDS, WORD_SIZE>(
        program.clone(),
        field_arithmetic_enabled,
        field_extension_enabled,
    );
    let options = vm.options();
    assert_eq!(vm.segments.len(), 1);
    let segment = &mut vm.segments[0];

    CpuChip::execute(segment).unwrap();
    let mut trace = CpuChip::generate_trace(segment);

    let mut rows = vec![];
    for i in 0..trace.height() {
        rows.push(CpuCols::<WORD_SIZE, BabyBear>::from_slice(
            trace.row_mut(i),
            &segment.cpu_chip.air,
        ));
    }
    change(&mut rows, segment);
    let mut flattened = vec![];
    for row in rows {
        flattened.extend(row.flatten(options));
    }
    let trace = DenseMatrix::new(flattened, trace.width());

    let program_air = DummyInteractionAir::new(9, false, READ_INSTRUCTION_BUS);
    let mut program_cells = vec![];
    for (pc, instruction) in program.instructions.iter().enumerate() {
        program_cells.extend(vec![
            BabyBear::from_canonical_usize(segment.program_chip.execution_frequencies[pc]),
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
            instruction.op_f,
            instruction.op_g,
        ]);
    }

    // Pad program cells with zeroes to make height a power of two.
    let width = 10;
    let desired_height = program.instructions.len().next_power_of_two();
    let cells_to_add = desired_height * width - program.instructions.len() * width;
    program_cells.extend(iter::repeat(BabyBear::zero()).take(cells_to_add));

    let program_trace = RowMajorMatrix::new(program_cells, width);

    let memory_interface_trace = segment
        .memory_chip
        .borrow()
        .generate_memory_interface_trace();
    let range_checker_trace = segment.range_checker.generate_trace();

    let arithmetic_air = DummyInteractionAir::new(4, false, ARITHMETIC_BUS);
    let mut arithmetic_rows = vec![];
    for arithmetic_op in segment.field_arithmetic_chip.operations.iter() {
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

    segment.cpu_chip.generate_pvs();
    let mut all_public_values = vec![segment.get_cpu_pis(), vec![], vec![], vec![], vec![]];

    let MemoryInterface::Volatile(memory_audit_chip) =
        &segment.memory_chip.borrow().interface_chip;

    let test_result = if field_arithmetic_enabled {
        run_simple_test(
            vec![
                &segment.cpu_chip.air,
                &program_air,
                &memory_audit_chip.air,
                &arithmetic_air,
                &segment.range_checker.air,
            ],
            vec![
                trace,
                program_trace,
                memory_interface_trace,
                arithmetic_trace,
                range_checker_trace,
            ],
            all_public_values,
        )
    } else {
        all_public_values.pop();
        run_simple_test(
            vec![
                &segment.cpu_chip.air,
                &program_air,
                &memory_audit_chip.air,
                &segment.range_checker.air,
            ],
            vec![
                trace,
                program_trace,
                memory_interface_trace,
                range_checker_trace,
            ],
            all_public_values,
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
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Instruction::<BabyBear>::from_isize(STOREW, n, 0, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::<BabyBear>::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::<BabyBear>::large_from_isize(FSUB, 0, 0, 1, 1, 1, 0, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::<BabyBear>::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::<BabyBear>::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    let mut expected_execution: Vec<usize> = vec![0, 1];
    for _ in 0..n {
        expected_execution.push(2);
        expected_execution.push(3);
        expected_execution.push(1);
    }
    expected_execution.push(4);

    let mut expected_arithmetic_operations = vec![];
    for t in 0..n {
        expected_arithmetic_operations.push(FieldArithmeticOperation::<BabyBear>::from_isize(
            FSUB,
            n - t,
            1,
            n - t - 1,
        ));
    }

    execution_test::<TEST_NUM_WORDS, TEST_WORD_SIZE>(
        true,
        false,
        program.clone(),
        expected_execution,
        expected_arithmetic_operations,
    );
    air_test::<TEST_NUM_WORDS, TEST_WORD_SIZE>(true, false, program);
}

#[test]
fn test_cpu_without_field_arithmetic() {
    let field_arithmetic_enabled = false;
    let field_extension_enabled = false;

    /*
    Instruction 0 assigns word[0]_1 to 5.
    Instruction 1 checks if word[0]_1 is *not* 4, and if so jumps to instruction 4.
    Instruction 2 is never run.
    Instruction 3 terminates.
    Instruction 4 checks if word[0]_1 is 5, and if so jumps to instruction 3 to terminate.
     */
    let instructions = vec![
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

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    let expected_execution: Vec<usize> = vec![0, 1, 4, 3];

    execution_test::<TEST_NUM_WORDS, TEST_WORD_SIZE>(
        field_arithmetic_enabled,
        field_extension_enabled,
        program.clone(),
        expected_execution,
        vec![],
    );
    air_test::<TEST_NUM_WORDS, TEST_WORD_SIZE>(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
    );
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
    let instructions = vec![
        // word[0]_1 <- word[6]_0
        Instruction::large_from_isize(STOREW, 6, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BEQ, 0, 4, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 2
        Instruction::from_isize(BNE, 0, 0, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 1
        Instruction::from_isize(BNE, 0, 0, 1, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    air_test_change_pc::<TEST_NUM_WORDS, TEST_WORD_SIZE>(true, false, program, true, 2, 3);
}

#[test]
fn test_cpu_negative_wrong_pc_check() {
    //Same program as test_cpu_negative.
    let instructions = vec![
        // word[0]_1 <- word[6]_0
        Instruction::large_from_isize(STOREW, 6, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BEQ, 0, 4, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 2
        Instruction::from_isize(BNE, 0, 0, 2, 1, 0),
        // if word[0]_1 != 0 then pc += 1
        Instruction::from_isize(BNE, 0, 0, 1, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    air_test_change_pc::<TEST_NUM_WORDS, TEST_WORD_SIZE>(true, false, program, false, 2, 2);
}

#[test]
#[should_panic(
    expected = "assertion `left == right` failed: constraints had nonzero value on row 0"
)]
fn test_cpu_negative_hasnt_terminated() {
    let instructions = vec![
        // word[0]_1 <- word[6]_0
        Instruction::large_from_isize(STOREW, 6, 0, 0, 0, 1, 0, 1),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 2],
    };

    air_test_change(
        true,
        false,
        program,
        true,
        |rows, segment: &mut ExecutionSegment<TEST_NUM_WORDS, TEST_WORD_SIZE, BabyBear>| {
            rows.remove(rows.len() - 1);
            segment.program_chip.execution_frequencies[1] = 0;
        },
    );
}

// #[test]
// #[should_panic(expected = "assertion `left == right` failed")]
// fn test_cpu_negative_secret_write() {
//     let instructions = vec![
//         // if word[0]_0 == word[0]_[0] then pc += 1
//         Instruction::from_isize(BEQ, 0, 0, 1, 0, 0),
//         // terminate
//         Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
//     ];

//     let program = Program {
//         instructions,
//         debug_infos: vec![None; 2],
//     };

//     air_test_change(
//         true,
//         false,
//         program,
//         true,
//         |rows, segment: &mut ExecutionSegment<TEST_NUM_WORDS, TEST_WORD_SIZE, BabyBear>| {
//             let is_zero_air = IsZeroAir;
//             let mut is_zero_trace = is_zero_air
//                 .generate_trace(vec![AbstractField::one()])
//                 .clone();
//             let is_zero_aux = is_zero_trace.row_mut(0)[2];

//             rows[0].aux.accesses[2] = MemoryAccessCols {
//                 enabled: AbstractField::one(),
//                 address_space: AbstractField::one(),
//                 is_immediate: AbstractField::zero(),
//                 is_zero_aux,
//                 address: AbstractField::zero(),
//                 data: decompose(AbstractField::from_canonical_usize(115)),
//             };

//             segment.memory_chip.accesses.push(MemoryAccess::from_isize(
//                 0,
//                 OpType::Write,
//                 1,
//                 0,
//                 115,
//             ));
//         },
//     );
// }

// #[test]
// #[should_panic(expected = "assertion `left == right` failed")]
// fn test_cpu_negative_disable_write() {
//     let instructions = vec![
//         // if word[0]_0 == word[0]_[0] then pc += 1
//         Instruction::from_isize(STOREW, 113, 0, 0, 0, 1),
//         // terminate
//         Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
//     ];

//     let program = Program {
//         instructions,
//         debug_infos: vec![None; 2],
//     };

//     air_test_change(
//         true,
//         false,
//         program,
//         true,
//         |rows, segment: &mut ExecutionSegment<TEST_WORD_SIZE, BabyBear>| {
//             rows[0].aux.accesses[CPU_MAX_READS_PER_CYCLE].enabled = AbstractField::zero();
//             segment.memory_chip.accesses.remove(0);
//         },
//     );
// }
//
// #[test]
// #[should_panic(expected = "assertion `left == right` failed")]
// fn test_cpu_negative_disable_read0() {
//     let instructions = vec![
//         // word[0]_1 <- 0
//         Instruction::large_from_isize(STOREW, 0, 0, 0, 0, 1, 0, 1),
//         // if word[0]_0 == word[0]_[0] then pc += 1
//         Instruction::large_from_isize(LOADW, 0, 0, 0, 1, 1, 0, 1),
//         // terminate
//         Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
//     ];

//     let program = Program {
//         instructions,
//         debug_infos: vec![None; 3],
//     };

//     air_test_change(
//         true,
//         false,
//         program,
//         true,
//         |rows, segment: &mut ExecutionSegment<TEST_WORD_SIZE, BabyBear>| {
//             rows[1].aux.accesses[0].enabled = AbstractField::zero();
//             segment.memory_chip.accesses.remove(1);
//         },
//     );
// }

// #[test]
// #[should_panic(expected = "assertion `left == right` failed")]
// fn test_cpu_negative_disable_read1() {
//     let instructions = vec![
//         // word[0]_1 <- 0
//         Instruction::large_from_isize(STOREW, 0, 0, 0, 0, 1, 0, 1),
//         // if word[0]_0 == word[0]_[0] then pc += 1
//         Instruction::large_from_isize(LOADW, 0, 0, 0, 1, 1, 0, 1),
//         // terminate
//         Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
//     ];
//     let program = Program {
//         instructions,
//         debug_infos: vec![None; 3],
//     };

//     air_test_change(
//         true,
//         false,
//         program,
//         true,
//         |rows, segment: &mut ExecutionSegment<TEST_WORD_SIZE, BabyBear>| {
//             rows[1].aux.accesses[1].enabled = AbstractField::zero();
//             segment.memory_chip.accesses.remove(2);
//         },
//     );
// }

// #[test]
// fn test_cpu_publish() {
//     let index = 2;
//     let value = 4;
//     let instructions = vec![
//         // word[0]_1 <- word[index]_0
//         Instruction::large_from_isize(STOREW, index, 0, 0, 0, 1, 0, 1),
//         // word[1]_1 <- word[value]_0
//         Instruction::large_from_isize(STOREW, value, 0, 1, 0, 1, 0, 1),
//         // public_values[word[0]_1] === word[1]_1
//         Instruction::from_isize(PUBLISH, 0, 1, 0, 1, 1),
//         // terminate
//         Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
//     ];

//     let program = Program {
//         instructions,
//         debug_infos: vec![None; 4],
//     };

//     air_test_change(
//         true,
//         false,
//         program,
//         false,
//         |_, segment: &mut ExecutionSegment<TEST_WORD_SIZE, BabyBear>| {
//             assert_eq!(
//                 segment.public_values[index as usize],
//                 Some(BabyBear::from_canonical_usize(value as usize))
//             );
//         },
//     );
// }
