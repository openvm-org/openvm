use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use afs_test_utils::config::baby_bear_poseidon2::{
    engine_from_perm, random_perm, run_simple_test_no_pis,
};
use afs_test_utils::config::fri_params::fri_params_with_80_bits_of_security;
use afs_test_utils::engine::StarkEngine;
use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::OpCode::*;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::get_chips;
use stark_vm::vm::VirtualMachine;

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 30;
const DECOMP: usize = 15;

fn air_test(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
    witness_stream: Vec<Vec<BabyBear>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            field_arithmetic_enabled,
            field_extension_enabled,
            compress_poseidon2_enabled: false,
            perm_poseidon2_enabled: false,
            limb_bits: LIMB_BITS,
            decomp: DECOMP,
        },
        program,
        witness_stream,
    );

    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);
    run_simple_test_no_pis(chips, traces).expect("Verification failed");
}

fn air_test_with_poseidon2(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    compress_poseidon2_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            field_arithmetic_enabled,
            field_extension_enabled,
            compress_poseidon2_enabled,
            perm_poseidon2_enabled: false,
            limb_bits: LIMB_BITS,
            decomp: DECOMP,
        },
        program,
        vec![],
    );

    let max_log_degree = vm.max_log_degree().unwrap();
    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);

    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    let num_chips = chips.len();

    engine
        .run_simple_test(chips, traces, vec![vec![]; num_chips])
        .expect("Verification failed");
}

#[test]
fn test_vm_1() {
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

    air_test(true, false, program, vec![]);
}

#[test]
fn test_vm_without_field_arithmetic() {
    let field_arithmetic_enabled = false;
    let field_extension_enabled = false;

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

    air_test(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
        vec![],
    );
}

#[test]
fn test_vm_fibonacci_old() {
    let program = vec![
        Instruction::from_isize(STOREW, 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW, 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW, 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW, 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW, 1, 0, 1, 0, 2),
        Instruction::from_isize(BEQ, 2, 0, 7, 1, 1),
        Instruction::from_isize(FADD, 2, 2, 3, 1, 1),
        Instruction::from_isize(LOADW, 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW, 5, -1, 2, 1, 2),
        Instruction::from_isize(FADD, 6, 4, 5, 1, 1),
        Instruction::from_isize(STOREW, 6, 0, 2, 1, 2),
        Instruction::from_isize(JAL, 7, -6, 0, 1, 0),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    air_test(true, false, program.clone(), vec![]);
}

#[test]
fn test_vm_field_extension_arithmetic() {
    let field_arithmetic_enabled = true;
    let field_extension_enabled = true;

    let program = vec![
        Instruction::from_isize(STOREW, 1, 0, 0, 0, 1),
        Instruction::from_isize(STOREW, 2, 1, 0, 0, 1),
        Instruction::from_isize(STOREW, 1, 2, 0, 0, 1),
        Instruction::from_isize(STOREW, 2, 3, 0, 0, 1),
        Instruction::from_isize(STOREW, 2, 4, 0, 0, 1),
        Instruction::from_isize(STOREW, 1, 5, 0, 0, 1),
        Instruction::from_isize(STOREW, 1, 6, 0, 0, 1),
        Instruction::from_isize(STOREW, 2, 7, 0, 0, 1),
        Instruction::from_isize(FE4ADD, 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4SUB, 12, 0, 4, 1, 1),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    air_test(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
        vec![],
    );
}

#[test]
fn test_vm_hint() {
    let field_arithmetic_enabled = true;
    let field_extension_enabled = false;

    let program = vec![
        Instruction::from_isize(STOREW, 0, 0, 16, 0, 1),
        Instruction::from_isize(FADD, 20, 16, 16777220, 1, 0),
        Instruction::from_isize(FADD, 32, 20, 0, 1, 0),
        Instruction::from_isize(FADD, 20, 20, 1, 1, 0),
        Instruction::from_isize(HINT_INPUT, 0, 0, 0, 1, 2),
        Instruction::from_isize(SHINTW, 32, 0, 0, 1, 2),
        Instruction::from_isize(LOADW, 38, 0, 32, 1, 2),
        Instruction::from_isize(FADD, 44, 20, 0, 1, 0),
        Instruction::from_isize(FMUL, 24, 38, 1, 1, 0),
        Instruction::from_isize(FADD, 20, 20, 24, 1, 1),
        Instruction::from_isize(FADD, 50, 16, 0, 1, 0),
        Instruction::from_isize(JAL, 24, 6, 0, 1, 0),
        Instruction::from_isize(FMUL, 0, 50, 1, 1, 0),
        Instruction::from_isize(FADD, 0, 44, 0, 1, 1),
        Instruction::from_isize(SHINTW, 0, 0, 0, 1, 2),
        Instruction::from_isize(FADD, 50, 50, 1, 1, 0),
        Instruction::from_isize(BNE, 50, 38, 2013265917, 1, 1),
        Instruction::from_isize(BNE, 50, 38, 2013265916, 1, 1),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    type F = BabyBear;

    let witness_stream: Vec<Vec<F>> = vec![vec![F::two()]];

    air_test(
        field_arithmetic_enabled,
        field_extension_enabled,
        program,
        witness_stream,
    );
}

#[test]
fn test_vm_compress_poseidon2() {
    let mut program = vec![];
    let input_a = 37;
    for i in 0..8 {
        program.push(Instruction::from_isize(
            STOREW,
            43 - (7 * i),
            input_a + i,
            0,
            0,
            1,
        ));
    }
    let input_b = 108;
    for i in 0..8 {
        program.push(Instruction::from_isize(
            STOREW,
            2 + (18 * i),
            input_b + i,
            0,
            0,
            1,
        ));
    }
    let output = 4;
    program.push(Instruction::from_isize(
        COMP_POS2, output, input_a, input_b, 0, 1,
    ));
    program.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    air_test_with_poseidon2(false, false, true, program);
}

#[test]
fn test_vm_compress_poseidon2_as2() {
    let mut program = vec![];
    let input_a = 37;
    for i in 0..8 {
        program.push(Instruction::from_isize(
            STOREW,
            43 - (7 * i),
            input_a + i,
            0,
            0,
            2,
        ));
    }
    let input_b = 108;
    for i in 0..8 {
        program.push(Instruction::from_isize(
            STOREW,
            2 + (18 * i),
            input_b + i,
            0,
            0,
            2,
        ));
    }
    let output = 4;
    program.push(Instruction::from_isize(STOREW, input_a, 0, 0, 0, 1));
    program.push(Instruction::from_isize(STOREW, input_b, 1, 0, 0, 1));
    program.push(Instruction::from_isize(STOREW, output, 2, 0, 0, 1));

    program.push(Instruction::from_isize(COMP_POS2, 2, 0, 1, 1, 2));
    program.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    air_test_with_poseidon2(false, false, true, program);
}
