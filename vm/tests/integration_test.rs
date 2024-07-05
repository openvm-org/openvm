use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;

use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::OpCode::*;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::config::VmParamsConfig;
use stark_vm::vm::get_chips;
use stark_vm::vm::VirtualMachine;

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 16;
const DECOMP: usize = 8;

fn air_test(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    program: Vec<Instruction<BabyBear>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled,
                field_extension_enabled,
                limb_bits: LIMB_BITS,
                decomp: DECOMP,
            },
        },
        program,
    );

    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);
    run_simple_test_no_pis(chips, traces).expect("Verification failed");
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

    air_test(true, false, program);
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

    air_test(field_arithmetic_enabled, field_extension_enabled, program);
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

    air_test(true, false, program.clone());
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

    air_test(field_arithmetic_enabled, field_extension_enabled, program);
}
