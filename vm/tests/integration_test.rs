/*use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use stark_vm::au::AUAir;
use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::CPUChip;
use stark_vm::cpu::OpCode::*;
use stark_vm::program::ProgramAir;

fn air_test(is_field_arithmetic_enabled: bool, program: Vec<Instruction<BabyBear>>) {
    let cpu_chip = CPUChip::new(is_field_arithmetic_enabled);
    let execution = cpu_chip.generate_trace(program);

    let program_air = ProgramAir::new(program);
    let program_trace = program_air.generate_trace(&execution);

    let memory_chip = MemoryChip::new();
    let memory_trace = memory_chip.generate_trace(&execution);

    let field_arithmetic_air = AUAir::new();
    let field_arithmetic_trace = field_arithmetic_air.generate_trace(&execution);

    run_simple_test_no_pis(
        vec![
            &cpu_chip.air,
            &program_air,
            &memory_air,
            &field_arithmetic_air,
        ],
        vec![
            execution.trace(),
            program_trace,
            memory_trace,
            field_arithmetic_trace,
        ],
    )
    .expect("Verification failed");
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

    /*
    Instruction 0 assigns word[0]_1 to n.
    Instruction 1 repeats this to make the trace height a power of 2.
    Instruction 2 assigns word[1]_1 to 1 for use in later arithmetic operations.
    The remainder is a loop that decrements word[0]_1 until it reaches 0, then terminates.
    Instruction 3 checks if word[0]_1 is 0 yet, and if so terminates (by setting pc to -1)
    Instruction 4 decrements word[0]_1 (using word[1]_1)
    Instruction 5 uses JAL as a simple jump to go back to instruction 3 (repeating the loop).
     */
    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction::new(STOREW, nf, zero, zero, zero, one),
        // word[0]_1 <- word[n]_0
        Instruction::new(STOREW, nf, zero, zero, zero, one),
        // word[1]_1 <- word[1]_1
        Instruction::new(STOREW, one, one, zero, zero, one),
        // if word[0]_1 == 0 then pc -= 4
        Instruction::new(BEQ, zero, zero, neg_four, one, zero),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::new(FSUB, zero, zero, one, one, one),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::new(JAL, two, neg_two, zero, one, zero),
    ];

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

    /*
    Instruction 0 assigns word[0]_1 to 5.
    Instruction 1 assigns word[0]_1 to 5 (repeat to make the trace height a power of 2)
    Instruction 2 checks if word[0]_1 is *not* 4, and if so jumps to instruction 4.
    Instruction 3 is never run.
    Instruction 4 checks if word[0]_1 is 5, and if so terminates (by setting pc to -1)
     */
    let program = vec![
        // word[0]_1 <- word[5]_0
        Instruction::new(STOREW, five, zero, zero, zero, one),
        // word[0]_1 <- word[5]_0
        Instruction::new(STOREW, five, zero, zero, zero, one),
        // if word[0]_1 != 4 then pc += 2
        Instruction::new(BNE, zero, four, two, one, zero),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::new(JAL, two, neg_two, zero, one, zero),
        // if word[0]_1 == 5 then pc -= 5
        Instruction::new(BEQ, zero, five, neg_five, one, zero),
    ];

    air_test(field_arithmetic_enabled, program);
}
*/