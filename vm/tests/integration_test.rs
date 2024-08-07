use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm, run_simple_test},
        fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
    },
    engine::StarkEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_keccak_air::U64_LIMBS;
use stark_vm::{
    cpu::{trace::Instruction, OpCode::*},
    program::Program,
    vm::{
        config::{VmConfig, DEFAULT_MAX_SEGMENT_LEN},
        ExecutionResult, VirtualMachine,
    },
};
use tiny_keccak::keccakf;

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 30;
const DECOMP: usize = 5;

fn vm_config_with_field_arithmetic() -> VmConfig {
    VmConfig {
        field_arithmetic_enabled: true,
        limb_bits: LIMB_BITS,
        decomp: DECOMP,
        ..VmConfig::core()
    }
}

// log_blowup = 2 by default
fn air_test(
    config: VmConfig,
    program: Program<BabyBear>,
    witness_stream: Vec<Vec<BabyBear>>,
    fast_segmentation: bool,
) {
    let vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            limb_bits: LIMB_BITS,
            decomp: DECOMP,
            num_public_values: 4,
            max_segment_len: if fast_segmentation {
                7
            } else {
                DEFAULT_MAX_SEGMENT_LEN
            },
            ..config
        },
        program,
        witness_stream,
    );

    let ExecutionResult {
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: pis,
        ..
    } = vm.execute().unwrap();
    let chips = VirtualMachine::<WORD_SIZE, _>::get_chips(&chips);

    run_simple_test(chips, traces, pis).expect("Verification failed");
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_poseidon2(
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
    compress_poseidon2_enabled: bool,
    program: Program<BabyBear>,
) {
    let vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            field_arithmetic_enabled,
            field_extension_enabled,
            compress_poseidon2_enabled,
            perm_poseidon2_enabled: false,
            limb_bits: LIMB_BITS,
            decomp: DECOMP,
            num_public_values: 4,
            max_segment_len: 6,
            ..Default::default()
        },
        program,
        vec![],
    );

    let ExecutionResult {
        max_log_degree,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: pis,
        ..
    } = vm.execute().unwrap();

    let perm = random_perm();
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    let chips = VirtualMachine::<WORD_SIZE, _>::get_chips(&chips);
    engine
        .run_simple_test(chips, traces, pis)
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
    let instructions = vec![
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

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    air_test(vm_config_with_field_arithmetic(), program, vec![], true);
}

#[test]
fn test_vm_without_field_arithmetic() {
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

    air_test(VmConfig::core(), program, vec![], true);
}

#[test]
fn test_vm_fibonacci_old() {
    let instructions = vec![
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

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        vm_config_with_field_arithmetic(),
        program.clone(),
        vec![],
        true,
    );
}

#[test]
fn test_vm_fibonacci_old_cycle_tracker() {
    // NOTE: Instructions commented until cycle tracker instructions are not counted as additional assembly Instructions
    let instructions = vec![
        Instruction::debug(CT_START, "full program"),
        Instruction::debug(CT_START, "store"),
        Instruction::from_isize(STOREW, 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW, 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW, 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW, 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW, 1, 0, 1, 0, 2),
        Instruction::debug(CT_END, "store"),
        Instruction::debug(CT_START, "total loop"),
        Instruction::from_isize(BEQ, 2, 0, 9, 1, 1), // Instruction::from_isize(BEQ, 2, 0, 7, 1, 1),
        Instruction::from_isize(FADD, 2, 2, 3, 1, 1),
        Instruction::debug(CT_START, "inner loop"),
        Instruction::from_isize(LOADW, 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW, 5, -1, 2, 1, 2),
        Instruction::from_isize(FADD, 6, 4, 5, 1, 1),
        Instruction::from_isize(STOREW, 6, 0, 2, 1, 2),
        Instruction::debug(CT_END, "inner loop"),
        Instruction::from_isize(JAL, 7, -8, 0, 1, 0), // Instruction::from_isize(JAL, 7, -6, 0, 1, 0),
        Instruction::debug(CT_END, "total loop"),
        Instruction::debug(CT_END, "full program"),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        vm_config_with_field_arithmetic(),
        program.clone(),
        vec![],
        false,
    );
}

#[test]
fn test_vm_field_extension_arithmetic() {
    let instructions = vec![
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

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            ..VmConfig::core()
        },
        program,
        vec![],
        true,
    );
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
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

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    type F = BabyBear;

    let witness_stream: Vec<Vec<F>> = vec![vec![F::two()]];

    air_test(
        vm_config_with_field_arithmetic(),
        program,
        witness_stream,
        true,
    );
}

#[test]
fn test_vm_compress_poseidon2_as2() {
    let mut instructions = vec![];
    let input_a = 37;
    for i in 0..8 {
        instructions.push(Instruction::from_isize(
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
        instructions.push(Instruction::from_isize(
            STOREW,
            2 + (18 * i),
            input_b + i,
            0,
            0,
            2,
        ));
    }
    let output = 4;
    // [0]_1 <- input_a
    // [1]_1 <- input_b
    instructions.push(Instruction::from_isize(STOREW, input_a, 0, 0, 0, 1));
    instructions.push(Instruction::from_isize(STOREW, input_b, 1, 0, 0, 1));
    instructions.push(Instruction::from_isize(STOREW, output, 2, 0, 0, 1));

    instructions.push(Instruction::from_isize(COMP_POS2, 2, 0, 1, 1, 2));
    instructions.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test_with_poseidon2(false, false, true, program);
}

#[test]
fn test_vm_permute_keccak() {
    let mut instructions = vec![];
    // src = word[0]_1 <- 0
    let src = 0;
    instructions.push(Instruction::from_isize(STOREW, src, 0, 0, 0, 1));
    // dst word[1]_1 <- 3 // use weird offset
    let dst = 3;
    instructions.push(Instruction::from_isize(STOREW, dst, 0, 1, 0, 1));
    let mut expected = [0u64; 25];

    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..U64_LIMBS {
                let index: usize = (y * 5 + x) * U64_LIMBS + limb;
                instructions.push(Instruction::from_isize(
                    STOREW,
                    ((expected[y * 5] >> (16 * limb)) & 0xFFFF) as isize,
                    0,
                    src + index as isize,
                    0,
                    2,
                ));
            }
        }
    }
    // dst = word[1]_1, src = word[0]_1, read and write to address space 2
    instructions.push(Instruction::from_isize(PERM_KECCAK, 1, 0, 0, 1, 2));

    keccakf(&mut expected);
    // read expected result to check correctness
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..U64_LIMBS {
                let index: usize = (y * 5 + x) * U64_LIMBS + limb;
                instructions.push(Instruction::from_isize(
                    BNE,
                    dst + index as isize,
                    ((expected[y * 5 + x] >> (16 * limb)) & 0xFFFF) as isize,
                    (U64_LIMBS * 25 + 1 - index) as isize, // jump to fail
                    2,
                    0,
                ));
            }
        }
    }
    instructions.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));
    // instructions.push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        VmConfig {
            perm_keccak_enabled: true,
            ..VmConfig::core()
        },
        program,
        vec![],
        false,
    );
}
