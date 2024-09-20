use std::ops::Deref;

use afs_stark_backend::rap::AnyRap;
use ax_sdk::{
    config::{
        baby_bear_poseidon2::{
            default_perm, engine_from_perm, random_perm, BabyBearPoseidon2Config,
        },
        fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
        setup_tracing_with_log_level,
    },
    engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;
use stark_vm::{
    arch::instructions::Opcode::*,
    hashes::keccak::hasher::utils::keccak256,
    program::{Instruction, Program},
    vm::{
        config::{MemoryConfig, VmConfig},
        VirtualMachine,
    },
};
use tracing::Level;

const LIMB_BITS: usize = 29;
const DECOMP: usize = 16;

fn vm_config_with_field_arithmetic() -> VmConfig {
    VmConfig {
        field_arithmetic_enabled: true,
        ..VmConfig::core()
    }
}

// log_blowup = 2 by default
fn air_test(config: VmConfig, program: Program<BabyBear>, witness_stream: Vec<Vec<BabyBear>>) {
    let vm = VirtualMachine::new(
        VmConfig {
            memory_config: MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
            num_public_values: 4,
            ..config
        },
        program,
        witness_stream,
    );

    // TODO: using log_blowup = 3 because keccak interaction chunking is not optimal right now
    let perm = default_perm();
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };

    let result = vm.execute_and_generate().unwrap();

    for segment_result in result.segment_results {
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        let airs: Vec<_> = segment_result.airs.iter().map(Box::deref).collect();
        engine
            .run_simple_test(&airs, segment_result.traces, &segment_result.public_values)
            .expect("Verification failed");
    }
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_compress_poseidon2(
    poseidon2_max_constraint_degree: usize,
    program: Program<BabyBear>,
) {
    let vm = VirtualMachine::new(
        VmConfig {
            field_arithmetic_enabled: false,
            field_extension_enabled: false,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: false,
            memory_config: MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
            num_public_values: 4,
            poseidon2_max_constraint_degree: Some(poseidon2_max_constraint_degree),
            ..Default::default()
        },
        program,
        vec![],
    );

    let result = vm.execute_and_generate().unwrap();

    let perm = random_perm();
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };

    for segment_result in result.segment_results {
        let airs = segment_result
            .airs
            .iter()
            .map(Box::deref)
            .collect::<Vec<_>>();
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        engine
            .run_simple_test(&airs, segment_result.traces, &segment_result.public_values)
            .expect("Verification failed");

        // Checking maximum constraint degree across all AIRs
        let mut keygen_builder = engine.keygen_builder();
        for (air, pvs) in airs.into_iter().zip(segment_result.public_values) {
            keygen_builder.add_air(air as &dyn AnyRap<BabyBearPoseidon2Config>, pvs.len());
        }
        let pk = keygen_builder.generate_pk();
        assert!(pk.max_constraint_degree == poseidon2_max_constraint_degree);
    }
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
        Instruction::large_from_isize(FSUB, 0, 0, 1, 1, 1, 0, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    air_test(vm_config_with_field_arithmetic(), program, vec![]);
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

    air_test(VmConfig::core(), program, vec![]);
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
        Instruction::large_from_isize(FADD, 2, 2, 3, 1, 1, 1, 0),
        Instruction::from_isize(LOADW, 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW, 5, -1, 2, 1, 2),
        Instruction::large_from_isize(FADD, 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(STOREW, 6, 0, 2, 1, 2),
        Instruction::from_isize(JAL, 7, -6, 0, 1, 0),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(vm_config_with_field_arithmetic(), program.clone(), vec![]);
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
        Instruction::large_from_isize(FADD, 2, 2, 3, 1, 1, 1, 0),
        Instruction::debug(CT_START, "inner loop"),
        Instruction::from_isize(LOADW, 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW, 5, -1, 2, 1, 2),
        Instruction::large_from_isize(FADD, 6, 4, 5, 1, 1, 1, 0),
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

    air_test(vm_config_with_field_arithmetic(), program.clone(), vec![]);
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
        Instruction::from_isize(FE4ADD, 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4SUB, 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4MUL, 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4DIV, 12, 0, 4, 1, 1),
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
    );
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
        Instruction::from_isize(STOREW, 0, 0, 16, 0, 1),
        Instruction::large_from_isize(FADD, 20, 16, 16777220, 1, 1, 0, 0),
        Instruction::large_from_isize(FADD, 32, 20, 0, 1, 1, 0, 0),
        Instruction::large_from_isize(FADD, 20, 20, 1, 1, 1, 0, 0),
        Instruction::from_isize(HINT_INPUT, 0, 0, 0, 1, 2),
        Instruction::from_isize(SHINTW, 32, 0, 0, 1, 2),
        Instruction::from_isize(LOADW, 38, 0, 32, 1, 2),
        Instruction::large_from_isize(FADD, 44, 20, 0, 1, 1, 0, 0),
        Instruction::from_isize(FMUL, 24, 38, 1, 1, 0),
        Instruction::large_from_isize(FADD, 20, 20, 24, 1, 1, 1, 0),
        Instruction::large_from_isize(FADD, 50, 16, 0, 1, 1, 0, 0),
        Instruction::from_isize(JAL, 24, 6, 0, 1, 0),
        Instruction::from_isize(FMUL, 0, 50, 1, 1, 0),
        Instruction::large_from_isize(FADD, 0, 44, 0, 1, 1, 1, 0),
        Instruction::from_isize(SHINTW, 0, 0, 0, 1, 2),
        Instruction::large_from_isize(FADD, 50, 50, 1, 1, 1, 0, 0),
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

    air_test(vm_config_with_field_arithmetic(), program, witness_stream);
}

#[test]
fn test_vm_compress_poseidon2_as2() {
    let mut rng = create_seeded_rng();

    let mut instructions = vec![];

    let lhs_ptr = rng.gen_range(1..1 << 20);
    for i in 0..8 {
        // [lhs_ptr + i]_2 <- rnd()
        instructions.push(Instruction::from_isize(
            STOREW,
            rng.gen_range(1..1 << 20),
            i,
            lhs_ptr,
            0,
            2,
        ));
    }
    let rhs_ptr = rng.gen_range(1..1 << 20);
    for i in 0..8 {
        // [rhs_ptr + i]_2 <- rnd()
        instructions.push(Instruction::from_isize(
            STOREW,
            rng.gen_range(1..1 << 20),
            i,
            rhs_ptr,
            0,
            2,
        ));
    }
    let dst_ptr = rng.gen_range(1..1 << 20);

    // [11]_1 <- lhs_ptr
    instructions.push(Instruction::from_isize(STOREW, lhs_ptr, 0, 11, 0, 1));
    // [22]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(STOREW, rhs_ptr, 0, 22, 0, 1));
    // [33]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(STOREW, dst_ptr, 0, 33, 0, 1));

    instructions.push(Instruction::from_isize(COMP_POS2, 33, 11, 22, 1, 2));
    instructions.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test_with_compress_poseidon2(7, program.clone());
    air_test_with_compress_poseidon2(3, program);
}

/// Add instruction to write input to memory, call KECCAK256 opcode, then check against expected output
fn instructions_for_keccak256_test(input: &[u8]) -> Vec<Instruction<BabyBear>> {
    let mut instructions = vec![];
    instructions.push(Instruction::from_isize(JAL, 0, 2, 0, 1, 1)); // skip fail
    instructions.push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));

    let [a, b, c] = [1, 0, (1 << LIMB_BITS) - 1];
    // src = word[b]_1 <- 0
    let src = 0;
    instructions.push(Instruction::from_isize(STOREW, src, 0, b, 0, 1));
    // dst word[a]_1 <- 3 // use weird offset
    let dst = 3;
    instructions.push(Instruction::from_isize(STOREW, dst, 0, a, 0, 1));
    // word[2^29 - 1]_1 <- len // emulate stack
    instructions.push(Instruction::from_isize(
        STOREW,
        input.len() as isize,
        0,
        c,
        0,
        1,
    ));

    let expected = keccak256(input);
    tracing::debug!(?input, ?expected);

    for (i, byte) in input.iter().enumerate() {
        instructions.push(Instruction::from_isize(
            STOREW,
            *byte as isize,
            0,
            src + i as isize,
            0,
            2,
        ));
    }
    // dst = word[a]_1, src = word[b]_1, len = word[c]_1,
    // read and write io to address space 2
    instructions.push(Instruction::large_from_isize(
        KECCAK256, a, b, c, 1, 2, 1, 0,
    ));

    // read expected result to check correctness
    for (i, expected_byte) in expected.into_iter().enumerate() {
        instructions.push(Instruction::from_isize(
            BNE,
            dst + i as isize,
            expected_byte as isize,
            -(instructions.len() as isize) + 1, // jump to fail
            2,
            0,
        ));
    }
    instructions
}

#[test]
fn test_vm_keccak() {
    setup_tracing_with_log_level(Level::TRACE);
    let inputs = [
        vec![],
        (0u8..1).collect::<Vec<_>>(),
        (0u8..135).collect::<Vec<_>>(),
        (0u8..136).collect::<Vec<_>>(),
        (0u8..200).collect::<Vec<_>>(),
    ];
    let mut instructions = inputs
        .iter()
        .flat_map(|input| instructions_for_keccak256_test(input))
        .collect::<Vec<_>>();
    instructions.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        VmConfig {
            keccak_enabled: true,
            ..VmConfig::core()
        },
        program,
        vec![],
    );
}

// This test dones one keccak in 24 rows, and then there are 8 dummy padding rows which don't make up a full round
#[test]
fn test_vm_keccak_non_full_round() {
    setup_tracing_with_log_level(Level::TRACE);
    let inputs = [[[0u8; 32], [1u8; 32]].concat()];
    let mut instructions = inputs
        .iter()
        .flat_map(|input| instructions_for_keccak256_test(input))
        .collect::<Vec<_>>();
    instructions.push(Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0));

    let program_len = instructions.len();

    let program = Program {
        instructions,
        debug_infos: vec![None; program_len],
    };

    air_test(
        VmConfig {
            keccak_enabled: true,
            ..VmConfig::core()
        },
        program,
        vec![],
    );
}
