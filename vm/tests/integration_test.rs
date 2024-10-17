use ax_sdk::{
    config::{
        baby_bear_poseidon2::{default_perm, engine_from_perm, random_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;
use stark_vm::{
    arch::{
        instructions::{
            CoreOpcode::*, FieldArithmeticOpcode::*, FieldExtensionOpcode::*, Keccak256Opcode::*,
            Poseidon2Opcode::*, UsizeOpcode,
        },
        ExecutorName,
    },
    intrinsics::hashes::{keccak::hasher::utils::keccak256, poseidon2::CHUNK},
    system::{
        program::{Instruction, Program},
        vm::{
            config::{MemoryConfig, PersistenceType, VmConfig},
            VirtualMachine,
        },
    },
};

const LIMB_BITS: usize = 29;

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}

fn vm_config_with_field_arithmetic() -> VmConfig {
    VmConfig::core().add_default_executor(ExecutorName::FieldArithmetic)
}

fn air_test(config: VmConfig, program: Program<BabyBear>, witness_stream: Vec<Vec<BabyBear>>) {
    let vm = VirtualMachine::new(config, program, witness_stream);

    let perm = default_perm();
    let fri_params = FriParameters::standard_fast();

    let result = vm.execute_and_generate().unwrap();

    for segment_result in result.segment_results {
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        engine
            .run_test_impl(segment_result.air_proof_inputs)
            .expect("Verification failed");
    }
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_compress_poseidon2(
    poseidon2_max_constraint_degree: usize,
    program: Program<BabyBear>,
    memory_persistence: PersistenceType,
) {
    let vm = VirtualMachine::new(
        VmConfig {
            poseidon2_max_constraint_degree,
            memory_config: MemoryConfig {
                persistence_type: memory_persistence,
                ..Default::default()
            },
            ..VmConfig::core()
        }
        .add_default_executor(ExecutorName::Poseidon2),
        program,
        vec![],
    );

    let result = vm.execute_and_generate().unwrap();

    let perm = random_perm();
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        FriParameters {
            log_blowup: 3,
            num_queries: 2,
            proof_of_work_bits: 0,
        }
    } else {
        standard_fri_params_with_100_bits_conjectured_security(3)
    };

    for segment_result in result.segment_results {
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        let air_proof_inputs = segment_result.air_proof_inputs;

        // Checking maximum constraint degree across all AIRs
        let mut keygen_builder = engine.keygen_builder();
        for air_proof_input in &air_proof_inputs {
            keygen_builder.add_air(air_proof_input.air.clone());
        }
        let pk = keygen_builder.generate_pk();
        assert!(pk.max_constraint_degree == poseidon2_max_constraint_degree);

        engine
            .run_test_impl(air_proof_inputs)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_1() {
    let n = 6;
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
        Instruction::from_isize(STOREW.with_default_offset(), n, 0, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ.with_default_offset(), 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::large_from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL.with_default_offset(), 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(vm_config_with_field_arithmetic(), program, vec![]);
}

#[test]
fn test_vm_1_persistent() {
    let n = 6;
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), n, 0, 0, 0, 1),
        Instruction::large_from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
        Instruction::from_isize(BNE.with_default_offset(), 0, 0, -1, 1, 0),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let config = VmConfig {
        poseidon2_max_constraint_degree: 3,
        memory_config: MemoryConfig::new(1, 16, 10, 6, PersistenceType::Persistent),
        ..VmConfig::core()
    }
    .add_default_executor(ExecutorName::FieldArithmetic);
    let vm = VirtualMachine::new(config, program, vec![]);

    let perm = default_perm();
    let fri_params = FriParameters::standard_fast();

    let result = vm.execute_and_generate().unwrap();

    let segment_result = result.segment_results.into_iter().next().unwrap();

    let merkle_air_proof_input = segment_result
        .air_proof_inputs
        .iter()
        .find(|info| info.air.name() == "MemoryMerkleAir<8>")
        .unwrap();
    assert_eq!(merkle_air_proof_input.raw.public_values.len(), 16);
    assert_eq!(
        merkle_air_proof_input.raw.public_values[..8],
        merkle_air_proof_input.raw.public_values[8..]
    );
    assert_eq!(
        merkle_air_proof_input.raw.public_values[..8],
        // The value when you start with zeros and repeatedly hash the value with itself
        // 13 times. We use 13 because addr_space_max_bits = 1 and pointer_max_bits = 16,
        // so the height of the tree is 1 + 16 - 3 = 14.
        [
            1860730809, 952766590, 1529251869, 978208824, 173743442, 1495326235, 1188286360,
            350327606
        ]
        .map(BabyBear::from_canonical_u32)
    );

    let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
    engine
        .run_test_impl(segment_result.air_proof_inputs)
        .expect("Verification failed");
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
        Instruction::from_isize(STOREW.with_default_offset(), 5, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE.with_default_offset(), 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL.with_default_offset(), 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ.with_default_offset(), 0, 5, -1, 1, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(VmConfig::core(), program, vec![]);
}

#[test]
fn test_vm_fibonacci_old() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 1, 0, 2),
        Instruction::from_isize(BEQ.with_default_offset(), 2, 0, 7, 1, 1),
        Instruction::large_from_isize(ADD.with_default_offset(), 2, 2, 3, 1, 1, 1, 0),
        Instruction::from_isize(LOADW.with_default_offset(), 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(STOREW.with_default_offset(), 6, 0, 2, 1, 2),
        Instruction::from_isize(JAL.with_default_offset(), 7, -6, 0, 1, 0),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(vm_config_with_field_arithmetic(), program.clone(), vec![]);
}

#[test]
fn test_vm_fibonacci_old_cycle_tracker() {
    // NOTE: Instructions commented until cycle tracker instructions are not counted as additional assembly Instructions
    let instructions = vec![
        Instruction::debug(CT_START.with_default_offset(), "full program"),
        Instruction::debug(CT_START.with_default_offset(), "store"),
        Instruction::from_isize(STOREW.with_default_offset(), 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 1, 0, 2),
        Instruction::debug(CT_END.with_default_offset(), "store"),
        Instruction::debug(CT_START.with_default_offset(), "total loop"),
        Instruction::from_isize(BEQ.with_default_offset(), 2, 0, 9, 1, 1), // Instruction::from_isize(BEQ.with_default_offset(), 2, 0, 7, 1, 1),
        Instruction::large_from_isize(ADD.with_default_offset(), 2, 2, 3, 1, 1, 1, 0),
        Instruction::debug(CT_START.with_default_offset(), "inner loop"),
        Instruction::from_isize(LOADW.with_default_offset(), 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(STOREW.with_default_offset(), 6, 0, 2, 1, 2),
        Instruction::debug(CT_END.with_default_offset(), "inner loop"),
        Instruction::from_isize(JAL.with_default_offset(), 7, -8, 0, 1, 0), // Instruction::from_isize(JAL.with_default_offset(), 7, -6, 0, 1, 0),
        Instruction::debug(CT_END.with_default_offset(), "total loop"),
        Instruction::debug(CT_END.with_default_offset(), "full program"),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(vm_config_with_field_arithmetic(), program.clone(), vec![]);
}

#[test]
fn test_vm_field_extension_arithmetic() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 1, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 2, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 3, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 4, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 5, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 6, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 7, 0, 0, 1),
        Instruction::from_isize(FE4ADD.with_default_offset(), 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4ADD.with_default_offset(), 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4SUB.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4MUL.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4DIV.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VmConfig::core()
            .add_default_executor(ExecutorName::FieldArithmetic)
            .add_default_executor(ExecutorName::FieldExtension),
        program,
        vec![],
    );
}

#[test]
fn test_vm_field_extension_arithmetic_persistent() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 1, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 2, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 3, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 4, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 5, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 6, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 2, 7, 0, 0, 1),
        Instruction::from_isize(FE4ADD.with_default_offset(), 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4ADD.with_default_offset(), 8, 0, 4, 1, 1),
        Instruction::from_isize(FE4SUB.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4MUL.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(BBE4DIV.with_default_offset(), 12, 0, 4, 1, 1),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VmConfig {
            poseidon2_max_constraint_degree: 3,
            memory_config: MemoryConfig::new(1, 16, 10, 6, PersistenceType::Persistent),
            ..VmConfig::core()
        }
        .add_default_executor(ExecutorName::FieldArithmetic)
        .add_default_executor(ExecutorName::FieldExtension),
        program,
        vec![],
    );
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 16, 0, 1),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 16, 16777220, 1, 1, 0, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 32, 20, 0, 1, 1, 0, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 20, 1, 1, 1, 0, 0),
        Instruction::from_isize(HINT_INPUT.with_default_offset(), 0, 0, 0, 1, 2),
        Instruction::from_isize(SHINTW.with_default_offset(), 32, 0, 0, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 38, 0, 32, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 44, 20, 0, 1, 1, 0, 0),
        Instruction::from_isize(MUL.with_default_offset(), 24, 38, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 20, 24, 1, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 50, 16, 0, 1, 1, 0, 0),
        Instruction::from_isize(JAL.with_default_offset(), 24, 6, 0, 1, 0),
        Instruction::from_isize(MUL.with_default_offset(), 0, 50, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 0, 44, 0, 1, 1, 1, 0),
        Instruction::from_isize(SHINTW.with_default_offset(), 0, 0, 0, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 50, 50, 1, 1, 1, 0, 0),
        Instruction::from_isize(BNE.with_default_offset(), 50, 38, 2013265917, 1, 1),
        Instruction::from_isize(BNE.with_default_offset(), 50, 38, 2013265916, 1, 1),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    type F = BabyBear;

    let witness_stream: Vec<Vec<F>> = vec![vec![F::two()]];

    air_test(vm_config_with_field_arithmetic(), program, witness_stream);
}

#[test]
fn test_vm_compress_poseidon2_as2() {
    let mut rng = create_seeded_rng();

    let mut instructions = vec![];

    let lhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [lhs_ptr + i]_2 <- rnd()
        instructions.push(Instruction::from_isize(
            STOREW.with_default_offset(),
            rng.gen_range(1..1 << 20),
            i,
            lhs_ptr,
            0,
            2,
        ));
    }
    let rhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [rhs_ptr + i]_2 <- rnd()
        instructions.push(Instruction::from_isize(
            STOREW.with_default_offset(),
            rng.gen_range(1..1 << 20),
            i,
            rhs_ptr,
            0,
            2,
        ));
    }
    let dst_ptr = gen_pointer(&mut rng, CHUNK) as isize;

    // [11]_1 <- lhs_ptr
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
        lhs_ptr,
        0,
        11,
        0,
        1,
    ));
    // [22]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
        rhs_ptr,
        0,
        22,
        0,
        1,
    ));
    // [33]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
        dst_ptr,
        0,
        33,
        0,
        1,
    ));

    instructions.push(Instruction::from_isize(
        COMP_POS2.with_default_offset(),
        33,
        11,
        22,
        1,
        2,
    ));
    instructions.push(Instruction::from_isize(
        TERMINATE.with_default_offset(),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test_with_compress_poseidon2(7, program.clone(), PersistenceType::Volatile);
    air_test_with_compress_poseidon2(3, program.clone(), PersistenceType::Volatile);
    air_test_with_compress_poseidon2(7, program.clone(), PersistenceType::Persistent);
    air_test_with_compress_poseidon2(3, program.clone(), PersistenceType::Persistent);
}

/// Add instruction to write input to memory, call KECCAK256 opcode, then check against expected output
fn instructions_for_keccak256_test(input: &[u8]) -> Vec<Instruction<BabyBear>> {
    let mut instructions = vec![];
    instructions.push(Instruction::from_isize(
        JAL.with_default_offset(),
        0,
        2,
        0,
        1,
        1,
    )); // skip fail
    instructions.push(Instruction::from_isize(
        FAIL.with_default_offset(),
        0,
        0,
        0,
        0,
        0,
    ));

    let [a, b, c] = [1, 0, (1 << LIMB_BITS) - 1];
    // src = word[b]_1 <- 0
    let src = 0;
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
        src,
        0,
        b,
        0,
        1,
    ));
    // dst word[a]_1 <- 3 // use weird offset
    let dst = 8;
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
        dst,
        0,
        a,
        0,
        1,
    ));
    // word[2^29 - 1]_1 <- len // emulate stack
    instructions.push(Instruction::from_isize(
        STOREW.with_default_offset(),
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
            STOREW.with_default_offset(),
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
        KECCAK256.with_default_offset(),
        a,
        b,
        c,
        1,
        2,
        1,
        0,
    ));

    // read expected result to check correctness
    for (i, expected_byte) in expected.into_iter().enumerate() {
        instructions.push(Instruction::from_isize(
            BNE.with_default_offset(),
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
    instructions.push(Instruction::from_isize(
        TERMINATE.with_default_offset(),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test(
        VmConfig::core().add_default_executor(ExecutorName::Keccak256),
        program,
        vec![],
    );
}

// This test dones one keccak in 24 rows, and then there are 8 dummy padding rows which don't make up a full round
#[test]
fn test_vm_keccak_non_full_round() {
    let inputs = [[[0u8; 32], [1u8; 32]].concat()];
    let mut instructions = inputs
        .iter()
        .flat_map(|input| instructions_for_keccak256_test(input))
        .collect::<Vec<_>>();
    instructions.push(Instruction::from_isize(
        TERMINATE.with_default_offset(),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test(
        VmConfig::core().add_default_executor(ExecutorName::Keccak256),
        program,
        vec![],
    );
}
