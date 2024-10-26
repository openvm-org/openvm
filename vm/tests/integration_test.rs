use std::{borrow::Borrow, sync::Arc};

use afs_stark_backend::{
    config::Val, keygen::types::MultiStarkVerifyingKey, p3_uni_stark::StarkGenericConfig,
    prover::types::Proof,
};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, setup_tracing,
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
    utils::create_seeded_rng,
};
use axvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    CommonOpcode, PhantomInstruction,
    PublishOpcode::PUBLISH,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::Rng;
use stark_vm::{
    arch::{
        instructions::{
            BranchEqualOpcode::*, CommonOpcode::*, FieldArithmeticOpcode::*,
            FieldExtensionOpcode::*, Keccak256Opcode::*, NativeBranchEqualOpcode,
            NativeJalOpcode::*, NativeLoadStoreOpcode::*, Poseidon2Opcode::*, UsizeOpcode,
        },
        ExecutorName,
    },
    intrinsics::hashes::{keccak::hasher::utils::keccak256, poseidon2::CHUNK},
    sdk::air_test,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{merkle::MemoryMerklePvs, Equipartition},
        program::trace::CommittedProgram,
        vm::{
            chip_set::{CONNECTOR_AIR_ID, MERKLE_AIR_ID},
            config::{MemoryConfig, PersistenceType, VmConfig},
            ExitCode, SingleSegmentVM, VirtualMachine,
        },
    },
};
use test_log::test;

const LIMB_BITS: usize = 29;

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}

fn vm_config_with_field_arithmetic() -> VmConfig {
    VmConfig::default()
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_compress_poseidon2(
    poseidon2_max_constraint_degree: usize,
    program: Program<BabyBear>,
    memory_persistence: PersistenceType,
) {
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        FriParameters {
            log_blowup: 3,
            num_queries: 2,
            proof_of_work_bits: 0,
        }
    } else {
        standard_fri_params_with_100_bits_conjectured_security(3)
    };
    let engine = BabyBearPoseidon2Engine::new(fri_params);

    let vm_config = VmConfig {
        poseidon2_max_constraint_degree,
        memory_config: MemoryConfig {
            persistence_type: memory_persistence,
            ..Default::default()
        },
        ..VmConfig::default()
    }
    .add_executor(ExecutorName::LoadStore)
    .add_executor(ExecutorName::Poseidon2);
    let pk = vm_config.generate_pk(engine.keygen_builder());

    let vm = VirtualMachine::new(vm_config);
    let result = vm.execute_and_generate(program).unwrap();

    for proof_input in result.per_segment {
        engine
            .prove_then_verify(&pk, proof_input)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_1() {
    setup_tracing();
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
        // if word[0]_1 == 0 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).with_default_offset(),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::large_from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.with_default_offset(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VirtualMachine::new(vm_config_with_field_arithmetic()),
        program,
    );
}

#[test]
fn test_vm_1_optional_air() {
    // Default VmConfig has Core/Poseidon2/FieldArithmetic/FieldExtension chips. The program only
    // uses Core and FieldArithmetic. All other chips should not have AIR proof inputs.
    let vm_config = VmConfig::default();
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let pk = vm_config.generate_pk(engine.keygen_builder());
    let num_airs = pk.per_air.len();

    {
        let n = 6;
        let instructions = vec![
            Instruction::from_isize(STOREW.with_default_offset(), n, 0, 0, 0, 1),
            Instruction::large_from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
            Instruction::from_isize(
                NativeBranchEqualOpcode(BNE).with_default_offset(),
                0,
                0,
                -(DEFAULT_PC_STEP as isize),
                1,
                0,
            ),
            Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
        ];

        let program = Program::from_instructions(&instructions);
        let vm = VirtualMachine::new(vm_config);
        let mut result = vm
            .execute_and_generate(program)
            .expect("Failed to execute VM");
        assert_eq!(result.per_segment.len(), 1);
        let proof_input = result.per_segment.pop().unwrap();
        assert!(
            proof_input.per_air.len() < num_airs,
            "Expect less used AIRs"
        );
        engine
            .prove_then_verify(&pk, proof_input)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_public_values() {
    let vm_config = VmConfig {
        num_public_values: 3,
        ..Default::default()
    };
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let pk = vm_config.generate_pk(engine.keygen_builder());

    {
        let instructions = vec![
            Instruction::from_usize(PUBLISH.with_default_offset(), [0, 12, 2, 0, 0, 0]),
            Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
        ];

        let program = Program::from_instructions(&instructions);
        let committed_program = Arc::new(CommittedProgram::commit(&program, engine.config.pcs()));
        let vm = SingleSegmentVM::new(vm_config);
        let pvs = vm.execute(program.clone(), vec![]).unwrap();
        assert_eq!(
            pvs,
            vec![None, None, Some(BabyBear::from_canonical_u32(12))]
        );
        let proof_input = vm.execute_and_generate(committed_program, vec![]).unwrap();
        engine
            .prove_then_verify(&pk, proof_input)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_initial_memory() {
    // Program that fails if mem[(1, 0)] != 101.
    let program = Program::from_instructions(&[
        Instruction::<BabyBear>::from_isize(
            NativeBranchEqualOpcode(BEQ).with_default_offset(),
            0,
            101,
            2 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        Instruction::<BabyBear>::from_isize(
            PHANTOM.with_default_offset(),
            0,
            0,
            PhantomInstruction::DebugPanic as isize,
            0,
            0,
        ),
        Instruction::<BabyBear>::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ]);

    let mut initial_memory = Equipartition::<BabyBear, CHUNK>::new();
    initial_memory.insert(
        (BabyBear::one(), 0),
        [101, 0, 0, 0, 0, 0, 0, 0].map(BabyBear::from_canonical_u32),
    );

    let config = VmConfig {
        poseidon2_max_constraint_degree: 3,
        memory_config: MemoryConfig {
            persistence_type: PersistenceType::Persistent,
            ..Default::default()
        },
        ..VmConfig::default()
    }
    .add_executor(ExecutorName::BranchEqual)
    .add_executor(ExecutorName::Jal);
    let vm = VirtualMachine::new(config).with_initial_memory(initial_memory);
    air_test(vm, program);
}

#[test]
fn test_vm_1_persistent() {
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let config = VmConfig {
        poseidon2_max_constraint_degree: 3,
        memory_config: MemoryConfig::new(1, 16, 10, 6, PersistenceType::Persistent),
        ..VmConfig::default()
    }
    .add_executor(ExecutorName::LoadStore)
    .add_executor(ExecutorName::FieldArithmetic)
    .add_executor(ExecutorName::BranchEqual);
    let pk = config.generate_pk(engine.keygen_builder());

    let n = 6;
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), n, 0, 0, 0, 1),
        Instruction::large_from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            0,
            0,
            -(DEFAULT_PC_STEP as isize),
            1,
            0,
        ),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let vm = VirtualMachine::new(config);
    let result = vm.execute_and_generate(program).unwrap();

    let proof_input = result.per_segment.into_iter().next().unwrap();

    let merkle_air_proof_input = &proof_input
        .per_air
        .iter()
        .find(|(_, info)| info.air.name() == "MemoryMerkleAir<8>")
        .unwrap()
        .1;
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

    engine
        .prove_then_verify(&pk, proof_input)
        .expect("Verification failed");
}

#[test]
fn test_vm_continuations() {
    let n = 200000;

    // Simple Fibonacci program to compute nth Fibonacci number mod BabyBear (with F_0 = 1).
    // Register [0]_1 <- stores the loop counter.
    // Register [1]_1 <- stores F_i at the beginning of iteration i.
    // Register [2]_1 <- stores F_{i+1} at the beginning of iteration i.
    // Register [3]_1 is used as a temporary register.
    let program = Program::from_instructions(&[
        // [0]_1 <- 0
        Instruction::from_isize(ADD.with_default_offset(), 0, 0, 0, 1, 0),
        // [1]_1 <- 0
        Instruction::from_isize(ADD.with_default_offset(), 1, 0, 0, 1, 0),
        // [2]_1 <- 1
        Instruction::from_isize(ADD.with_default_offset(), 2, 0, 1, 1, 0),
        // loop_start
        // [3]_1 <- [1]_1 + [2]_1
        Instruction::large_from_isize(ADD.with_default_offset(), 3, 1, 2, 1, 1, 1, 0),
        // [1]_1 <- [2]_1
        Instruction::large_from_isize(ADD.with_default_offset(), 1, 2, 0, 1, 1, 0, 0),
        // [2]_1 <- [3]_1
        Instruction::large_from_isize(ADD.with_default_offset(), 2, 3, 0, 1, 1, 0, 0),
        // [0]_1 <- [0]_1 + 1
        Instruction::large_from_isize(ADD.with_default_offset(), 0, 0, 1, 1, 1, 0, 0),
        // if [0]_1 != n, pc <- pc - 3
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            n,
            0,
            -4 * DEFAULT_PC_STEP as isize,
            0,
            1,
        ),
        // [0]_3 <- [1]_1
        Instruction::from_isize(ADD.with_default_offset(), 0, 1, 0, 3, 1),
        Instruction::from_isize(
            TERMINATE.with_default_offset(),
            0,
            0,
            ExitCode::Success as isize,
            0,
            0,
        ),
    ]);

    let config = VmConfig {
        num_public_values: 0,
        poseidon2_max_constraint_degree: 3,
        max_segment_len: 200000,
        memory_config: MemoryConfig {
            persistence_type: PersistenceType::Persistent,
            ..Default::default()
        },
        ..VmConfig::default()
    }
    .add_executor(ExecutorName::FieldArithmetic)
    .add_executor(ExecutorName::BranchEqual)
    .add_executor(ExecutorName::Jal);

    let vm = VirtualMachine::new(config);

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let pk = vm.config.generate_pk(engine.keygen_builder());
    let vk = pk.get_vk();
    let result = vm.execute_and_generate(program).unwrap();

    // Let's make sure we have at least 3 segments.
    let num_segments = result.per_segment.len();
    assert!(num_segments >= 3);

    let expected_output = {
        let mut a = 0;
        let mut b = 1;
        for _ in 0..n {
            (a, b) = (b, a + b);
            b %= BabyBear::ORDER_U32;
        }
        BabyBear::from_canonical_u32(a)
    };

    let proofs: Vec<Proof<_>> = result
        .per_segment
        .into_iter()
        .map(|proof_input| engine.prove(&pk, proof_input))
        .collect();
    aggregate_segment_proofs(engine, vk, proofs, vec![expected_output]);
}

fn aggregate_segment_proofs<SC: StarkGenericConfig>(
    engine: impl StarkEngine<SC>,
    vk: MultiStarkVerifyingKey<SC>,
    proofs: Vec<Proof<SC>>,
    _expected_outputs: Vec<Val<SC>>,
) {
    let mut prev_final_memory_root = None;
    let mut prev_final_pc = None;

    for (i, proof) in proofs.iter().enumerate() {
        engine
            .verify(&vk, proof)
            .expect("segment proof should verify");

        // Check public values.
        for air_proof_data in proof.per_air.iter() {
            let pvs = &air_proof_data.public_values;
            let air_vk = &vk.per_air[air_proof_data.air_id];

            if air_proof_data.air_id == CONNECTOR_AIR_ID {
                let pvs: &VmConnectorPvs<_> = pvs.as_slice().borrow();

                // Check initial pc matches the previous final pc.
                assert_eq!(
                    pvs.initial_pc,
                    if i == 0 {
                        // TODO: Make this program PC.
                        Val::<SC>::zero()
                    } else {
                        prev_final_pc.unwrap()
                    }
                );
                prev_final_pc = Some(pvs.final_pc);

                let expected_is_terminate = i == proofs.len() - 1;
                assert_eq!(
                    pvs.is_terminate,
                    Val::<SC>::from_bool(expected_is_terminate)
                );

                let expected_exit_code = if expected_is_terminate {
                    ExitCode::Success as u32
                } else {
                    DEFAULT_SUSPEND_EXIT_CODE
                };
                assert_eq!(
                    pvs.exit_code,
                    Val::<SC>::from_canonical_u32(expected_exit_code)
                );
            } else if air_proof_data.air_id == MERKLE_AIR_ID {
                let pvs: &MemoryMerklePvs<_, CHUNK> = pvs.as_slice().borrow();

                // Check that initial root matches the previous final root.
                if i != 0 {
                    assert_eq!(pvs.initial_root, prev_final_memory_root.unwrap());
                }
                prev_final_memory_root = Some(pvs.final_root);
            } else {
                assert_eq!(pvs.len(), 0);
                assert_eq!(air_vk.params.num_public_values, 0);
            }
        }
    }
    // TODO: Compute root of _expected_outputs and verify Merkle proof to final_memory_root.
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
        // if word[0]_1 != 4 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            0,
            4,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.with_default_offset(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).with_default_offset(),
            0,
            5,
            -(DEFAULT_PC_STEP as isize),
            1,
            0,
        ),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VirtualMachine::new(
            VmConfig::default()
                .add_executor(ExecutorName::LoadStore)
                .add_executor(ExecutorName::BranchEqual)
                .add_executor(ExecutorName::Jal),
        ),
        program,
    );
}

#[test]
fn test_vm_fibonacci_old() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 1, 0, 2),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).with_default_offset(),
            2,
            0,
            7 * DEFAULT_PC_STEP as isize,
            1,
            1,
        ),
        Instruction::large_from_isize(ADD.with_default_offset(), 2, 2, 3, 1, 1, 1, 0),
        Instruction::from_isize(LOADW.with_default_offset(), 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(STOREW.with_default_offset(), 6, 0, 2, 1, 2),
        Instruction::from_isize(
            JAL.with_default_offset(),
            7,
            -6 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VirtualMachine::new(vm_config_with_field_arithmetic()),
        program,
    );
}

#[test]
fn test_vm_fibonacci_old_cycle_tracker() {
    // NOTE: Instructions commented until cycle tracker instructions are not counted as additional assembly Instructions
    let instructions = vec![
        Instruction::debug(PhantomInstruction::CtStart, "full program"),
        Instruction::debug(PhantomInstruction::CtStart, "store"),
        Instruction::from_isize(STOREW.with_default_offset(), 9, 0, 0, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 2, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 3, 0, 1),
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 0, 0, 2),
        Instruction::from_isize(STOREW.with_default_offset(), 1, 0, 1, 0, 2),
        Instruction::debug(PhantomInstruction::CtEnd, "store"),
        Instruction::debug(PhantomInstruction::CtStart, "total loop"),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).with_default_offset(),
            2,
            0,
            9 * DEFAULT_PC_STEP as isize,
            1,
            1,
        ), // Instruction::from_isize(BEQ.with_default_offset(), 2, 0, 7, 1, 1),
        Instruction::large_from_isize(ADD.with_default_offset(), 2, 2, 3, 1, 1, 1, 0),
        Instruction::debug(PhantomInstruction::CtStart, "inner loop"),
        Instruction::from_isize(LOADW.with_default_offset(), 4, -2, 2, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(STOREW.with_default_offset(), 6, 0, 2, 1, 2),
        Instruction::debug(PhantomInstruction::CtEnd, "inner loop"),
        Instruction::from_isize(
            JAL.with_default_offset(),
            7,
            -8 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::debug(PhantomInstruction::CtEnd, "total loop"),
        Instruction::debug(PhantomInstruction::CtEnd, "full program"),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(
        VirtualMachine::new(vm_config_with_field_arithmetic()),
        program,
    );
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

    let vm = VirtualMachine::new(
        VmConfig::default()
            .add_executor(ExecutorName::LoadStore)
            .add_executor(ExecutorName::FieldArithmetic)
            .add_executor(ExecutorName::FieldExtension),
    );

    air_test(vm, program);
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
    let vm = VirtualMachine::new(
        VmConfig {
            poseidon2_max_constraint_degree: 3,
            memory_config: MemoryConfig::new(1, 16, 10, 6, PersistenceType::Persistent),
            ..VmConfig::default()
        }
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::FieldExtension),
    );

    air_test(vm, program);
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
        Instruction::from_isize(STOREW.with_default_offset(), 0, 0, 16, 0, 1),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 16, 16777220, 1, 1, 0, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 32, 20, 0, 1, 1, 0, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 20, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            CommonOpcode::PHANTOM.with_default_offset(),
            0,
            0,
            PhantomInstruction::HintInput as isize,
            1,
            2,
        ),
        Instruction::from_isize(SHINTW.with_default_offset(), 32, 0, 0, 1, 2),
        Instruction::from_isize(LOADW.with_default_offset(), 38, 0, 32, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 44, 20, 0, 1, 1, 0, 0),
        Instruction::from_isize(MUL.with_default_offset(), 24, 38, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 20, 20, 24, 1, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 50, 16, 0, 1, 1, 0, 0),
        Instruction::from_isize(
            JAL.with_default_offset(),
            24,
            6 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::from_isize(MUL.with_default_offset(), 0, 50, 1, 1, 0),
        Instruction::large_from_isize(ADD.with_default_offset(), 0, 44, 0, 1, 1, 1, 0),
        Instruction::from_isize(SHINTW.with_default_offset(), 0, 0, 0, 1, 2),
        Instruction::large_from_isize(ADD.with_default_offset(), 50, 50, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            50,
            38,
            -4 * (DEFAULT_PC_STEP as isize),
            1,
            1,
        ),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            50,
            38,
            -5 * (DEFAULT_PC_STEP as isize),
            1,
            1,
        ),
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    type F = BabyBear;

    let input_stream: Vec<Vec<F>> = vec![vec![F::two()]];
    let vm = VirtualMachine::new(vm_config_with_field_arithmetic()).with_input_stream(input_stream);

    air_test(vm, program);
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
        2 * DEFAULT_PC_STEP as isize,
        0,
        1,
        0,
    )); // skip fail
    instructions.push(Instruction::from_isize(
        PHANTOM.with_default_offset(),
        0,
        0,
        PhantomInstruction::DebugPanic as isize,
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
            NativeBranchEqualOpcode(BNE).with_default_offset(),
            dst + i as isize,
            expected_byte as isize,
            (-(instructions.len() as isize) + 1) * DEFAULT_PC_STEP as isize, // jump to fail
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
        VirtualMachine::new(
            VmConfig::default()
                .add_executor(ExecutorName::LoadStore)
                .add_executor(ExecutorName::Keccak256)
                .add_executor(ExecutorName::BranchEqual)
                .add_executor(ExecutorName::Jal),
        ),
        program,
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
        VirtualMachine::new(
            VmConfig::default()
                .add_executor(ExecutorName::LoadStore)
                .add_executor(ExecutorName::Keccak256)
                .add_executor(ExecutorName::BranchEqual)
                .add_executor(ExecutorName::Jal),
        ),
        program,
    );
}
