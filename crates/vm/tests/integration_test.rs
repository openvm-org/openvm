use std::{collections::BTreeMap, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_stark_backend::{
    config::StarkGenericConfig,
    engine::StarkEngine,
    p3_field::{AbstractField, PrimeField32},
};
use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        setup_tracing, FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use axvm_circuit::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        ChipId, ExitCode, MemoryConfig, SingleSegmentVmExecutor, SystemConfig, SystemExecutor,
        SystemPeriphery, SystemTraceHeights, VirtualMachine, VmChipComplex, VmComplexTraceHeights,
        VmConfig, VmInventoryError, VmInventoryTraceHeights,
    },
    derive::{AnyEnum, InstructionExecutor, VmConfig},
    system::{
        memory::{
            tree::public_values::UserPublicValuesProof, MemoryTraceHeights,
            VolatileMemoryTraceHeights, CHUNK,
        },
        program::trace::AxVmCommittedExe,
    },
    utils::{air_test, air_test_with_min_segments},
};
use axvm_instructions::{
    exe::AxVmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    AxVmOpcode, PhantomDiscriminant,
    Poseidon2Opcode::*,
    PublishOpcode::PUBLISH,
    SysPhantom,
    SystemOpcode::*,
};
use axvm_keccak256_circuit::{utils::keccak256, Keccak256, Keccak256Executor, Keccak256Periphery};
use axvm_keccak256_transpiler::Rv32KeccakOpcode::*;
use axvm_native_circuit::{Native, NativeConfig, NativeExecutor, NativePeriphery};
use axvm_native_compiler::{
    FieldArithmeticOpcode::*, FieldExtensionOpcode::*, NativeBranchEqualOpcode, NativeJalOpcode::*,
    NativeLoadStoreOpcode::*, NativePhantom,
};
use axvm_rv32im_transpiler::BranchEqualOpcode::*;
use derive_more::derive::From;
use rand::Rng;
use test_log::test;

const LIMB_BITS: usize = 29;

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_compress_poseidon2(
    poseidon2_max_constraint_degree: usize,
    program: Program<BabyBear>,
    continuation_enabled: bool,
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

    let config = if continuation_enabled {
        NativeConfig::aggregation(0, poseidon2_max_constraint_degree).with_continuations()
    } else {
        NativeConfig::aggregation(0, poseidon2_max_constraint_degree)
    };
    let vm = VirtualMachine::new(engine, config);

    let pk = vm.keygen();
    let result = vm.execute_and_generate(program, vec![]).unwrap();
    let proofs = vm.prove(&pk, result);
    for proof in proofs {
        vm.verify_single(&pk.get_vk(), &proof)
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
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), n, 0, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(SUB), 0, 0, 1, 1, 1, 0, 0),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(NativeConfig::default(), program);
}

#[test]
fn test_vm_override_executor_height() {
    let fri_params = FriParameters::standard_fast();
    let e = BabyBearPoseidon2Engine::new(fri_params);
    let program = Program::<BabyBear>::from_instructions(&[
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 4, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ]);
    let committed_exe = Arc::new(AxVmCommittedExe::<BabyBearPoseidon2Config>::commit(
        program.into(),
        e.config().pcs(),
    ));

    // Test getting heights.
    let vm_config = NativeConfig::aggregation(8, 3);

    let executor = SingleSegmentVmExecutor::new(vm_config.clone());
    let res = executor.execute(committed_exe.exe.clone(), vec![]).unwrap();
    assert_eq!(
        res.internal_heights.system,
        SystemTraceHeights {
            memory: MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                boundary: 1,
                access_adapters: vec![(2, 0), (4, 0), (8, 0)].into_iter().collect(),
            }),
        }
    );
    assert_eq!(
        res.internal_heights.inventory,
        VmInventoryTraceHeights {
            chips: vec![
                (ChipId::Executor(0), 0),
                (ChipId::Executor(1), 0),
                (ChipId::Executor(2), 1),
                (ChipId::Executor(3), 0),
                (ChipId::Executor(4), 0),
                (ChipId::Executor(5), 0),
                (ChipId::Executor(6), 0),
                (ChipId::Executor(7), 0),
                (ChipId::Executor(8), 0),
            ]
            .into_iter()
            .collect(),
        }
    );

    // Test overriding heights.
    let system_overridden_heights = SystemTraceHeights {
        memory: MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
            boundary: 1,
            access_adapters: vec![(2, 8), (4, 4), (8, 2)].into_iter().collect(),
        }),
    };
    let inventory_overridden_heights = VmInventoryTraceHeights {
        chips: vec![
            (ChipId::Executor(0), 1),
            (ChipId::Executor(1), 2),
            (ChipId::Executor(2), 4),
            (ChipId::Executor(3), 8),
            (ChipId::Executor(4), 16),
            (ChipId::Executor(5), 8),
            (ChipId::Executor(6), 4),
            (ChipId::Executor(7), 2),
            (ChipId::Executor(8), 1),
        ]
        .into_iter()
        .collect(),
    };
    let overridden_heights = VmComplexTraceHeights::new(
        system_overridden_heights.clone(),
        inventory_overridden_heights.clone(),
    );
    let executor = SingleSegmentVmExecutor::new_with_overridden_trace_heights(
        vm_config,
        Some(overridden_heights),
    );
    let proof_input = executor
        .execute_and_generate(committed_exe, vec![])
        .unwrap();
    let air_heights: Vec<_> = proof_input
        .per_air
        .iter()
        .map(|(_, api)| api.main_trace_height())
        .collect();
    // It's hard to define the mapping semantically. Please recompute the following magical AIR
    // heights by hands whenever something changes.
    assert_eq!(
        air_heights,
        vec![2, 2, 1, 1, 8, 4, 2, 1, 2, 4, 8, 16, 8, 4, 2, 262144]
    );
}

#[test]
fn test_vm_1_optional_air() {
    // Aggregation VmConfig has Core/Poseidon2/FieldArithmetic/FieldExtension chips. The program only
    // uses Core and FieldArithmetic. All other chips should not have AIR proof inputs.
    let config = NativeConfig::aggregation(4, 3);
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    let num_airs = pk.per_air.len();

    {
        let n = 6;
        let instructions = vec![
            Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), n, 0, 0, 0, 1),
            Instruction::large_from_isize(
                AxVmOpcode::with_default_offset(SUB),
                0,
                0,
                1,
                1,
                1,
                0,
                0,
            ),
            Instruction::from_isize(
                AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
                0,
                0,
                -(DEFAULT_PC_STEP as isize),
                1,
                0,
            ),
            Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
        ];

        let program = Program::from_instructions(&instructions);
        let result = vm
            .execute_and_generate(program, vec![])
            .expect("Failed to execute VM");
        assert_eq!(result.per_segment.len(), 1);
        let proof_input = result.per_segment.last().unwrap();
        assert!(
            proof_input.per_air.len() < num_airs,
            "Expect less used AIRs"
        );
        let proofs = vm.prove(&pk, result);
        vm.verify_single(&pk.get_vk(), &proofs[0])
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_public_values() {
    setup_tracing();
    let num_public_values = 100;
    let config = SystemConfig::default()
        .with_public_values(num_public_values)
        .with_metric_collection();
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let vm = VirtualMachine::new(engine, config.clone());
    let pk = vm.keygen();

    {
        let instructions = vec![
            Instruction::from_usize(
                AxVmOpcode::with_default_offset(PUBLISH),
                [0, 12, 2, 0, 0, 0],
            ),
            Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
        ];

        let program = Program::from_instructions(&instructions);
        let committed_exe = Arc::new(AxVmCommittedExe::commit(
            program.clone().into(),
            vm.engine.config.pcs(),
        ));
        let single_vm = SingleSegmentVmExecutor::new(config);
        let exe_result = single_vm.execute(program, vec![]).unwrap();
        assert_eq!(
            exe_result.public_values,
            [
                vec![None, None, Some(BabyBear::from_canonical_u32(12))],
                vec![None; num_public_values - 3]
            ]
            .concat(),
        );
        let proof_input = single_vm
            .execute_and_generate(committed_exe, vec![])
            .unwrap();
        vm.engine
            .prove_then_verify(&pk, proof_input)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_initial_memory() {
    // Program that fails if mem[(1, 0)] != 101.
    let program = Program::from_instructions(&[
        Instruction::<BabyBear>::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            7,
            101,
            2 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        Instruction::<BabyBear>::from_isize(
            AxVmOpcode::with_default_offset(PHANTOM),
            0,
            0,
            SysPhantom::DebugPanic as isize,
            0,
            0,
        ),
        Instruction::<BabyBear>::from_isize(
            AxVmOpcode::with_default_offset(TERMINATE),
            0,
            0,
            0,
            0,
            0,
        ),
    ]);

    let init_memory: BTreeMap<_, _> = [(
        (BabyBear::ONE, BabyBear::from_canonical_u32(7)),
        BabyBear::from_canonical_u32(101),
    )]
    .into_iter()
    .collect();

    let config = NativeConfig::aggregation(0, 3).with_continuations();
    let exe = AxVmExe {
        program,
        pc_start: 0,
        init_memory,
        fn_bounds: Default::default(),
    };
    air_test(config, exe);
}

#[test]
fn test_vm_1_persistent() {
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let config = NativeConfig {
        system: SystemConfig::new(3, MemoryConfig::new(1, 1, 16, 10, 6, 64), 0),
        native: Default::default(),
    }
    .with_continuations();

    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();

    let n = 6;
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), n, 0, 0, 0, 1),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(SUB), 0, 0, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            0,
            0,
            -(DEFAULT_PC_STEP as isize),
            1,
            0,
        ),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let result = vm.execute_and_generate(program.clone(), vec![]).unwrap();
    {
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
        let mut digest = [BabyBear::ZERO; CHUNK];
        let compression = vm_poseidon2_hasher();
        for _ in 0..15 {
            digest = compression.compress(&digest, &digest);
        }
        assert_eq!(
            merkle_air_proof_input.raw.public_values[..8],
            // The value when you start with zeros and repeatedly hash the value with itself
            // 15 times. We use 15 because addr_space_max_bits = 1 and pointer_max_bits = 16,
            // so the height of the tree is 1 + 16 - 3 = 14. The leaf also must be hashed once
            // with padding for security.
            digest
        );
    }

    let result_for_proof = vm.execute_and_generate(program, vec![]).unwrap();
    let proofs = vm.prove(&pk, result_for_proof);
    vm.verify(&pk.get_vk(), proofs)
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
        Instruction::from_isize(AxVmOpcode::with_default_offset(ADD), 0, 0, 0, 1, 0),
        // [1]_1 <- 0
        Instruction::from_isize(AxVmOpcode::with_default_offset(ADD), 1, 0, 0, 1, 0),
        // [2]_1 <- 1
        Instruction::from_isize(AxVmOpcode::with_default_offset(ADD), 2, 0, 1, 1, 0),
        // loop_start
        // [3]_1 <- [1]_1 + [2]_1
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 3, 1, 2, 1, 1, 1, 0),
        // [1]_1 <- [2]_1
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 1, 2, 0, 1, 1, 0, 0),
        // [2]_1 <- [3]_1
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 2, 3, 0, 1, 1, 0, 0),
        // [0]_1 <- [0]_1 + 1
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 0, 0, 1, 1, 1, 0, 0),
        // if [0]_1 != n, pc <- pc - 3
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            n,
            0,
            -4 * DEFAULT_PC_STEP as isize,
            0,
            1,
        ),
        // [0]_3 <- [1]_1
        Instruction::from_isize(AxVmOpcode::with_default_offset(ADD), 0, 1, 0, 3, 1),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(TERMINATE),
            0,
            0,
            ExitCode::Success as isize,
            0,
            0,
        ),
    ]);

    let config = NativeConfig {
        system: SystemConfig::new(3, MemoryConfig::default(), 0).with_max_segment_len(200000),
        native: Default::default(),
    }
    .with_continuations();

    let expected_output = {
        let mut a = 0;
        let mut b = 1;
        for _ in 0..n {
            (a, b) = (b, a + b);
            b %= BabyBear::ORDER_U32;
        }
        BabyBear::from_canonical_u32(a)
    };

    let memory_dimensions = config.system.memory_config.memory_dimensions();
    let final_state = air_test_with_min_segments(config, program, vec![], 3).unwrap();
    let hasher = vm_poseidon2_hasher();
    let num_public_values = 8;
    let pv_proof =
        UserPublicValuesProof::compute(memory_dimensions, num_public_values, &hasher, &final_state);
    assert_eq!(pv_proof.public_values.len(), num_public_values);
    assert_eq!(pv_proof.public_values[0], expected_output);
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
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 5, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            0,
            4,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            0,
            5,
            -(DEFAULT_PC_STEP as isize),
            1,
            0,
        ),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(NativeConfig::default(), program);
}

#[test]
fn test_vm_fibonacci_old() {
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 9, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 2, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 3, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 0, 0, 0, 0, 2),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 1, 0, 2),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            2,
            0,
            7 * DEFAULT_PC_STEP as isize,
            1,
            1,
        ),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 2, 2, 3, 1, 1, 1, 0),
        Instruction::from_isize(AxVmOpcode::with_default_offset(LOADW), 4, -2, 2, 1, 2),
        Instruction::from_isize(AxVmOpcode::with_default_offset(LOADW), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 6, 0, 2, 1, 2),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            7,
            -6 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(NativeConfig::default(), program);
}

#[test]
fn test_vm_fibonacci_old_cycle_tracker() {
    // NOTE: Instructions commented until cycle tracker instructions are not counted as additional assembly Instructions
    let instructions = vec![
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 9, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 2, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 3, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 0, 0, 0, 0, 2),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 1, 0, 2),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            2,
            0,
            9 * DEFAULT_PC_STEP as isize,
            1,
            1,
        ), // Instruction::from_isize(BEQ.with_default_offset(), 2, 0, 7, 1, 1),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 2, 2, 3, 1, 1, 1, 0),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        Instruction::from_isize(AxVmOpcode::with_default_offset(LOADW), 4, -2, 2, 1, 2),
        Instruction::from_isize(AxVmOpcode::with_default_offset(LOADW), 5, -1, 2, 1, 2),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 6, 4, 5, 1, 1, 1, 0),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 6, 0, 2, 1, 2),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            7,
            -8 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(NativeConfig::default(), program);
}

#[test]
fn test_vm_field_extension_arithmetic() {
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 1, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 2, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 3, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 4, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 5, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 6, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 7, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4SUB), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4MUL), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4DIV), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(NativeConfig::default(), program);
}

#[test]
fn test_vm_max_access_adapter_8() {
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 1, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 2, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 3, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 4, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 5, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 6, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 7, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4SUB), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4MUL), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4DIV), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let mut config = NativeConfig::default();
    {
        let chip_complex1 = config.create_chip_complex().unwrap();
        let mem_ctrl1 = chip_complex1.base.memory_controller.borrow();
        config.system.memory_config.max_access_adapter_n = 8;
        let chip_complex2 = config.create_chip_complex().unwrap();
        let mem_ctrl2 = chip_complex2.base.memory_controller.borrow();
        // AccessAdapterAir with N=16/32/64 are disabled.
        assert_eq!(mem_ctrl1.air_names().len(), mem_ctrl2.air_names().len() + 3);
        assert_eq!(
            mem_ctrl1.airs::<BabyBearPoseidon2Config>().len(),
            mem_ctrl2.airs::<BabyBearPoseidon2Config>().len() + 3
        );
        assert_eq!(
            mem_ctrl1.current_trace_heights().len(),
            mem_ctrl2.current_trace_heights().len() + 3
        );
    }
    air_test(config, program);
}

#[test]
fn test_vm_field_extension_arithmetic_persistent() {
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 0, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 1, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 2, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 3, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 4, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 5, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 6, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 2, 7, 0, 0, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4ADD), 8, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(FE4SUB), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4MUL), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(BBE4DIV), 12, 0, 4, 1, 1),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);
    let config = NativeConfig {
        system: SystemConfig::new(3, MemoryConfig::new(1, 1, 16, 10, 6, 64), 0)
            .with_continuations(),
        native: Default::default(),
    };
    air_test(config, program);
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
        Instruction::from_isize(AxVmOpcode::with_default_offset(STOREW), 0, 0, 16, 0, 1),
        Instruction::large_from_isize(
            AxVmOpcode::with_default_offset(ADD),
            20,
            16,
            16777220,
            1,
            1,
            0,
            0,
        ),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 32, 20, 0, 1, 1, 0, 0),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 20, 20, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(PHANTOM),
            0,
            0,
            NativePhantom::HintInput as isize,
            0,
            0,
        ),
        Instruction::from_isize(AxVmOpcode::with_default_offset(SHINTW), 32, 0, 0, 1, 2),
        Instruction::from_isize(AxVmOpcode::with_default_offset(LOADW), 38, 0, 32, 1, 2),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 44, 20, 0, 1, 1, 0, 0),
        Instruction::from_isize(AxVmOpcode::with_default_offset(MUL), 24, 38, 1, 1, 0),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 20, 20, 24, 1, 1, 1, 0),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 50, 16, 0, 1, 1, 0, 0),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            24,
            6 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        Instruction::from_isize(AxVmOpcode::with_default_offset(MUL), 0, 50, 1, 1, 0),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 0, 44, 0, 1, 1, 1, 0),
        Instruction::from_isize(AxVmOpcode::with_default_offset(SHINTW), 0, 0, 0, 1, 2),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(ADD), 50, 50, 1, 1, 1, 0, 0),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            50,
            38,
            -4 * (DEFAULT_PC_STEP as isize),
            1,
            1,
        ),
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            50,
            38,
            -5 * (DEFAULT_PC_STEP as isize),
            1,
            1,
        ),
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    type F = BabyBear;

    let input_stream: Vec<Vec<F>> = vec![vec![F::TWO]];
    let config = NativeConfig::default();
    air_test_with_min_segments(config, program, input_stream, 1);
}

#[test]
fn test_vm_compress_poseidon2_as2() {
    let mut rng = create_seeded_rng();

    let mut instructions = vec![];

    let lhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [lhs_ptr + i]_2 <- rnd()
        instructions.push(Instruction::from_isize(
            AxVmOpcode::with_default_offset(STOREW),
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
            AxVmOpcode::with_default_offset(STOREW),
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
        AxVmOpcode::with_default_offset(STOREW),
        lhs_ptr,
        0,
        11,
        0,
        1,
    ));
    // [22]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(STOREW),
        rhs_ptr,
        0,
        22,
        0,
        1,
    ));
    // [33]_1 <- rhs_ptr
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(STOREW),
        dst_ptr,
        0,
        33,
        0,
        1,
    ));

    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(COMP_POS2),
        33,
        11,
        22,
        1,
        2,
    ));
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(TERMINATE),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test_with_compress_poseidon2(7, program.clone(), false);
    air_test_with_compress_poseidon2(3, program.clone(), false);
    air_test_with_compress_poseidon2(7, program.clone(), true);
    air_test_with_compress_poseidon2(3, program.clone(), true);
}

/// Add instruction to write input to memory, call KECCAK256 opcode, then check against expected output
fn instructions_for_keccak256_test(input: &[u8]) -> Vec<Instruction<BabyBear>> {
    let mut instructions = vec![];
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(JAL),
        0,
        2 * DEFAULT_PC_STEP as isize,
        0,
        1,
        0,
    )); // skip fail
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(PHANTOM),
        0,
        0,
        SysPhantom::DebugPanic as isize,
        0,
        0,
    ));

    let [a, b, c] = [4, 0, (1 << LIMB_BITS) - 4];
    // [jpw] Cheating here and assuming src, dst, len all bit in a byte so we skip writing the other register bytes
    // src = word[b]_1 <- 0
    let src = 0;
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(STOREW),
        src,
        0,
        b,
        0,
        1,
    ));
    // dst word[a]_1 <- 3 // use weird offset
    let dst = 8;
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(STOREW),
        dst,
        0,
        a,
        0,
        1,
    ));
    // word[c]_1 <- len // emulate stack
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(STOREW),
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
            AxVmOpcode::with_default_offset(STOREW),
            *byte as isize,
            0,
            src + i as isize,
            0,
            2,
        ));
    }
    // dst = word[a]_1, src = word[b]_1, len = word[c]_1,
    // read and write io to address space 2
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(KECCAK256),
        a,
        b,
        c,
        1,
        2,
    ));

    // read expected result to check correctness
    for (i, expected_byte) in expected.into_iter().enumerate() {
        instructions.push(Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            dst + i as isize,
            expected_byte as isize,
            (-(instructions.len() as isize) + 1) * DEFAULT_PC_STEP as isize, // jump to fail
            2,
            0,
        ));
    }
    instructions
}

#[derive(Clone, Debug, VmConfig)]
pub struct NativeKeccakConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub native: Native,
    #[extension]
    pub keccak: Keccak256,
}

impl Default for NativeKeccakConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            native: Default::default(),
            keccak: Default::default(),
        }
    }
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
        AxVmOpcode::with_default_offset(TERMINATE),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test(NativeKeccakConfig::default(), program);
}

// This test does one keccak in 24 rows, and then there are 8 dummy padding rows which don't make up a full round
#[test]
fn test_vm_keccak_non_full_round() {
    let inputs = [[[0u8; 32], [1u8; 32]].concat()];
    let mut instructions = inputs
        .iter()
        .flat_map(|input| instructions_for_keccak256_test(input))
        .collect::<Vec<_>>();
    instructions.push(Instruction::from_isize(
        AxVmOpcode::with_default_offset(TERMINATE),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test(NativeKeccakConfig::default(), program);
}
