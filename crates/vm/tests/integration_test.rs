use std::{
    collections::{BTreeMap, VecDeque},
    iter::zip,
    mem::transmute,
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        create_and_initialize_chip_complex,
        execution_control::ExecutionControl,
        execution_mode::{
            e1::E1ExecutionControl, tracegen::TracegenExecutionControlWithSegmentation,
        },
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        interpreter::InterpretedInstance,
        ChipId, MemoryConfig, SingleSegmentVmExecutor, SystemConfig, SystemTraceHeights,
        VirtualMachine, VmComplexTraceHeights, VmConfig, VmInventoryTraceHeights,
        VmSegmentExecutor, VmSegmentState,
    },
    system::{
        memory::{MemoryTraceHeights, VolatileMemoryTraceHeights, CHUNK},
        program::trace::VmCommittedExe,
    },
    utils::{air_test, air_test_with_min_segments},
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, PhantomDiscriminant,
    PublishOpcode::PUBLISH,
    SysPhantom,
    SystemOpcode::*,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::{
    FieldArithmeticOpcode::*, FieldExtensionOpcode::*, NativeBranchEqualOpcode, NativeJalOpcode::*,
    NativeLoadStoreOpcode::*, NativePhantom,
};
use openvm_rv32im_transpiler::BranchEqualOpcode::*;
use openvm_stark_backend::{
    config::StarkGenericConfig, engine::StarkEngine, p3_field::FieldAlgebra,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        setup_tracing, FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use rand::Rng;
use test_log::test;

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}

fn test_native_config() -> NativeConfig {
    NativeConfig {
        system: SystemConfig::new(3, MemoryConfig::new(2, 1, 16, 29, 15, 32, 1024), 0),
        native: Default::default(),
    }
}

fn test_native_continuations_config() -> NativeConfig {
    let mut config = test_native_config();
    config.system = config.system.with_continuations();
    config
}

#[test]
fn test_vm_1() {
    let n = 6;
    /*
    Instruction 0 assigns word[0]_4 to n.
    Instruction 4 terminates
    The remainder is a loop that decrements word[0]_4 until it reaches 0, then terminates.
    Instruction 1 checks if word[0]_4 is 0 yet, and if so sets pc to 5 in order to terminate
    Instruction 2 decrements word[0]_4 (using word[1]_4)
    Instruction 3 uses JAL as a simple jump to go back to instruction 1 (repeating the loop).
     */
    let instructions = vec![
        // word[0]_4 <- word[n]_0
        Instruction::large_from_isize(ADD.global_opcode(), 0, n, 0, 4, 0, 0, 0),
        // if word[0]_4 == 0 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            4,
            0,
        ),
        // word[0]_4 <- word[0]_4 - word[1]_4
        Instruction::large_from_isize(SUB.global_opcode(), 0, 0, 1, 4, 4, 0, 0),
        // word[2]_4 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(test_native_config(), program);
}

#[test]
fn test_vm_override_executor_height() {
    let e = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let program = Program::<BabyBear>::from_instructions(&[
        Instruction::large_from_isize(ADD.global_opcode(), 0, 4, 0, 4, 0, 0, 0),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ]);
    let committed_exe = Arc::new(VmCommittedExe::<BabyBearPoseidon2Config>::commit(
        program.into(),
        e.config().pcs(),
    ));

    // Test getting heights.
    let vm_config = NativeConfig::aggregation(8, 3);

    let executor = SingleSegmentVmExecutor::new(vm_config.clone());
    let res = executor
        .execute_and_compute_heights(committed_exe.exe.clone(), vec![])
        .unwrap();
    // Memory trace heights are not computed during execution.
    assert_eq!(
        res.vm_heights.system,
        SystemTraceHeights {
            memory: MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                boundary: 1,
                access_adapters: vec![0, 0, 0],
            }),
        }
    );
    assert_eq!(
        res.vm_heights.inventory,
        VmInventoryTraceHeights {
            chips: vec![
                (ChipId::Executor(0), 0),
                (ChipId::Executor(1), 0),
                (ChipId::Executor(2), 0),
                (ChipId::Executor(3), 0),
                (ChipId::Executor(4), 0),
                (ChipId::Executor(5), 0),
                (ChipId::Executor(6), 1), // corresponds to FieldArithmeticChip
                (ChipId::Executor(7), 0),
                (ChipId::Executor(8), 0),
                (ChipId::Executor(9), 0),
            ]
            .into_iter()
            .collect(),
        }
    );

    // Test overriding heights.
    let system_overridden_heights = SystemTraceHeights {
        memory: MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
            boundary: 1,
            access_adapters: vec![8, 4, 2],
        }),
    };
    let inventory_overridden_heights = VmInventoryTraceHeights {
        chips: vec![
            (ChipId::Executor(0), 16),
            (ChipId::Executor(1), 32),
            (ChipId::Executor(2), 64),
            (ChipId::Executor(3), 128),
            (ChipId::Executor(4), 256),
            (ChipId::Executor(5), 512),
            (ChipId::Executor(6), 1024),
            (ChipId::Executor(7), 2048),
            (ChipId::Executor(8), 4096),
            (ChipId::Executor(9), 8192),
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
        vec![2, 2, 16, 1, 8, 4, 2, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 262144]
    );
}

#[test]
fn test_vm_1_optional_air() {
    // Aggregation VmConfig has Core/Poseidon2/FieldArithmetic/FieldExtension chips. The program
    // only uses Core and FieldArithmetic. All other chips should not have AIR proof inputs.
    let config = NativeConfig::aggregation(4, 3);
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    let num_airs = pk.per_air.len();

    {
        let n = 6;
        let instructions = vec![
            Instruction::large_from_isize(ADD.global_opcode(), 0, n, 0, 4, 0, 0, 0),
            Instruction::large_from_isize(SUB.global_opcode(), 0, 0, 1, 4, 4, 0, 0),
            Instruction::from_isize(
                NativeBranchEqualOpcode(BNE).global_opcode(),
                0,
                0,
                -(DEFAULT_PC_STEP as isize),
                4,
                0,
            ),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
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
        assert_eq!(proofs.len(), 1);
        vm.verify(&pk.get_vk(), proofs)
            .expect("Verification failed");
    }
}

#[test]
fn test_vm_public_values() {
    setup_tracing();
    let num_public_values = 100;
    let config = SystemConfig::default().with_public_values(num_public_values);
    let engine =
        BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let vm = VirtualMachine::new(engine, config.clone());
    let pk = vm.keygen();

    {
        let instructions = vec![
            Instruction::from_usize(PUBLISH.global_opcode(), [0, 12, 2, 0, 0, 0]),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];

        let program = Program::from_instructions(&instructions);
        let committed_exe = Arc::new(VmCommittedExe::commit(
            program.clone().into(),
            vm.engine.config.pcs(),
        ));
        let single_vm = SingleSegmentVmExecutor::new(config);
        let exe_result = single_vm
            .execute_and_compute_heights(program, vec![])
            .unwrap();
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
    // Program that fails if mem[(4, 7)] != 101.
    let program = Program::from_instructions(&[
        Instruction::<BabyBear>::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            7,
            101,
            2 * DEFAULT_PC_STEP as isize,
            4,
            0,
        ),
        Instruction::<BabyBear>::from_isize(
            PHANTOM.global_opcode(),
            0,
            0,
            SysPhantom::DebugPanic as isize,
            0,
            0,
        ),
        Instruction::<BabyBear>::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ]);

    let raw = unsafe { transmute::<BabyBear, [u8; 4]>(BabyBear::from_canonical_u32(101)) };
    let init_memory = BTreeMap::from_iter((0..4).map(|i| ((4u32, 7u32 * 4 + i), raw[i as usize])));

    let config = test_native_continuations_config();
    let exe = VmExe {
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
    let config = test_native_continuations_config();
    let ptr_max_bits = config.system.memory_config.pointer_max_bits;
    let as_height = config.system.memory_config.as_height;
    let airs = VmConfig::<BabyBear>::create_chip_complex(&config)
        .unwrap()
        .airs::<BabyBearPoseidon2Config>();

    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();

    let n = 6;
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 0, n, 0, 4, 0, 0, 0),
        Instruction::large_from_isize(SUB.global_opcode(), 0, 0, 1, 4, 4, 0, 0),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            0,
            0,
            -(DEFAULT_PC_STEP as isize),
            4,
            0,
        ),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let result = vm.execute_and_generate(program.clone(), vec![]).unwrap();
    {
        let proof_input = result.per_segment.into_iter().next().unwrap();

        let ((_, merkle_air_proof_input), _) = zip(&proof_input.per_air, &airs)
            .find(|(_, air)| air.name() == "MemoryMerkleAir<8>")
            .unwrap();
        assert_eq!(merkle_air_proof_input.raw.public_values.len(), 16);
        assert_eq!(
            merkle_air_proof_input.raw.public_values[..8],
            merkle_air_proof_input.raw.public_values[8..]
        );
        let mut digest = [BabyBear::ZERO; CHUNK];
        let compression = vm_poseidon2_hasher();
        for _ in 0..ptr_max_bits + as_height - 2 {
            digest = compression.compress(&digest, &digest);
        }
        assert_eq!(
            merkle_air_proof_input.raw.public_values[..8],
            // The value when you start with zeros and repeatedly hash the value with itself
            // ptr_max_bits + as_height - 2 times.
            // The height of the tree is ptr_max_bits + as_height - log2(8). The leaf also must be
            // hashed once with padding for security.
            digest
        );
    }

    let result_for_proof = vm.execute_and_generate(program, vec![]).unwrap();
    let proofs = vm.prove(&pk, result_for_proof);
    vm.verify(&pk.get_vk(), proofs)
        .expect("Verification failed");
}

#[test]
fn test_vm_without_field_arithmetic() {
    /*
    Instruction 0 assigns word[0]_4 to 5.
    Instruction 1 checks if word[0]_4 is *not* 4, and if so jumps to instruction 4.
    Instruction 2 is never run.
    Instruction 3 terminates.
    Instruction 4 checks if word[0]_4 is 5, and if so jumps to instruction 3 to terminate.
     */
    let instructions = vec![
        // word[0]_4 <- word[5]_0
        Instruction::large_from_isize(ADD.global_opcode(), 0, 5, 0, 4, 0, 0, 0),
        // if word[0]_4 != 4 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            0,
            4,
            3 * DEFAULT_PC_STEP as isize,
            4,
            0,
        ),
        // word[2]_4 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        // if word[0]_4 == 5 then pc -= 1
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            5,
            -(DEFAULT_PC_STEP as isize),
            4,
            0,
        ),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(test_native_config(), program);
}

#[test]
fn test_vm_fibonacci_old() {
    let instructions = vec![
        // [0]_4 <- [19]_0
        Instruction::large_from_isize(ADD.global_opcode(), 0, 19, 0, 4, 0, 0, 0),
        // [2]_4 <- [11]_0
        Instruction::large_from_isize(ADD.global_opcode(), 2, 11, 0, 4, 0, 0, 0),
        // [3]_4 <- [1]_0
        Instruction::large_from_isize(ADD.global_opcode(), 3, 1, 0, 4, 0, 0, 0),
        // [10]_4 <- [0]_4 + [2]_4
        Instruction::large_from_isize(ADD.global_opcode(), 10, 0, 0, 4, 0, 0, 0),
        // [11]_4 <- [1]_4 + [3]_4
        Instruction::large_from_isize(ADD.global_opcode(), 11, 1, 0, 4, 0, 0, 0),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            2,
            0,
            7 * DEFAULT_PC_STEP as isize,
            4,
            4,
        ),
        // [2]_4 <- [2]_4 + [3]_4
        Instruction::large_from_isize(ADD.global_opcode(), 2, 2, 3, 4, 4, 4, 0),
        // [4]_4 <- [[2]_4 - 2]_4
        Instruction::from_isize(LOADW.global_opcode(), 4, -2, 2, 4, 4),
        // [5]_4 <- [[2]_4 - 1]_4
        Instruction::from_isize(LOADW.global_opcode(), 5, -1, 2, 4, 4),
        // [6]_4 <- [4]_4 + [5]_4
        Instruction::large_from_isize(ADD.global_opcode(), 6, 4, 5, 4, 4, 4, 0),
        // [[2]_4]_4 <- [6]_4
        Instruction::from_isize(STOREW.global_opcode(), 6, 0, 2, 4, 4),
        Instruction::from_isize(
            JAL.global_opcode(),
            7,
            -6 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(test_native_config(), program);
}

#[test]
fn test_vm_fibonacci_old_cycle_tracker() {
    // NOTE: Instructions commented until cycle tracker instructions are not counted as additional
    // assembly Instructions
    let instructions = vec![
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        // [0]_4 <- [19]_0
        Instruction::large_from_isize(ADD.global_opcode(), 0, 19, 0, 4, 0, 0, 0),
        // [2]_4 <- [11]_0
        Instruction::large_from_isize(ADD.global_opcode(), 2, 11, 0, 4, 0, 0, 0),
        // [3]_4 <- [1]_0
        Instruction::large_from_isize(ADD.global_opcode(), 3, 1, 0, 4, 0, 0, 0),
        // [10]_4 <- [0]_4 + [2]_4
        Instruction::large_from_isize(ADD.global_opcode(), 10, 0, 0, 4, 0, 0, 0),
        // [11]_4 <- [1]_4 + [3]_4
        Instruction::large_from_isize(ADD.global_opcode(), 11, 1, 0, 4, 0, 0, 0),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        // if [2]_4 == [0]_4 then pc += 9 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            2,
            0,
            9 * DEFAULT_PC_STEP as isize,
            4,
            4,
        ),
        // [2]_4 <- [2]_4 + [3]_4
        Instruction::large_from_isize(ADD.global_opcode(), 2, 2, 3, 4, 4, 4, 0),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16)),
        // [4]_4 <- [[2]_4 - 2]_4
        Instruction::from_isize(LOADW.global_opcode(), 4, -2, 2, 4, 4),
        // [5]_4 <- [[2]_4 - 1]_4
        Instruction::from_isize(LOADW.global_opcode(), 5, -1, 2, 4, 4),
        // [6]_4 <- [4]_4 + [5]_4
        Instruction::large_from_isize(ADD.global_opcode(), 6, 4, 5, 4, 4, 4, 0),
        // [[2]_4]_4 <- [6]_4
        Instruction::from_isize(STOREW.global_opcode(), 6, 0, 2, 4, 4),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        // [a]_4 <- pc + 4, pc -= 8 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            7,
            -8 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16)),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(test_native_config(), program);
}

#[test]
fn test_vm_field_extension_arithmetic() {
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 1, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 2, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 3, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 4, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 5, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 6, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 7, 0, 2, 4, 0, 0, 0),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4SUB.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4MUL.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4DIV.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    air_test(test_native_config(), program);
}

#[test]
fn test_vm_max_access_adapter_8() {
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 1, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 2, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 3, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 4, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 5, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 6, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 7, 0, 2, 4, 0, 0, 0),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4SUB.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4MUL.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4DIV.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let mut config = test_native_config();
    {
        let chip_complex1 = config.create_chip_complex().unwrap();
        let mem_ctrl1 = chip_complex1.base.memory_controller;
        config.system.memory_config.max_access_adapter_n = 8;
        let chip_complex2 = config.create_chip_complex().unwrap();
        let mem_ctrl2 = chip_complex2.base.memory_controller;
        // AccessAdapterAir with N=16/32 are disabled.
        assert_eq!(mem_ctrl1.air_names().len(), mem_ctrl2.air_names().len() + 2);
        assert_eq!(
            mem_ctrl1.airs::<BabyBearPoseidon2Config>().len(),
            mem_ctrl2.airs::<BabyBearPoseidon2Config>().len() + 2
        );
        assert_eq!(
            mem_ctrl1.current_trace_heights().len(),
            mem_ctrl2.current_trace_heights().len() + 2
        );
    }
    air_test(config, program);
}

#[test]
fn test_vm_field_extension_arithmetic_persistent() {
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 1, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 2, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 3, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 4, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 5, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 6, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 7, 0, 2, 4, 0, 0, 0),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4SUB.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4MUL.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4DIV.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);
    let config = test_native_continuations_config();
    air_test(config, program);
}

#[test]
fn test_vm_hint() {
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 16, 0, 0, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 20, 16, 16777220, 4, 4, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 32, 20, 0, 4, 4, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 20, 20, 1, 4, 4, 0, 0),
        Instruction::from_isize(
            PHANTOM.global_opcode(),
            0,
            0,
            NativePhantom::HintInput as isize,
            0,
            0,
        ),
        Instruction::from_isize(HINT_STOREW.global_opcode(), 32, 0, 0, 4, 4),
        Instruction::from_isize(LOADW.global_opcode(), 38, 0, 32, 4, 4),
        Instruction::large_from_isize(ADD.global_opcode(), 44, 20, 0, 4, 4, 0, 0),
        Instruction::from_isize(MUL.global_opcode(), 24, 38, 1, 4, 4),
        Instruction::large_from_isize(ADD.global_opcode(), 20, 20, 24, 4, 4, 4, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 50, 16, 0, 4, 4, 0, 0),
        Instruction::from_isize(
            JAL.global_opcode(),
            24,
            6 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        Instruction::from_isize(MUL.global_opcode(), 0, 50, 1, 4, 4),
        Instruction::large_from_isize(ADD.global_opcode(), 0, 44, 0, 4, 4, 4, 0),
        Instruction::from_isize(HINT_STOREW.global_opcode(), 0, 0, 0, 4, 4),
        Instruction::large_from_isize(ADD.global_opcode(), 50, 50, 1, 4, 4, 0, 0),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            50,
            38,
            -4 * (DEFAULT_PC_STEP as isize),
            4,
            4,
        ),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            50,
            38,
            -5 * (DEFAULT_PC_STEP as isize),
            4,
            4,
        ),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    type F = BabyBear;

    let input_stream: Vec<Vec<F>> = vec![vec![F::TWO]];
    let config = NativeConfig::new(SystemConfig::default(), Default::default());
    air_test_with_min_segments(config, program, input_stream, 1);
}

#[test]
fn test_hint_load_1() {
    type F = BabyBear;
    let instructions = vec![
        Instruction::phantom(
            PhantomDiscriminant(NativePhantom::HintLoad as u16),
            F::ZERO,
            F::ZERO,
            0,
        ),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let chip_complex = create_and_initialize_chip_complex(
        &test_native_config(),
        program,
        vec![vec![F::ONE, F::TWO]].into(),
        None,
    )
    .unwrap();
    let ctrl = TracegenExecutionControlWithSegmentation::new(chip_complex.air_names());
    let ctx = ExecutionControl::<F, NativeConfig>::initialize_context(&ctrl);
    let mut segment = VmSegmentExecutor::<F, NativeConfig, _>::new(
        chip_complex,
        vec![],
        Default::default(),
        ctrl,
    );

    let mut exec_state = VmSegmentState::new(0, 0, None, ctx);
    segment.execute_from_state(&mut exec_state).unwrap();

    let streams = segment.chip_complex.take_streams();
    assert!(streams.input_stream.is_empty());
    assert_eq!(streams.hint_stream, VecDeque::from(vec![F::ZERO]));
    assert_eq!(streams.hint_space, vec![vec![F::ONE, F::TWO]]);
}

#[test]
fn test_hint_load_2() {
    type F = BabyBear;
    let instructions = vec![
        Instruction::phantom(
            PhantomDiscriminant(NativePhantom::HintLoad as u16),
            F::ZERO,
            F::ZERO,
            0,
        ),
        Instruction::from_isize(HINT_STOREW.global_opcode(), 32, 0, 0, 4, 4),
        Instruction::phantom(
            PhantomDiscriminant(NativePhantom::HintLoad as u16),
            F::ZERO,
            F::ZERO,
            0,
        ),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let chip_complex = create_and_initialize_chip_complex(
        &test_native_config(),
        program,
        vec![vec![F::ONE, F::TWO], vec![F::TWO, F::ONE]].into(),
        None,
    )
    .unwrap();
    let ctrl = TracegenExecutionControlWithSegmentation::new(chip_complex.air_names());
    let ctx = ExecutionControl::<F, NativeConfig>::initialize_context(&ctrl);
    let mut segment = VmSegmentExecutor::<F, NativeConfig, _>::new(
        chip_complex,
        vec![],
        Default::default(),
        ctrl,
    );

    let mut exec_state = VmSegmentState::new(0, 0, None, ctx);
    segment.execute_from_state(&mut exec_state).unwrap();

    let [read] = unsafe {
        segment
            .chip_complex
            .memory_controller()
            .memory
            .data
            .read::<F, 1>(4, 32)
    };
    assert_eq!(read, F::ZERO);
    let streams = segment.chip_complex.take_streams();
    assert!(streams.input_stream.is_empty());
    assert_eq!(streams.hint_stream, VecDeque::from(vec![F::ONE]));
    assert_eq!(
        streams.hint_space,
        vec![vec![F::ONE, F::TWO], vec![F::TWO, F::ONE]]
    );
}

#[test]
fn test_vm_pure_execution_non_continuation() {
    type F = BabyBear;
    let n = 6;
    /*
    Instruction 0 assigns word[0]_4 to n.
    Instruction 4 terminates
    The remainder is a loop that decrements word[0]_4 until it reaches 0, then terminates.
    Instruction 1 checks if word[0]_4 is 0 yet, and if so sets pc to 5 in order to terminate
    Instruction 2 decrements word[0]_4 (using word[1]_4)
    Instruction 3 uses JAL as a simple jump to go back to instruction 1 (repeating the loop).
     */
    let instructions = vec![
        // word[0]_4 <- word[n]_0
        Instruction::large_from_isize(ADD.global_opcode(), 0, n, 0, 4, 0, 0, 0),
        // if word[0]_4 == 0 then pc += 3 * DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            4,
            0,
        ),
        // word[0]_4 <- word[0]_4 - word[1]_4
        Instruction::large_from_isize(SUB.global_opcode(), 0, 0, 1, 4, 4, 0, 0),
        // word[2]_4 <- pc + DEFAULT_PC_STEP, pc -= 2 * DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    let executor = InterpretedInstance::<F, _>::new(test_native_config(), program);
    executor
        .execute(E1ExecutionControl::new(None), vec![])
        .expect("Failed to execute");
}

#[test]
fn test_vm_pure_execution_continuation() {
    type F = BabyBear;
    let instructions = vec![
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 1, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 2, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 3, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 4, 0, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 5, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 6, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(ADD.global_opcode(), 7, 0, 2, 4, 0, 0, 0),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4SUB.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4MUL.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(BBE4DIV.global_opcode(), 12, 0, 4, 4, 4),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);
    let executor = InterpretedInstance::<F, _>::new(test_native_continuations_config(), program);
    executor
        .execute(E1ExecutionControl::new(None), vec![])
        .expect("Failed to execute");
}
