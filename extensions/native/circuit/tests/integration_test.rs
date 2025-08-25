use std::{
    collections::{BTreeMap, VecDeque},
    mem::transmute,
    sync::Arc,
};

use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_circuit::system::cuda::extensions::SystemGpuBuilder as SystemBuilder;
#[cfg(not(feature = "cuda"))]
use openvm_circuit::{arch::RowMajorMatrixArena, system::SystemCpuBuilder as SystemBuilder};
use openvm_circuit::{
    arch::{
        execution_mode::metered::segment_ctx::{SegmentationLimits, DEFAULT_SEGMENT_CHECK_INSNS},
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        verify_segments, verify_single, AirInventory, ContinuationVmProver,
        PreflightExecutionOutput, SingleSegmentVmProver, VirtualMachine, VmCircuitConfig,
        VmExecutor, VmInstance, VmState, PUBLIC_VALUES_AIR_ID,
    },
    system::{memory::CHUNK, program::trace::VmCommittedExe},
    utils::{
        air_test, air_test_with_min_segments, test_system_config_without_continuations,
        TestStarkEngine as TestEngine,
    },
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
use openvm_native_circuit::{
    execute_program, test_native_config, test_native_continuations_config,
    test_rv32_with_kernels_config, NativeBuilder, NativeConfig,
};
use openvm_native_compiler::{
    CastfOpcode,
    FieldArithmeticOpcode::*,
    FieldExtensionOpcode::*,
    FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode::{self, *},
    NativeLoadStoreOpcode::*,
    NativePhantom, NativeRangeCheckOpcode, Poseidon2Opcode,
};
use openvm_rv32im_transpiler::BranchEqualOpcode::*;
use openvm_stark_backend::{
    config::StarkGenericConfig, engine::StarkEngine, p3_field::FieldAlgebra,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, setup_tracing,
        FriParameters,
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

    air_test(NativeBuilder::default(), test_native_config(), program);
}

// See crates/sdk/src/prover/root.rs for intended usage
#[test]
fn test_vm_override_trace_heights() -> eyre::Result<()> {
    let e = TestEngine::new(FriParameters::standard_fast());
    let program = Program::<BabyBear>::from_instructions(&[
        Instruction::large_from_isize(ADD.global_opcode(), 0, 4, 0, 4, 0, 0, 0),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ]);
    let committed_exe = Arc::new(VmCommittedExe::<BabyBearPoseidon2Config>::commit(
        program.into(),
        e.config().pcs(),
    ));
    // It's hard to define the mapping semantically. Please recompute the following magical AIR
    // heights by hands whenever something changes.
    let fixed_air_heights = vec![
        2, 2, 16, 1, 8, 4, 2, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 262144,
    ];

    // Test getting heights.
    let vm_config = NativeConfig::aggregation(8, 3);
    let (mut vm, pk) = VirtualMachine::new_with_keygen(e, NativeBuilder::default(), vm_config)?;
    let vk = pk.get_vk();

    let state = vm.create_initial_state(&committed_exe.exe, vec![]);
    vm.transport_init_memory_to_device(&state.memory);
    let cached_program_trace = vm.transport_committed_exe_to_device(&committed_exe);
    vm.load_program(cached_program_trace);
    let mut preflight_interpreter = vm.preflight_interpreter(&committed_exe.exe)?;
    let PreflightExecutionOutput {
        system_records,
        #[cfg(feature = "cuda")]
        record_arenas,
        #[cfg(not(feature = "cuda"))]
        mut record_arenas,
        ..
    } = vm.execute_preflight(&mut preflight_interpreter, state, None, &fixed_air_heights)?;

    let mut expected_actual_heights = vec![0; vk.inner.per_air.len()];
    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    expected_actual_heights[executor_idx_to_air_idx[6]] = 1; // corresponds to FieldArithmeticChip
    #[cfg(not(feature = "cuda"))]
    {
        assert_eq!(
            record_arenas
                .iter()
                .map(|ra| ra.trace_offset() / ra.width())
                .collect_vec(),
            expected_actual_heights
        );
        for ra in &mut record_arenas {
            ra.force_matrix_dimensions();
        }
        vm.override_system_trace_heights(&fixed_air_heights);
    }

    let ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
    let air_heights: Vec<_> = ctx
        .per_air
        .iter()
        .map(|(_, air_ctx)| air_ctx.main_trace_height() as u32)
        .collect();
    assert_eq!(air_heights, fixed_air_heights);
    Ok(())
}

#[test]
fn test_vm_1_optional_air() -> eyre::Result<()> {
    // Aggregation VmConfig has Core/Poseidon2/FieldArithmetic/FieldExtension chips. The program
    // only uses Core and FieldArithmetic. All other chips should not have AIR proof inputs.
    let config = NativeConfig::aggregation(4, 3);
    let engine = TestEngine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let (vm, pk) = VirtualMachine::new_with_keygen(engine, NativeBuilder::default(), config)?;
    let num_airs = pk.per_air.len();

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
    let cached_program_trace = vm.commit_program_on_device(&program);
    let exe = Arc::new(VmExe::new(program));
    let mut prover = VmInstance::new(vm, exe, cached_program_trace)?;
    let proof = SingleSegmentVmProver::prove(&mut prover, vec![], &vec![256; num_airs])?;
    assert!(proof.per_air.len() < num_airs, "Expect less used AIRs");
    verify_single(&prover.vm.engine, &pk.get_vk(), &proof)?;
    Ok(())
}

#[test]
fn test_vm_public_values() -> eyre::Result<()> {
    setup_tracing();
    let num_public_values = 100;
    let config = test_system_config_without_continuations().with_public_values(num_public_values);
    assert!(!config.continuation_enabled);
    let engine = TestEngine::new(standard_fri_params_with_100_bits_conjectured_security(3));
    let (vm, pk) = VirtualMachine::new_with_keygen(engine, SystemBuilder, config)?;

    let instructions = vec![
        Instruction::from_usize(PUBLISH.global_opcode(), [0, 12, 2, 0, 0, 0]),
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let cached_program_trace = vm.commit_program_on_device(&program);
    let exe = Arc::new(VmExe::new(program));
    let mut prover = VmInstance::new(vm, exe, cached_program_trace)?;
    let proof = SingleSegmentVmProver::prove(&mut prover, vec![], &vec![256; pk.per_air.len()])?;
    assert_eq!(
        proof.per_air[PUBLIC_VALUES_AIR_ID].air_id,
        PUBLIC_VALUES_AIR_ID
    );
    assert_eq!(
        proof.per_air[PUBLIC_VALUES_AIR_ID].public_values,
        [
            vec![
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::from_canonical_u32(12)
            ],
            vec![BabyBear::ZERO; num_public_values - 3]
        ]
        .concat(),
    );
    verify_single(&prover.vm.engine, &pk.get_vk(), &proof)?;
    Ok(())
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
    air_test(NativeBuilder::default(), config, exe);
}

#[test]
fn test_vm_1_persistent() -> eyre::Result<()> {
    let engine = TestEngine::new(FriParameters::standard_fast());
    let config = test_native_continuations_config();
    let merkle_air_idx = config.system.memory_boundary_air_id() + 1;
    let ptr_max_bits = config.system.memory_config.pointer_max_bits;
    let addr_space_height = config.system.memory_config.addr_space_height;

    let (vm, pk) = VirtualMachine::new_with_keygen(engine, NativeBuilder::default(), config)?;

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
    let cached_program_trace = vm.commit_program_on_device(&program);
    let exe = Arc::new(VmExe::new(program));
    let mut prover = VmInstance::new(vm, exe, cached_program_trace)?;
    let proof = ContinuationVmProver::prove(&mut prover, vec![])?;

    {
        assert_eq!(proof.per_segment.len(), 1);
        let public_values = proof.per_segment[0].per_air[merkle_air_idx]
            .public_values
            .clone();
        assert_eq!(public_values.len(), 16);
        assert_eq!(public_values[..8], public_values[8..]);
        let mut digest = [BabyBear::ZERO; CHUNK];
        let compression = vm_poseidon2_hasher();
        for _ in 0..ptr_max_bits + addr_space_height - 2 {
            digest = compression.compress(&digest, &digest);
        }
        assert_eq!(
            public_values[..8],
            // The value when you start with zeros and repeatedly hash the value with itself
            // ptr_max_bits + addr_space_height - 2 times.
            // The height of the tree is ptr_max_bits + addr_space_height - log2(8). The leaf also
            // must be hashed once with padding for security.
            digest
        );
    }
    verify_segments(&prover.vm.engine, &pk.get_vk(), &proof.per_segment)?;
    Ok(())
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

    air_test(NativeBuilder::default(), test_native_config(), program);
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

    air_test(NativeBuilder::default(), test_native_config(), program);
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

    air_test(NativeBuilder::default(), test_native_config(), program);
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

    air_test(NativeBuilder::default(), test_native_config(), program);
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
        let num_sys_airs1 = config.system.num_airs();
        let inventory1: AirInventory<BabyBearPoseidon2Config> = config.create_airs().unwrap();
        let num_ext_airs = inventory1.ext_airs().len();
        let mem_inv1 = &inventory1.system().memory;
        config.system.memory_config.max_access_adapter_n = 8;
        let num_sys_airs2 = config.system.num_airs();
        let inventory2: AirInventory<BabyBearPoseidon2Config> = config.create_airs().unwrap();
        let mem_inv2 = &inventory2.system().memory;
        // AccessAdapterAir with N=16/32 are disabled.
        assert_eq!(
            mem_inv1.access_adapters.len(),
            mem_inv2.access_adapters.len() + 2
        );
        assert_eq!(num_sys_airs1, num_sys_airs2 + 2);
        assert_eq!(
            inventory1.into_airs().collect_vec().len(),
            num_sys_airs1 + num_ext_airs
        );
        assert_eq!(
            inventory2.into_airs().collect_vec().len(),
            num_sys_airs2 + num_ext_airs
        );
    }
    air_test(NativeBuilder::default(), test_native_config(), program);
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
    air_test(NativeBuilder::default(), config, program);
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
    let config = test_native_config();
    air_test_with_min_segments(NativeBuilder::default(), config, program, input_stream, 1);
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
    let input = vec![vec![F::ONE, F::TWO]];

    let state = execute_program(program, input);
    let streams = state.streams;
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
    let input = vec![vec![F::ONE, F::TWO], vec![F::TWO, F::ONE]];

    let state = execute_program(program, input);
    let [read] = unsafe { state.memory.read::<F, 1>(4, 32) };
    assert_eq!(read, F::ZERO);
    let streams = state.streams;
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
    let instructions: Vec<Instruction<F>> = vec![
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

    let exe = VmExe::new(Program::from_instructions(&instructions));
    let executor = VmExecutor::new(test_native_config()).unwrap();
    let instance = executor.instance(&exe).unwrap();
    instance.execute(vec![], None).expect("Failed to execute");
}

#[test]
fn test_vm_pure_execution_continuation() {
    type F = BabyBear;
    let instructions: Vec<Instruction<F>> = vec![
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

    let exe = VmExe::new(Program::from_instructions(&instructions));
    let executor = VmExecutor::new(test_native_continuations_config()).unwrap();
    let instance = executor.instance(&exe).unwrap();
    instance.execute(vec![], None).expect("Failed to execute");
}

#[test]
fn test_vm_execute_native_chips() {
    type F = BabyBear;

    let instructions = vec![
        // Field Arithmetic operations (FieldArithmeticChip)
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(SUB.global_opcode(), 1, 10, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(MUL.global_opcode(), 2, 3, 4, 4, 0, 0, 0),
        Instruction::large_from_isize(DIV.global_opcode(), 3, 20, 5, 4, 0, 0, 0),
        // Field Extension operations (FieldExtensionChip)
        Instruction::from_isize(FE4ADD.global_opcode(), 8, 0, 4, 4, 4),
        Instruction::from_isize(FE4SUB.global_opcode(), 12, 8, 4, 4, 4),
        Instruction::from_isize(BBE4MUL.global_opcode(), 16, 12, 8, 4, 4),
        Instruction::from_isize(BBE4DIV.global_opcode(), 20, 16, 12, 4, 4),
        // Branch operations (NativeBranchEqChip)
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            0,
            DEFAULT_PC_STEP as isize,
            4,
            4,
        ),
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            1,
            2,
            DEFAULT_PC_STEP as isize,
            4,
            4,
        ),
        // JAL operation (JalRangeCheckChip)
        Instruction::from_isize(
            NativeJalOpcode::JAL.global_opcode(),
            24,
            DEFAULT_PC_STEP as isize,
            0,
            4,
            0,
        ),
        // Range check operation (JalRangeCheckChip)
        Instruction::from_isize(
            NativeRangeCheckOpcode::RANGE_CHECK.global_opcode(),
            0,
            10,
            8,
            4,
            0,
        ),
        // Load/Store operations (NativeLoadStoreChip)
        Instruction::from_isize(STOREW.global_opcode(), 0, 0, 28, 4, 4),
        Instruction::from_isize(LOADW.global_opcode(), 32, 0, 28, 4, 4),
        Instruction::from_isize(
            PHANTOM.global_opcode(),
            0,
            0,
            NativePhantom::HintInput as isize,
            0,
            0,
        ),
        Instruction::from_isize(HINT_STOREW.global_opcode(), 32, 0, 0, 4, 4),
        // Cast to field operation (CastFChip)
        Instruction::from_usize(CastfOpcode::CASTF.global_opcode(), [36, 40, 0, 2, 4]),
        // Poseidon2 operations (Poseidon2Chip)
        Instruction::new(
            Poseidon2Opcode::PERM_POS2.global_opcode(),
            F::from_canonical_usize(44),
            F::from_canonical_usize(48),
            F::ZERO,
            F::from_canonical_usize(4),
            F::from_canonical_usize(4),
            F::ZERO,
            F::ZERO,
        ),
        Instruction::new(
            Poseidon2Opcode::COMP_POS2.global_opcode(),
            F::from_canonical_usize(52),
            F::from_canonical_usize(44),
            F::from_canonical_usize(48),
            F::from_canonical_usize(4),
            F::from_canonical_usize(4),
            F::ZERO,
            F::ZERO,
        ),
        // FRI operation (FriReducedOpeningChip)
        Instruction::large_from_isize(ADD.global_opcode(), 60, 64, 0, 4, 4, 0, 0), /* a_pointer_pointer, */
        Instruction::large_from_isize(ADD.global_opcode(), 64, 68, 0, 4, 4, 0, 0), /* b_pointer_pointer, */
        Instruction::large_from_isize(ADD.global_opcode(), 68, 2, 0, 4, 0, 0, 0), /* length_pointer (value 2), */
        Instruction::large_from_isize(ADD.global_opcode(), 72, 1, 0, 4, 0, 0, 0), //alpha_pointer
        Instruction::large_from_isize(ADD.global_opcode(), 76, 80, 0, 4, 4, 0, 0), /* result_pointer, */
        Instruction::large_from_isize(ADD.global_opcode(), 80, 1, 0, 4, 0, 0, 0), /* is_init (value 1) , */
        Instruction::from_usize(
            FriOpcode::FRI_REDUCED_OPENING.global_opcode(),
            [60, 64, 68, 72, 76, 0, 80],
        ),
        // Terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let exe = VmExe::new(Program::from_instructions(&instructions));
    let input_stream: Vec<Vec<F>> = vec![vec![]];

    let executor = VmExecutor::new(test_rv32_with_kernels_config()).unwrap();
    let instance = executor.instance(&exe).unwrap();
    instance
        .execute(input_stream, None)
        .expect("Failed to execute");
}

// This test ensures that metered execution never segments when continuations is disabled
#[test]
fn test_single_segment_executor_no_segmentation() {
    setup_tracing();

    let mut config = test_native_config();
    config
        .system
        .set_segmentation_limits(SegmentationLimits::default().with_max_trace_height(1));

    let engine = TestEngine::new(FriParameters::new_for_testing(3));
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, NativeBuilder::default(), config).unwrap();
    let instructions: Vec<_> = (0..2 * DEFAULT_SEGMENT_CHECK_INSNS)
        .map(|_| Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0))
        .chain(std::iter::once(Instruction::from_isize(
            TERMINATE.global_opcode(),
            0,
            0,
            0,
            0,
            0,
        )))
        .collect();

    let exe = VmExe::new(Program::from_instructions(&instructions));
    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let metered_ctx = vm.build_metered_ctx();
    vm.executor()
        .metered_instance(&exe, &executor_idx_to_air_idx)
        .unwrap()
        .execute_metered(vec![], metered_ctx)
        .unwrap();
}

#[test]
fn test_vm_execute_metered_cost_native_chips() {
    type F = BabyBear;

    setup_tracing();
    let config = test_native_config();

    let engine = TestEngine::new(FriParameters::new_for_testing(3));
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, NativeBuilder::default(), config).unwrap();

    let instructions = vec![
        // Field Arithmetic operations (FieldArithmeticChip)
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(SUB.global_opcode(), 1, 10, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(MUL.global_opcode(), 2, 3, 4, 4, 0, 0, 0),
        Instruction::large_from_isize(DIV.global_opcode(), 3, 20, 5, 4, 0, 0, 0),
        // Terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let exe = VmExe::new(Program::<F>::from_instructions(&instructions));

    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let instance = vm
        .executor()
        .metered_cost_instance(&exe, &executor_idx_to_air_idx)
        .unwrap();
    let ctx = vm.build_metered_cost_ctx();
    let (cost, VmState { instret, .. }) = instance
        .execute_metered_cost(vec![], ctx)
        .expect("Failed to execute");

    assert_eq!(instret, instructions.len() as u64);
    assert!(cost > 0);
}

#[test]
fn test_vm_execute_metered_cost_halt() {
    type F = BabyBear;

    setup_tracing();
    let config = test_native_config();

    let engine = TestEngine::new(FriParameters::new_for_testing(3));
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, NativeBuilder::default(), config.clone()).unwrap();

    let instructions = vec![
        // Field Arithmetic operations (FieldArithmeticChip)
        Instruction::large_from_isize(ADD.global_opcode(), 0, 0, 1, 4, 0, 0, 0),
        Instruction::large_from_isize(SUB.global_opcode(), 1, 10, 2, 4, 0, 0, 0),
        Instruction::large_from_isize(MUL.global_opcode(), 2, 3, 4, 4, 0, 0, 0),
        Instruction::large_from_isize(DIV.global_opcode(), 3, 20, 5, 4, 0, 0, 0),
        // Terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let exe = VmExe::new(Program::<F>::from_instructions(&instructions));

    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let instance1 = vm
        .executor()
        .metered_cost_instance(&exe, &executor_idx_to_air_idx)
        .unwrap();
    let ctx = vm.build_metered_cost_ctx();
    let (
        cost1,
        VmState {
            instret: instret1, ..
        },
    ) = instance1
        .execute_metered_cost(vec![], ctx)
        .expect("Failed to execute");

    assert_eq!(instret1, instructions.len() as u64);

    let executor_idx_to_air_idx2 = vm.executor_idx_to_air_idx();
    let instance2 = vm
        .executor()
        .metered_cost_instance(&exe, &executor_idx_to_air_idx2)
        .unwrap();
    let ctx2 = vm.build_metered_cost_ctx().with_max_execution_cost(0);
    let (
        cost2,
        VmState {
            instret: instret2, ..
        },
    ) = instance2
        .execute_metered_cost(vec![], ctx2)
        .expect("Failed to execute");

    assert_eq!(instret2, 1);
    assert!(cost2 < cost1);
}
