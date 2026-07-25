#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use eyre::Result;
    use itertools::Itertools;
    use openvm_circuit::{
        arch::{
            deferral::{DeferralResult, DeferralState},
            Streams,
        },
        utils::{air_test_with_min_segments, test_system_config},
    };
    #[cfg(feature = "rvr")]
    use openvm_circuit::{
        arch::{
            rvr::RvrCheckpointPreflightLimits, ExecutionError, VirtualMachine, VmExecutor, VmState,
        },
        system::memory::online::{GuestMemory, LinearMemory, TouchedPages, PAGE_SIZE},
        utils::test_cpu_engine,
    };
    use openvm_deferral_circuit::{
        DeferralExtension, DeferralFn, Rv64DeferralBuilder, Rv64DeferralConfig,
    };
    #[cfg(feature = "rvr")]
    use openvm_deferral_transpiler::DeferralOpcode;
    use openvm_deferral_transpiler::DeferralTranspilerExtension;
    use openvm_instructions::{exe::VmExe, DEFERRAL_AS};
    #[cfg(feature = "rvr")]
    use openvm_instructions::{
        instruction::Instruction,
        program::Program,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, SystemOpcode,
    };
    use openvm_riscv_circuit::{Rv64I, Rv64Io, Rv64M};
    #[cfg(feature = "rvr")]
    use openvm_riscv_transpiler::Rv64JalLuiOpcode;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::DIGEST_SIZE,
        openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32},
        p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    const INPUT_COMMIT_0: [u8; 32] = [0x11; 32];
    const INPUT_COMMIT_1: [u8; 32] = [0x22; 32];
    const INPUT_COMMIT_2: [u8; 32] = [0x33; 32];

    const INPUT_RAW_0: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const INPUT_RAW_1: [u8; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    const INPUT_RAW_2: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

    fn make_config(num_deferrals: usize) -> Rv64DeferralConfig {
        let mut system = test_system_config();
        system.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;
        Rv64DeferralConfig {
            system,
            rv64i: Rv64I,
            rv64m: Rv64M::default(),
            io: Rv64Io,
            deferral: make_deferral_extension(num_deferrals),
        }
    }

    fn make_commits(num_deferrals: usize) -> Vec<[F; DIGEST_SIZE]> {
        (0..num_deferrals)
            .map(|_| [F::ONE; DIGEST_SIZE])
            .collect::<Vec<_>>()
    }

    fn make_deferral_extension(num_deferrals: usize) -> DeferralExtension {
        let fns: Vec<_> = (0..num_deferrals)
            .map(|idx| {
                Arc::new(DeferralFn::new(move |input_raw| {
                    let mut prefix_sum = 0u16;
                    input_raw
                        .iter()
                        .map(|&byte| {
                            prefix_sum += byte as u16;
                            (prefix_sum + idx as u16) as u8
                        })
                        .collect()
                }))
            })
            .collect();
        let commits = make_commits(num_deferrals)
            .iter()
            .map(|c| {
                c.iter()
                    .flat_map(|f| f.to_unique_u32().to_le_bytes())
                    .collect_array()
                    .unwrap()
            })
            .collect_vec();
        DeferralExtension::new(fns, commits)
    }

    fn run_test(config: Rv64DeferralConfig, example_name: &str, streams: Streams) -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(DeferralTranspilerExtension::new(
                    config.deferral.def_circuit_commits.clone(),
                )),
        )?;
        air_test_with_min_segments(Rv64DeferralBuilder, config, exe, streams, 1).unwrap();
        Ok(())
    }

    #[test]
    fn test_deferral_single() -> Result<()> {
        let mut state = DeferralState::new(Vec::<DeferralResult>::new());
        state.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
        state.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
        state.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());

        let streams = Streams {
            deferrals: vec![state],
            ..Default::default()
        };
        let config = make_config(1);
        run_test(config, "single", streams)
    }

    #[test]
    fn test_deferral_multiple() -> Result<()> {
        let mut state0 = DeferralState::new(Vec::<DeferralResult>::new());
        state0.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
        state0.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
        state0.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());

        let mut state1 = DeferralState::new(Vec::<DeferralResult>::new());
        state1.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());

        let streams = Streams {
            deferrals: vec![state0, state1],
            ..Default::default()
        };
        let config = make_config(2);
        run_test(config, "multiple", streams)
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_checkpoint_preflight_carries_deferral_state_across_segments() -> Result<()> {
        let config = make_config(1);
        let instructions = [
            Instruction::<F>::from_usize(
                DeferralOpcode::CALL.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    2 * RV64_REGISTER_NUM_LIMBS,
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                ],
            ),
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
            ),
            Instruction::<F>::from_usize(
                DeferralOpcode::CALL.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    2 * RV64_REGISTER_NUM_LIMBS,
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                ],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let output_ptr = 128u64;
        let input_ptr = 64u64;
        let mut exe = VmExe::from(Program::from_instructions(&instructions));
        for (offset, byte) in output_ptr.to_le_bytes().into_iter().enumerate() {
            exe.init_memory.insert(
                (
                    RV64_REGISTER_AS,
                    RV64_REGISTER_NUM_LIMBS as u32 + offset as u32,
                ),
                byte,
            );
        }
        for (offset, byte) in input_ptr.to_le_bytes().into_iter().enumerate() {
            exe.init_memory.insert(
                (
                    RV64_REGISTER_AS,
                    (2 * RV64_REGISTER_NUM_LIMBS) as u32 + offset as u32,
                ),
                byte,
            );
        }
        for (offset, byte) in INPUT_COMMIT_0.into_iter().enumerate() {
            exe.init_memory
                .insert((RV64_MEMORY_AS, input_ptr as u32 + offset as u32), byte);
        }
        for (offset, byte) in config.deferral.def_circuit_commits[0]
            .into_iter()
            .enumerate()
        {
            exe.init_memory.insert((DEFERRAL_AS, offset as u32), byte);
        }

        let mut deferral = DeferralState::new(Vec::<DeferralResult>::new());
        deferral.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
        let streams = Streams {
            deferrals: vec![deferral],
            ..Default::default()
        };
        let executor = VmExecutor::new(config)?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;
        let mut initial = checkpoint.create_initial_vm_state(streams);
        let deferral_bytes = initial.memory.memory.mem[DEFERRAL_AS as usize].size();
        initial.memory.memory.touched_pages[DEFERRAL_AS as usize] =
            TouchedPages::new(deferral_bytes);
        assert!(initial.memory.memory.touched_pages[DEFERRAL_AS as usize]
            .touched_byte_ranges(PAGE_SIZE)
            .is_empty());
        let split_initial = initial.clone();

        let first = checkpoint
            .execute_from_state_for(split_initial, RvrCheckpointPreflightLimits::new(2, 13, 1))?;
        assert!(matches!(
            first.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended { resume_pc: 8, .. }
        ));
        assert_eq!(
            first.state.memory.memory.touched_pages[DEFERRAL_AS as usize]
                .touched_byte_ranges(PAGE_SIZE),
            vec![(0, PAGE_SIZE)]
        );

        let split = checkpoint
            .execute_from_state(first.state, RvrCheckpointPreflightLimits::new(2, 13, 1))?;
        let unbounded =
            checkpoint.execute_from_state(initial, RvrCheckpointPreflightLimits::new(4, 26, 1))?;
        assert_eq!(split.state.pc(), unbounded.state.pc());
        assert_sparse_state_eq(&split.state, &unbounded.state);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn deferral_output_oob_sizing_read_traps_in_every_rvr_mode() -> Result<()> {
        let config = make_config(1);
        let instructions = [
            Instruction::<F>::from_usize(
                DeferralOpcode::OUTPUT.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    2 * RV64_REGISTER_NUM_LIMBS,
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                ],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let mut exe = VmExe::from(Program::from_instructions(&instructions));
        for (offset, byte) in u64::MAX.to_le_bytes().into_iter().enumerate() {
            exe.init_memory.insert(
                (
                    RV64_REGISTER_AS,
                    (2 * RV64_REGISTER_NUM_LIMBS) as u32 + offset as u32,
                ),
                byte,
            );
        }

        let executor = VmExecutor::new(config.clone())?;
        let pure_error = executor
            .rvr_instance(&exe, None)?
            .execute(Streams::default())
            .err()
            .expect("pure RVR must trap an out-of-bounds OUTPUT key");
        assert_rvr_trap(pure_error);

        let checkpoint_error = executor
            .checkpoint_preflight_instance(&exe, None)?
            .execute(
                Streams::default(),
                RvrCheckpointPreflightLimits::new(instructions.len(), 0, 1),
            )
            .err()
            .expect("checkpoint RVR must trap before its OUTPUT sizing peek");
        assert_rvr_trap(checkpoint_error);

        let (vm, _) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64DeferralBuilder, config)?;
        let metered_error = vm
            .metered_instance(&exe)?
            .execute_metered(Streams::default(), vm.build_metered_ctx(&exe))
            .err()
            .expect("metered RVR must trap an out-of-bounds OUTPUT key");
        assert_rvr_trap(metered_error);
        Ok(())
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_trap(error: ExecutionError) {
        match error {
            ExecutionError::RvrExecution(message) => {
                assert_eq!(message, "execution returned error code: 3");
            }
            error => panic!("expected a typed RVR trap, got {error}"),
        }
    }

    #[cfg(feature = "rvr")]
    fn assert_sparse_state_eq(left: &VmState<GuestMemory>, right: &VmState<GuestMemory>) {
        assert_eq!(left.memory.memory.mem.len(), right.memory.memory.mem.len());
        for address_space in 0..left.memory.memory.mem.len() {
            let left_memory = left.memory.memory.mem[address_space].as_slice();
            let right_memory = right.memory.memory.mem[address_space].as_slice();
            assert_eq!(left_memory.len(), right_memory.len());
            let left_ranges = left.memory.memory.touched_pages[address_space]
                .touched_byte_ranges(left_memory.len());
            let right_ranges = right.memory.memory.touched_pages[address_space]
                .touched_byte_ranges(right_memory.len());
            assert_eq!(left_ranges, right_ranges, "address space {address_space}");
            for (start, end) in left_ranges {
                assert_eq!(
                    &left_memory[start..end],
                    &right_memory[start..end],
                    "address space {address_space}, byte range {start}..{end}"
                );
            }
        }
    }
}
