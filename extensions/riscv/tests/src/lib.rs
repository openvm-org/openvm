#[cfg(test)]
mod tests {
    #[cfg(feature = "rvr")]
    use std::{sync::Barrier, thread, time::Instant};

    use eyre::Result;
    #[cfg(feature = "rvr")]
    use openvm_circuit::arch::{
        ExecutionOutcome, PreflightExecutionOutput, VirtualMachine, VmState,
    };
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, VmExecutor},
        system::memory::{
            merkle::{
                public_values::{extract_public_values, UserPublicValuesProof},
                MerkleTree,
            },
            online::LinearMemory,
        },
        utils::{air_test, air_test_with_min_segments, test_system_config},
    };
    use openvm_instructions::{
        exe::VmExe, instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode,
        SystemOpcode,
    };
    use openvm_riscv_circuit::{Rv64IBuilder, Rv64IConfig, Rv64ImBuilder, Rv64ImConfig};
    use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;
    use openvm_riscv_transpiler::{
        DivRemOpcode, MulHOpcode, MulOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
        Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::{
        openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    #[cfg(feature = "rvr")]
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use strum::IntoEnumIterator;
    use test_case::test_case;
    #[cfg(feature = "rvr")]
    use {
        openvm_circuit::{
            system::memory::online::{GuestMemory, PAGE_SIZE},
            utils::test_cpu_engine,
        },
        openvm_instructions::{
            program::Program,
            riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
            SysPhantom, PUBLIC_VALUES_AS,
        },
        openvm_riscv_transpiler::{
            BaseAluImmOpcode, BaseAluOpcode, BranchEqualOpcode, Rv64HintStoreOpcode,
            Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode, Rv64Phantom,
        },
        openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeField32,
    };

    type F = BabyBear;
    #[cfg(test)]
    fn test_rv64im_config() -> Rv64ImConfig {
        Rv64ImConfig {
            rv64i: Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[cfg(feature = "rvr")]
    fn callback_phantom_exe() -> VmExe<F> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let instructions = [
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                0,
                0,
                Rv64Phantom::HintInput as isize,
                0,
                0,
            ),
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
            ),
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                reg(1) as isize,
                0,
                Rv64Phantom::HintRandom as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                reg(2) as isize,
                reg(3) as isize,
                Rv64Phantom::PrintStr as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        VmExe::from(Program::from_instructions(&instructions))
    }

    #[cfg(feature = "rvr")]
    fn configure_callback_state(mut state: VmState<GuestMemory>) -> VmState<GuestMemory> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        state.streams.hint_stream.set_hint(vec![0xa5]);
        unsafe {
            state
                .memory
                .write_bytes(RV64_REGISTER_AS, reg(1) as u32, 1u64.to_le_bytes());
            state
                .memory
                .write_bytes(RV64_REGISTER_AS, reg(2) as u32, 0u64.to_le_bytes());
            state
                .memory
                .write_bytes(RV64_REGISTER_AS, reg(3) as u32, 3u64.to_le_bytes());
            state.memory.write_bytes(RV64_MEMORY_AS, 0, *b"ok\n");
        }
        state
    }

    #[cfg(feature = "rvr")]
    fn take_hint(state: &mut VmState<GuestMemory>) -> Vec<u8> {
        let mut hint = vec![0; state.streams.hint_stream.remaining()];
        state.streams.hint_stream.copy_to_slice(&mut hint);
        hint
    }

    #[cfg(feature = "rvr")]
    fn configure_hint_state(
        mut state: VmState<GuestMemory>,
        registers: &[(usize, u64)],
        hint_words: &[u64],
    ) -> VmState<GuestMemory> {
        for &(index, value) in registers {
            unsafe {
                state.memory.write_bytes(
                    RV64_REGISTER_AS,
                    (index * RV64_REGISTER_NUM_LIMBS) as u32,
                    value.to_le_bytes(),
                );
            }
        }
        state.streams.hint_stream.set_hint(
            hint_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect(),
        );
        state
    }

    #[cfg(feature = "rvr")]
    fn read_main_word(state: &VmState<GuestMemory>, byte_addr: u32) -> u64 {
        let limbs: [u16; 4] = unsafe { state.memory.read(RV64_MEMORY_AS, byte_addr / 2) };
        u64::from(limbs[0])
            | (u64::from(limbs[1]) << 16)
            | (u64::from(limbs[2]) << 32)
            | (u64::from(limbs[3]) << 48)
    }

    #[cfg(feature = "rvr")]
    fn read_register(state: &VmState<GuestMemory>, index: usize) -> u64 {
        let limbs: [u16; 4] = unsafe {
            state.memory.read(
                RV64_REGISTER_AS,
                (index * RV64_REGISTER_NUM_LIMBS / 2) as u32,
            )
        };
        u64::from(limbs[0])
            | (u64::from(limbs[1]) << 16)
            | (u64::from(limbs[2]) << 32)
            | (u64::from(limbs[3]) << 48)
    }

    #[cfg(feature = "rvr")]
    fn final_preflight_timestamp(
        execution: &openvm_circuit::arch::rvr::RvrPreflightExecution,
    ) -> u32 {
        execution
            .transcript
            .program_log
            .last()
            .expect("preflight always appends a final program event")
            .timestamp
    }

    #[cfg(feature = "rvr")]
    fn hint_store_instruction(
        opcode: Rv64HintStoreOpcode,
        ptr_reg: usize,
        count_reg: usize,
    ) -> Instruction<F> {
        Instruction::from_usize(
            opcode.global_opcode(),
            [
                count_reg * RV64_REGISTER_NUM_LIMBS,
                ptr_reg * RV64_REGISTER_NUM_LIMBS,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        )
    }

    #[cfg(feature = "rvr")]
    fn reveal_instruction(
        opcode: Rv64LoadStoreOpcode,
        src_reg: usize,
        base_reg: usize,
        offset: i16,
    ) -> Instruction<F> {
        Instruction::from_usize(
            opcode.global_opcode(),
            [
                src_reg * RV64_REGISTER_NUM_LIMBS,
                base_reg * RV64_REGISTER_NUM_LIMBS,
                offset as u16 as usize,
                RV64_REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                usize::from(offset.is_negative()),
            ],
        )
    }

    #[cfg(feature = "rvr")]
    fn configure_reveal_state(
        mut state: VmState<GuestMemory>,
        registers: &[(usize, u64)],
        public_values: &[u8],
    ) -> VmState<GuestMemory> {
        for &(index, value) in registers {
            unsafe {
                state.memory.write_bytes(
                    RV64_REGISTER_AS,
                    (index * RV64_REGISTER_NUM_LIMBS) as u32,
                    value.to_le_bytes(),
                );
            }
        }
        let storage = state.memory.memory.mem[PUBLIC_VALUES_AS as usize].as_mut_slice();
        assert_eq!(storage.len(), public_values.len());
        storage.copy_from_slice(public_values);
        state
    }

    #[cfg(feature = "rvr")]
    fn execute_rvr_example(program_name: &str) {
        execute_rvr_example_with_input(program_name, vec![]);
    }

    #[cfg(feature = "rvr")]
    fn execute_rvr_example_with_input(program_name: &str, input: Vec<Vec<u8>>) {
        let config = test_rv64im_config();
        let elf =
            build_example_program_at_path(get_programs_dir!(), program_name, &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(input).unwrap();
    }

    #[test_case("fibonacci"; "fibonacci")]
    #[test_case("rvr_x0_shifts"; "x0_shifts")]
    #[test_case("rvr_embedded_text_data"; "embedded_text_data")]
    #[test_case("rvr_invalid_branch_fallthrough"; "invalid_branch_fallthrough")]
    #[cfg(feature = "rvr")]
    fn test_rvr_example_executes(program_name: &str) {
        execute_rvr_example(program_name);
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_preflight_logs_register_schedule() -> Result<()> {
        let instructions = [
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    0,
                    5,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    1,
                    0,
                ],
            ),
            Instruction::<F>::from_usize(
                BaseAluOpcode::ADD.global_opcode(),
                [
                    3 * RV64_REGISTER_NUM_LIMBS,
                    3 * RV64_REGISTER_NUM_LIMBS,
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_REGISTER_AS as usize,
                    1,
                    0,
                ],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let execution = instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 5),
        )?;

        let program = &execution.transcript.program_log;
        assert_eq!(
            program
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 3), (8, 6), (8, 6)]
        );

        let memory = &execution.transcript.memory_log;
        assert_eq!(memory.len(), 5);
        assert_eq!(
            memory
                .iter()
                .map(|event| event.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(memory[0].pointer, 0);
        assert_eq!(memory[1].pointer, 4);
        assert_eq!(memory[1].value, [5, 0, 0, 0]);
        assert_eq!(memory[2].pointer, 12);
        assert_eq!(memory[3].pointer, 0);
        assert_eq!(memory[4].pointer, 12);

        // x1's first event is a write. x3 is read before it is written, so
        // cold candidate filtering must not emit an initial-write entry for it.
        assert_eq!(execution.transcript.initial_write_log.len(), 1);
        assert_eq!(execution.transcript.initial_write_log[0].pointer, 4);
        assert_eq!(
            execution.transcript.initial_write_log[0].initial_value,
            [0; 4]
        );

        let capacity_error = match instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(0, 0),
        ) {
            Ok(_) => panic!("zero-capacity execution unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(capacity_error
            .to_string()
            .contains("execution returned error code: 2"));

        let allocation_error = match instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(0, usize::MAX),
        ) {
            Ok(_) => panic!("impossible capacity unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(allocation_error
            .to_string()
            .contains("failed to reserve preflight memory log"));
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_preflight_suspends_and_resumes_at_whole_blocks() -> Result<()> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let instructions = [
            Instruction::<F>::from_isize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                reg(1) as isize,
                reg(0) as isize,
                1,
                RV64_REGISTER_AS as isize,
                RV64_IMM_AS as isize,
            ),
            Instruction::<F>::from_isize(
                BranchEqualOpcode::BNE.global_opcode(),
                reg(1) as isize,
                reg(0) as isize,
                8,
                RV64_REGISTER_AS as isize,
                RV64_REGISTER_AS as isize,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 1, 0, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;

        let too_small = instance.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(1, 8),
        )?;
        assert_eq!(
            too_small.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 0,
                final_timestamp: 1,
            }
        );
        assert_eq!(
            too_small
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1)]
        );

        let first = instance.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 8),
        )?;
        assert_eq!(
            first.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 12,
                final_timestamp: 5,
            }
        );
        assert_eq!(
            first
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 3), (12, 5)]
        );
        assert_eq!(first.state.pc(), 12);

        let second = instance.execute_from_state_for(
            first.state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(1, 0),
        )?;
        assert_eq!(
            second.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Terminated
        );
        assert_eq!(
            second
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            // The resumed call resets its timestamp to one. The second entry is
            // the single final sentinel at the terminal PC.
            vec![(12, 1), (12, 1)]
        );

        let unbounded = instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 4),
        )?;
        let segmented_pcs = first
            .transcript
            .program_log
            .iter()
            .rev()
            .skip(1)
            .rev()
            .chain(second.transcript.program_log.iter().rev().skip(1).rev())
            .map(|event| event.pc)
            .collect::<Vec<_>>();
        assert_eq!(
            segmented_pcs,
            unbounded
                .transcript
                .program_log
                .iter()
                .rev()
                .skip(1)
                .rev()
                .map(|event| event.pc)
                .collect::<Vec<_>>()
        );
        assert_eq!(second.state.pc(), unbounded.state.pc());
        let x1_ptr = (reg(1) / 2) as u32;
        let segmented_x1: [u16; 4] = unsafe { second.state.memory.read(RV64_REGISTER_AS, x1_ptr) };
        let unbounded_x1: [u16; 4] =
            unsafe { unbounded.state.memory.read(RV64_REGISTER_AS, x1_ptr) };
        assert_eq!(segmented_x1, [1, 0, 0, 0]);
        assert_eq!(segmented_x1, unbounded_x1);

        let termination_required = match instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 8),
        ) {
            Ok(_) => panic!("termination-required preflight unexpectedly suspended successfully"),
            Err(error) => error,
        };
        assert!(termination_required
            .to_string()
            .contains("execution returned error code: 2"));

        let memory_error = match instance.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 3),
        ) {
            Ok(_) => panic!("mid-block memory exhaustion unexpectedly suspended"),
            Err(error) => error,
        };
        assert!(memory_error.to_string().contains("code 2"));

        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_checkpoint_preflight_matches_branch_suspension_and_resume() -> Result<()> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let instructions = [
            Instruction::<F>::from_isize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                reg(1) as isize,
                reg(0) as isize,
                1,
                RV64_REGISTER_AS as isize,
                RV64_IMM_AS as isize,
            ),
            Instruction::<F>::from_isize(
                BranchEqualOpcode::BNE.global_opcode(),
                reg(1) as isize,
                reg(0) as isize,
                8,
                RV64_REGISTER_AS as isize,
                RV64_REGISTER_AS as isize,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 1, 0, 0),
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 8, RV64_REGISTER_AS as usize, 0, 0],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 2, 0, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let full = executor.rvr_preflight_instance(&exe, None)?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;

        let exact_error = match checkpoint.execute_from_state_for_exact(
            checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(5, 0, 2),
        ) {
            Ok(_) => panic!("an early termination must not satisfy a metered segment boundary"),
            Err(error) => error,
        };
        assert!(exact_error
            .to_string()
            .contains("retired 4 instructions, expected 5"));

        let full_first = full.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 8),
        )?;
        let checkpoint_first = checkpoint.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(3, 0, 2),
        )?;
        assert_eq!(checkpoint_first.endpoint, full_first.endpoint);
        assert_eq!(checkpoint_first.state.pc(), full_first.state.pc());
        assert_eq!(
            checkpoint_first.to_state,
            openvm_circuit::arch::ExecutionState::new(
                full_first.state.pc(),
                final_preflight_timestamp(&full_first),
            )
        );
        assert_eq!(checkpoint_first.retired, 3);
        assert_eq!(checkpoint_first.transcript.checkpoints.len(), 1);
        let boundary = checkpoint_first.transcript.checkpoints[0];
        assert_eq!(
            (boundary.pc, boundary.timestamp, boundary.retired),
            (12, 5, 2)
        );
        assert_eq!(boundary.residual_cursor, 0);
        assert_eq!(boundary.regs[0], 1);
        assert!(checkpoint_first.transcript.residuals.is_empty());

        let full_second = full.execute_from_state_for(
            full_first.state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(1, 0),
        )?;
        let checkpoint_second = checkpoint.execute_from_state_for(
            checkpoint_first.state,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(1, 0, 2),
        )?;
        assert_eq!(checkpoint_second.endpoint, full_second.endpoint);
        assert_eq!(checkpoint_second.state.pc(), full_second.state.pc());
        assert_eq!(
            checkpoint_second.to_state,
            openvm_circuit::arch::ExecutionState::new(
                full_second.state.pc(),
                final_preflight_timestamp(&full_second),
            )
        );
        assert_eq!(read_register(&checkpoint_second.state, 1), 1);
        assert_eq!(
            read_register(&checkpoint_second.state, 1),
            read_register(&full_second.state, 1)
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_checkpoint_preflight_carries_dirty_memory_across_segments() -> Result<()> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let memory = |opcode: Rv64LoadStoreOpcode, value: usize, base: usize| {
            Instruction::<F>::from_usize(
                opcode.global_opcode(),
                [
                    reg(value),
                    reg(base),
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            )
        };
        let jump_to_next = || {
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
            )
        };
        let instructions = [
            memory(Rv64LoadStoreOpcode::STORED, 2, 1),
            jump_to_next(),
            memory(Rv64LoadStoreOpcode::LOADD, 3, 1),
            memory(Rv64LoadStoreOpcode::STORED, 0, 1),
            jump_to_next(),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;
        let address = PAGE_SIZE as u64 + 8;
        let value = 0x0123_4567_89ab_cdef;
        let initial = configure_hint_state(
            checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, address), (2, value)],
            &[],
        );
        let page_is_marked = |state: &VmState<GuestMemory>| {
            state.memory.memory.touched_pages[RV64_MEMORY_AS as usize]
                .touched_byte_ranges(2 * PAGE_SIZE)
                .iter()
                .any(|&(start, end)| start <= address as usize && (address as usize) < end)
        };
        assert!(!page_is_marked(&initial));

        let first = checkpoint.execute_from_state_for(
            initial,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(2, 0, 2),
        )?;
        assert_eq!(
            first.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 8,
                final_timestamp: 6,
            }
        );
        assert_eq!(read_main_word(&first.state, address as u32), value);
        assert!(page_is_marked(&first.state));
        assert!(
            !first.state.memory.memory.touched_pages[RV64_REGISTER_AS as usize]
                .touched_byte_ranges(RV64_REGISTER_NUM_LIMBS * 32 * 2)
                .is_empty()
        );

        let second = checkpoint.execute_from_state_for(
            first.state,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(3, 1, 2),
        )?;
        assert_eq!(read_register(&second.state, 3), value);
        assert_eq!(read_main_word(&second.state, address as u32), 0);
        assert!(page_is_marked(&second.state));
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_checkpoint_preflight_load_residuals_omit_x0() -> Result<()> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let load = |opcode: Rv64LoadStoreOpcode, rd: usize, offset: usize| {
            Instruction::<F>::from_usize(
                opcode.global_opcode(),
                [
                    reg(rd),
                    reg(1),
                    offset,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            )
        };
        let instructions = [
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    reg(1),
                    reg(0),
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    1,
                    0,
                ],
            ),
            load(Rv64LoadStoreOpcode::LOADD, 2, 0),
            load(Rv64LoadStoreOpcode::LOADD, 0, 8),
            load(Rv64LoadStoreOpcode::LOADW, 3, 16),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let full = executor.rvr_preflight_instance(&exe, None)?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;
        let loaded = 0x0123_4567_89ab_cdefu64;
        let x0_only = 0xfedc_ba98_7654_3210u64;
        let sign_extended = 0xffff_ffff_8000_0001u64;

        let mut full_initial = full.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let mut checkpoint_initial = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        for state in [&mut full_initial, &mut checkpoint_initial] {
            unsafe {
                state
                    .memory
                    .write_bytes(RV64_MEMORY_AS, 0, loaded.to_le_bytes());
                state
                    .memory
                    .write_bytes(RV64_MEMORY_AS, 8, x0_only.to_le_bytes());
                state
                    .memory
                    .write_bytes(RV64_MEMORY_AS, 16, (sign_extended as u32).to_le_bytes());
            }
        }

        let full_execution = full.execute_from_state(
            full_initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(instructions.len(), 16),
        )?;
        let checkpoint_execution = checkpoint.execute_from_state(
            checkpoint_initial,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(instructions.len(), 2, 2),
        )?;
        assert_eq!(checkpoint_execution.endpoint, full_execution.endpoint);
        assert_eq!(checkpoint_execution.state.pc(), full_execution.state.pc());
        assert_eq!(
            checkpoint_execution.to_state.timestamp,
            final_preflight_timestamp(&full_execution)
        );
        assert_eq!(
            checkpoint_execution.transcript.residuals,
            vec![loaded, sign_extended]
        );
        assert_eq!(read_register(&checkpoint_execution.state, 0), 0);
        assert_eq!(read_register(&checkpoint_execution.state, 2), loaded);
        assert_eq!(read_register(&checkpoint_execution.state, 3), sign_extended);
        assert_eq!(
            read_register(&checkpoint_execution.state, 2),
            read_register(&full_execution.state, 2)
        );
        assert_eq!(
            read_register(&checkpoint_execution.state, 3),
            read_register(&full_execution.state, 3)
        );

        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            openvm_riscv_circuit::Rv64ImCpuBuilder,
            test_rv64im_config(),
        )?;
        let mut metered_initial = vm.create_initial_state(&exe, Vec::<Vec<u8>>::new());
        unsafe {
            metered_initial
                .memory
                .write_bytes(RV64_MEMORY_AS, 0, loaded.to_le_bytes());
            metered_initial
                .memory
                .write_bytes(RV64_MEMORY_AS, 8, x0_only.to_le_bytes());
            metered_initial.memory.write_bytes(
                RV64_MEMORY_AS,
                16,
                (sign_extended as u32).to_le_bytes(),
            );
        }
        let metered_ctx = vm.build_metered_ctx(&exe);
        let (segments, _) = vm
            .metered_instance(&exe)?
            .execute_metered_from_state(metered_initial, metered_ctx)?;
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].num_insns, instructions.len() as u64);
        assert_eq!(segments[0].num_checkpoint_residuals, 2);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_checkpoint_preflight_hint_residual_order_and_memory() -> Result<()> {
        let instructions = [
            hint_store_instruction(Rv64HintStoreOpcode::HINT_STORED, 1, 0),
            hint_store_instruction(Rv64HintStoreOpcode::HINT_BUFFER, 2, 3),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let full = executor.rvr_preflight_instance(&exe, None)?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;
        let hint_words = [
            0x0123_4567_89ab_cdef,
            0x1111_2222_3333_4444,
            0xaaaa_bbbb_cccc_dddd,
        ];
        let registers = [(1, 32), (2, 64), (3, 2)];
        let full_initial = configure_hint_state(
            full.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &hint_words,
        );
        let checkpoint_initial = configure_hint_state(
            checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &hint_words,
        );

        let full_execution = full.execute_from_state(
            full_initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(instructions.len(), 8),
        )?;
        let checkpoint_execution = checkpoint.execute_from_state(
            checkpoint_initial,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(
                instructions.len(),
                hint_words.len(),
                2,
            ),
        )?;
        assert_eq!(checkpoint_execution.endpoint, full_execution.endpoint);
        assert_eq!(checkpoint_execution.state.pc(), full_execution.state.pc());
        assert_eq!(
            checkpoint_execution.to_state.timestamp,
            final_preflight_timestamp(&full_execution)
        );
        assert_eq!(checkpoint_execution.transcript.residuals, hint_words);
        assert_eq!(
            checkpoint_execution.state.streams.hint_stream.remaining(),
            0
        );
        assert_eq!(full_execution.state.streams.hint_stream.remaining(), 0);
        for (address, expected) in [
            (32, hint_words[0]),
            (64, hint_words[1]),
            (72, hint_words[2]),
        ] {
            assert_eq!(
                read_main_word(&checkpoint_execution.state, address),
                expected
            );
            assert_eq!(
                read_main_word(&checkpoint_execution.state, address),
                read_main_word(&full_execution.state, address)
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_checkpoint_preflight_reveal_matches_clock_and_public_values() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(16);
        let instructions = [
            reveal_instruction(Rv64LoadStoreOpcode::STOREB, 1, 2, 0),
            // A word at byte address 7 crosses an eight-byte memory block.
            reveal_instruction(Rv64LoadStoreOpcode::STOREW, 3, 4, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(config)?;
        let full = executor.rvr_preflight_instance(&exe, None)?;
        let checkpoint = executor.checkpoint_preflight_instance(&exe, None)?;
        let registers = [(1, 0xa5), (2, 2), (3, 0x1122_3344), (4, 7)];
        let initial_public_values = (0u8..16).collect::<Vec<_>>();
        let full_initial = configure_reveal_state(
            full.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &initial_public_values,
        );
        let checkpoint_initial = configure_reveal_state(
            checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &initial_public_values,
        );

        let full_execution = full.execute_from_state(
            full_initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(instructions.len(), 7),
        )?;
        let checkpoint_execution = checkpoint.execute_from_state(
            checkpoint_initial,
            openvm_circuit::arch::rvr::RvrCheckpointPreflightLimits::new(instructions.len(), 0, 2),
        )?;

        assert_eq!(checkpoint_execution.endpoint, full_execution.endpoint);
        assert_eq!(checkpoint_execution.state.pc(), full_execution.state.pc());
        assert_eq!(
            checkpoint_execution.to_state.timestamp,
            final_preflight_timestamp(&full_execution)
        );
        // Two register reads plus one memory slot for STOREB, followed by two
        // register reads plus two slots for the crossing STOREW.
        assert_eq!(checkpoint_execution.to_state.timestamp, 8);
        assert!(checkpoint_execution.transcript.residuals.is_empty());

        let mut expected = initial_public_values;
        expected[2] = 0xa5;
        expected[7..11].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        assert_eq!(
            extract_public_values(16, &checkpoint_execution.state.memory.memory),
            expected
        );
        assert_eq!(
            extract_public_values(16, &checkpoint_execution.state.memory.memory),
            extract_public_values(16, &full_execution.state.memory.memory)
        );
        assert_eq!(
            checkpoint_execution.state.memory.memory.touched_pages[PUBLIC_VALUES_AS as usize]
                .touched_byte_ranges(16),
            vec![(0, 16)]
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_builtin_phantoms_have_one_slot_and_no_memory_events() -> Result<()> {
        let instructions = [
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                0,
                0,
                SysPhantom::Nop as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                0,
                0,
                SysPhantom::CtStart as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                0,
                0,
                SysPhantom::CtEnd as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let preflight = executor.rvr_preflight_instance(&exe, None)?;

        let suspended = preflight.execute_for(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 0),
        )?;
        assert_eq!(
            suspended.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 0,
                final_timestamp: 1,
            }
        );
        assert_eq!(
            suspended
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1)]
        );
        assert!(suspended.transcript.memory_log.is_empty());
        assert!(suspended.transcript.initial_write_log.is_empty());

        let execution = preflight.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(4, 0),
        )?;
        assert_eq!(
            execution.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Terminated
        );
        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 2), (8, 3), (12, 4), (12, 4)]
        );
        assert!(execution.transcript.memory_log.is_empty());
        assert!(execution.transcript.initial_write_log.is_empty());
        assert_eq!(execution.state.pc(), 12);

        let pure = executor.rvr_instance(&exe, None)?;
        let pure_state = pure.execute(Vec::<Vec<u8>>::new())?;
        assert_eq!(pure_state.pc(), 12);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_callback_phantoms_are_serial_without_memory_events() -> Result<()> {
        let exe = callback_phantom_exe();
        let executor = VmExecutor::new(test_rv64im_config())?;
        let preflight = executor.rvr_preflight_instance(&exe, None)?;
        let inputs = vec![b"first".to_vec(), b"second".to_vec()];
        let initial_state =
            configure_callback_state(preflight.create_initial_vm_state(inputs.clone()));

        let suspended = preflight.execute_from_state_for(
            initial_state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 0),
        )?;
        assert_eq!(
            suspended.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 8,
                final_timestamp: 3,
            }
        );
        assert_eq!(
            suspended
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 2), (8, 3)]
        );
        assert!(suspended.transcript.memory_log.is_empty());
        assert!(suspended.transcript.initial_write_log.is_empty());
        assert_eq!(
            suspended
                .state
                .streams
                .input_stream
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![b"second".to_vec()]
        );
        let mut suspended_hint = suspended.state.streams.hint_stream.clone();
        let mut input_hint = vec![0; suspended_hint.remaining()];
        suspended_hint.copy_to_slice(&mut input_hint);
        let mut expected_input_hint = (b"first".len() as u64).to_le_bytes().to_vec();
        expected_input_hint.extend_from_slice(b"first");
        expected_input_hint.resize(16, 0);
        assert_eq!(input_hint, expected_input_hint);
        let mut suspended_rng = suspended.state.rng.clone();
        let mut initial_rng = StdRng::seed_from_u64(0);
        assert_eq!(suspended_rng.random::<u64>(), initial_rng.random::<u64>());

        let mut execution = preflight.execute_from_state_for(
            suspended.state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 0),
        )?;
        assert_eq!(
            execution.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Terminated
        );
        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(8, 1), (12, 2), (16, 3), (16, 3)]
        );
        assert!(execution.transcript.memory_log.is_empty());
        assert!(execution.transcript.initial_write_log.is_empty());
        assert_eq!(
            execution
                .state
                .streams
                .input_stream
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![b"second".to_vec()]
        );

        let mut expected_rng = StdRng::seed_from_u64(0);
        let expected_hint = (0..8)
            .map(|_| expected_rng.random::<u8>())
            .collect::<Vec<_>>();
        let expected_next_random = expected_rng.random::<u64>();
        assert_eq!(take_hint(&mut execution.state), expected_hint);
        assert_eq!(execution.state.rng.random::<u64>(), expected_next_random);

        let pure = executor.rvr_instance(&exe, None)?;
        let pure_initial = configure_callback_state(pure.create_initial_vm_state(inputs));
        let mut pure_state = pure.execute_from_state(pure_initial)?;
        assert_eq!(pure_state.pc(), 16);
        assert_eq!(
            pure_state
                .streams
                .input_stream
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![b"second".to_vec()]
        );
        assert_eq!(take_hint(&mut pure_state), expected_hint);
        assert_eq!(pure_state.rng.random::<u64>(), expected_next_random);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_store_preflight_logs_all_word_counts_and_matches_pure() -> Result<()> {
        let instructions = [
            hint_store_instruction(Rv64HintStoreOpcode::HINT_BUFFER, 1, 2),
            hint_store_instruction(Rv64HintStoreOpcode::HINT_BUFFER, 3, 4),
            hint_store_instruction(Rv64HintStoreOpcode::HINT_BUFFER, 5, 6),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let word_counts = [1usize, 3, MAX_HINT_BUFFER_DWORDS];
        let destinations = [0u64, 16, 64];
        let registers = [
            (1, destinations[0]),
            (2, word_counts[0] as u64),
            (3, destinations[1]),
            (4, word_counts[1] as u64),
            (5, destinations[2]),
            (6, word_counts[2] as u64),
        ];
        let hint_words = (0..word_counts.iter().sum())
            .map(|index| 0x1000_0000_0000_0000u64 + index as u64)
            .collect::<Vec<_>>();
        let initial = configure_hint_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &hint_words,
        );
        let total_memory_events = 2 * word_counts.len() + hint_words.len();
        let execution = instance.execute_from_state(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(
                instructions.len(),
                total_memory_events,
            ),
        )?;

        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 4), (8, 13), (12, 3082), (12, 3082)]
        );
        assert_eq!(execution.transcript.memory_log.len(), total_memory_events);
        assert_eq!(
            execution
                .transcript
                .memory_log
                .iter()
                .map(|event| event.timestamp)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
                .into_iter()
                .chain([6, 9, 12])
                .chain([13, 14])
                .chain((0..MAX_HINT_BUFFER_DWORDS).map(|index| 15 + 3 * index as u32))
                .collect::<Vec<_>>()
        );

        const WRITE_BIT: u32 = 1 << 31;
        let writes = execution
            .transcript
            .memory_log
            .iter()
            .filter(|event| event.address_space_and_kind & WRITE_BIT != 0)
            .collect::<Vec<_>>();
        assert_eq!(writes.len(), hint_words.len());
        assert_eq!(
            writes
                .iter()
                .map(|event| event.timestamp)
                .take(4)
                .collect::<Vec<_>>(),
            vec![3, 6, 9, 12]
        );
        assert_eq!(writes[4].timestamp, 15);
        assert_eq!(writes.last().unwrap().timestamp, 3081);
        assert_eq!(
            writes.iter().map(|event| event.pointer).collect::<Vec<_>>(),
            destinations
                .into_iter()
                .zip(word_counts)
                .flat_map(|(dest, count)| {
                    (0..count).map(move |index| (dest as u32 / 2) + 4 * index as u32)
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            execution.transcript.initial_write_log.len(),
            hint_words.len()
        );
        assert!(execution
            .transcript
            .initial_write_log
            .iter()
            .all(|event| event.address_space == RV64_MEMORY_AS && event.initial_value == [0; 4]));
        assert_eq!(execution.state.streams.hint_stream.remaining(), 0);

        let pure = executor.rvr_instance(&exe, None)?;
        let pure_initial = configure_hint_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &registers,
            &hint_words,
        );
        let pure_state = pure.execute_from_state(pure_initial)?;
        assert_eq!(pure_state.streams.hint_stream.remaining(), 0);

        let mut hint_index = 0;
        for (&dest, &count) in destinations.iter().zip(&word_counts) {
            for word_index in 0..count {
                let addr = dest as u32 + (word_index * 8) as u32;
                assert_eq!(
                    read_main_word(&execution.state, addr),
                    hint_words[hint_index]
                );
                assert_eq!(read_main_word(&pure_state, addr), hint_words[hint_index]);
                hint_index += 1;
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_store_repeated_write_has_one_initial_seed() -> Result<()> {
        let instructions = [
            hint_store_instruction(Rv64HintStoreOpcode::HINT_STORED, 1, 0),
            hint_store_instruction(Rv64HintStoreOpcode::HINT_STORED, 1, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let hint_words = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let initial = configure_hint_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 32)],
            &hint_words,
        );
        let execution = instance.execute_from_state(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 4),
        )?;

        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 4), (8, 7), (8, 7)]
        );
        assert_eq!(
            execution
                .transcript
                .memory_log
                .iter()
                .map(|event| (event.timestamp, event.pointer, event.value))
                .collect::<Vec<_>>(),
            vec![
                (1, 4, [32, 0, 0, 0]),
                (3, 16, [0xcdef, 0x89ab, 0x4567, 0x0123]),
                (4, 4, [32, 0, 0, 0]),
                (6, 16, [0x3210, 0x7654, 0xba98, 0xfedc]),
            ]
        );
        assert_eq!(execution.transcript.initial_write_log.len(), 1);
        assert_eq!(execution.transcript.initial_write_log[0].pointer, 16);
        assert_eq!(
            execution.transcript.initial_write_log[0].initial_value,
            [0; 4]
        );
        assert_eq!(read_main_word(&execution.state, 32), hint_words[1]);
        assert_eq!(execution.state.streams.hint_stream.remaining(), 0);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_store_suspends_only_after_consuming_the_whole_instruction() -> Result<()> {
        let instructions = [
            hint_store_instruction(Rv64HintStoreOpcode::HINT_STORED, 1, 0),
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
            ),
            hint_store_instruction(Rv64HintStoreOpcode::HINT_STORED, 1, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let hint_words = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let initial = configure_hint_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 32)],
            &hint_words,
        );

        let first = instance.execute_from_state_for(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 2),
        )?;
        assert_eq!(
            first.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 8,
                final_timestamp: 5,
            }
        );
        assert_eq!(
            first
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 4), (8, 5)]
        );
        assert_eq!(read_main_word(&first.state, 32), hint_words[0]);
        assert_eq!(first.state.streams.hint_stream.remaining(), 8);

        let second = instance.execute_from_state(
            first.state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 2),
        )?;
        assert_eq!(
            second.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Terminated
        );
        assert_eq!(
            second
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(8, 1), (12, 4), (12, 4)]
        );
        assert_eq!(read_main_word(&second.state, 32), hint_words[1]);
        assert_eq!(second.state.streams.hint_stream.remaining(), 0);
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_hint_store_uses_the_memory_block_alignment_contract() -> Result<()> {
        let instructions = [
            hint_store_instruction(Rv64HintStoreOpcode::HINT_BUFFER, 1, 2),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let initial = configure_hint_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 0), (2, 3)],
            &[1, 2, 3],
        );
        let error = match instance.execute_from_state(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 4),
        ) {
            Ok(_) => panic!("undersized whole-operation reservation unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("code 2"), "{error}");

        let pure = executor.rvr_instance(&exe, None)?;
        let unaligned = configure_hint_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 1), (2, 1)],
            &[0xdead_beef],
        );
        let error = match pure.execute_from_state(unaligned) {
            Ok(_) => panic!("unaligned hint destination unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("error code: 3"));

        let interpreter = executor.interpreter_instance(&exe)?;
        let odd = configure_hint_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 1), (2, 1)],
            &[0x0123_4567_89ab_cdef],
        );
        let error = match interpreter.execute_from_state(odd) {
            Ok(_) => panic!("odd hint destination unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("eight-byte aligned"), "{error}");

        // A proof-visible memory event is one fixed eight-byte block. Both
        // executors reject every two-byte-aligned pointer that is not block-aligned.
        let hint = 0x0123_4567_89ab_cdefu64;
        for pointer in [2, 4, 6] {
            let rvr_unaligned = configure_hint_state(
                pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
                &[(1, pointer), (2, 1)],
                &[hint],
            );
            let error = match pure.execute_from_state(rvr_unaligned) {
                Ok(_) => panic!("misaligned hint destination {pointer} unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("error code: 3"));
            let interpreter_unaligned = configure_hint_state(
                pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
                &[(1, pointer), (2, 1)],
                &[hint],
            );
            let error = match interpreter.execute_from_state(interpreter_unaligned) {
                Ok(_) => panic!("misaligned hint destination {pointer} unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("eight-byte aligned"), "{error}");
        }

        let rvr_initial = configure_hint_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 8), (2, 1)],
            &[hint],
        );
        let interpreter_initial = configure_hint_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 8), (2, 1)],
            &[hint],
        );
        let rvr_state = pure.execute_from_state(rvr_initial)?;
        let interpreter_state = interpreter.execute_from_state(interpreter_initial)?;
        for state in [&rvr_state, &interpreter_state] {
            let bytes: [u8; 8] = unsafe {
                state
                    .memory
                    .memory
                    .get_memory()
                    .get_unchecked(RV64_MEMORY_AS as usize)
                    .read(8)
            };
            assert_eq!(bytes, hint.to_le_bytes());
            assert_eq!(state.streams.hint_stream.remaining(), 0);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_reveal_preflight_matches_interpreter_logs_and_touched_pages() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(2 * PAGE_SIZE);
        let instructions = [
            reveal_instruction(Rv64LoadStoreOpcode::STORED, 1, 2, 0),
            reveal_instruction(Rv64LoadStoreOpcode::STOREW, 3, 4, -4),
            reveal_instruction(Rv64LoadStoreOpcode::STOREH, 5, 6, -2),
            reveal_instruction(Rv64LoadStoreOpcode::STOREB, 7, 8, 0),
            reveal_instruction(Rv64LoadStoreOpcode::STOREW, 9, 10, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let registers = [
            (1, 0x1122_3344_5566_7788),
            (2, (PAGE_SIZE - 4) as u64),
            (3, 0xa1b2_c3d4),
            (4, 12),
            (5, 0xe5f6),
            (6, 9),
            (7, 0xaa),
            (8, 15),
            (9, 0xdead_beef),
            (10, 8),
        ];
        let initial_public_values = (0..2 * PAGE_SIZE)
            .map(|index| index as u8)
            .collect::<Vec<_>>();
        let mut expected_public_values = initial_public_values.clone();
        expected_public_values[PAGE_SIZE - 4..PAGE_SIZE + 4]
            .copy_from_slice(&registers[0].1.to_le_bytes());
        expected_public_values[8..12].copy_from_slice(&(registers[2].1 as u32).to_le_bytes());
        expected_public_values[7..9].copy_from_slice(&(registers[4].1 as u16).to_le_bytes());
        expected_public_values[15] = registers[6].1 as u8;
        expected_public_values[8..12].copy_from_slice(&(registers[8].1 as u32).to_le_bytes());

        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            openvm_riscv_circuit::Rv64ImCpuBuilder,
            config,
        )?;
        let initial = configure_reveal_state(
            vm.create_initial_state(&exe, Vec::<Vec<u8>>::new()),
            &registers,
            &initial_public_values,
        );
        assert!(
            initial.memory.memory.touched_pages[PUBLIC_VALUES_AS as usize]
                .touched_byte_ranges(initial_public_values.len())
                .is_empty()
        );

        let metered_ctx = vm.build_metered_ctx(&exe);
        let (segments, metered_state) = vm
            .metered_instance(&exe)?
            .execute_metered_from_state(initial.clone(), metered_ctx)?;
        assert_eq!(segments.len(), 1);
        assert_eq!(
            extract_public_values(2 * PAGE_SIZE, &metered_state.memory.memory),
            expected_public_values
        );
        let segment = &segments[0];

        let mut interpreter = vm.preflight_interpreter(&exe)?;
        let PreflightExecutionOutput {
            system_records,
            to_state: interpreter_state,
            ..
        } = vm.execute_preflight_for(
            &mut interpreter,
            initial.clone(),
            instructions.len() as u64,
            &segment.trace_heights,
        )?;
        let rvr = vm.executor().rvr_preflight_instance(&exe, None)?;
        let execution = rvr.execute_from_state(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(instructions.len(), 17),
        )?;

        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![
                (0, 1),
                (4, 5),
                (8, 9),
                (12, 13),
                (16, 16),
                (20, 20),
                (20, 20),
            ]
        );
        assert_eq!(
            execution
                .transcript
                .memory_log
                .iter()
                .map(|event| event.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        let u64_cells = |value: u64| {
            [
                value as u16,
                (value >> 16) as u16,
                (value >> 32) as u16,
                (value >> 48) as u16,
            ]
        };
        assert_eq!(
            execution
                .transcript
                .memory_log
                .iter()
                .filter(|event| event.address_space() == RV64_REGISTER_AS)
                .map(|event| (event.timestamp, event.pointer, event.value))
                .collect::<Vec<_>>(),
            vec![
                (1, 8, u64_cells((PAGE_SIZE - 4) as u64)),
                (2, 4, u64_cells(0x1122_3344_5566_7788)),
                (5, 16, u64_cells(12)),
                (6, 12, u64_cells(0xa1b2_c3d4)),
                (9, 24, u64_cells(9)),
                (10, 20, u64_cells(0xe5f6)),
                (13, 32, u64_cells(15)),
                (14, 28, u64_cells(0xaa)),
                (16, 40, u64_cells(8)),
                (17, 36, u64_cells(0xdead_beef)),
            ]
        );
        assert_eq!(
            execution
                .transcript
                .memory_log
                .iter()
                .filter(|event| event.address_space() == PUBLIC_VALUES_AS)
                .map(|event| (event.timestamp, event.pointer, event.value))
                .collect::<Vec<_>>(),
            vec![
                (
                    3,
                    ((PAGE_SIZE - 8) / 2) as u32,
                    [0xf9f8, 0xfbfa, 0x7788, 0x5566],
                ),
                (4, (PAGE_SIZE / 2) as u32, [0x3344, 0x1122, 0x0504, 0x0706],),
                (7, 4, [0xc3d4, 0xa1b2, 0x0d0c, 0x0f0e]),
                (11, 0, [0x0100, 0x0302, 0x0504, 0xf606]),
                (12, 4, [0xc3e5, 0xa1b2, 0x0d0c, 0x0f0e]),
                (15, 4, [0xc3e5, 0xa1b2, 0x0d0c, 0xaa0e]),
                (18, 4, [0xbeef, 0xdead, 0x0d0c, 0xaa0e]),
            ]
        );
        assert_eq!(
            execution
                .transcript
                .initial_write_log
                .iter()
                .filter(|event| event.address_space == PUBLIC_VALUES_AS)
                .map(|event| event.pointer)
                .collect::<Vec<_>>(),
            vec![((PAGE_SIZE - 8) / 2) as u32, (PAGE_SIZE / 2) as u32, 4, 0]
        );
        for seed in execution
            .transcript
            .initial_write_log
            .iter()
            .filter(|event| event.address_space == PUBLIC_VALUES_AS)
        {
            let byte_start = seed.pointer as usize * 2;
            let expected = std::array::from_fn(|index| {
                u16::from_le_bytes([
                    initial_public_values[byte_start + 2 * index],
                    initial_public_values[byte_start + 2 * index + 1],
                ])
            });
            assert_eq!(seed.initial_value, expected);
        }

        assert_eq!(system_records.to_state.timestamp, 20);
        assert_eq!(
            execution.transcript.program_log.last().unwrap().timestamp,
            system_records.to_state.timestamp
        );
        assert_eq!(
            extract_public_values(2 * PAGE_SIZE, &execution.state.memory.memory),
            expected_public_values
        );
        assert_eq!(
            extract_public_values(2 * PAGE_SIZE, &interpreter_state.memory.memory),
            expected_public_values
        );

        let rvr_as3_last_events = execution
            .transcript
            .memory_log
            .iter()
            .filter(|event| event.address_space() == PUBLIC_VALUES_AS)
            .fold(std::collections::BTreeMap::new(), |mut events, event| {
                events.insert(event.pointer, (event.timestamp, event.value));
                events
            });
        let interpreter_as3 = system_records
            .touched_memory
            .iter()
            .filter(|block| block.address_space == PUBLIC_VALUES_AS)
            .map(|block| {
                (
                    block.ptr,
                    (
                        block.timestamp,
                        block.values.map(|value| value.as_canonical_u32() as u16),
                    ),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(rvr_as3_last_events, interpreter_as3);

        let touched_ranges = execution.state.memory.memory.touched_pages[PUBLIC_VALUES_AS as usize]
            .touched_byte_ranges(2 * PAGE_SIZE);
        assert_eq!(touched_ranges, vec![(0, 2 * PAGE_SIZE)]);
        assert_eq!(
            touched_ranges,
            interpreter_state.memory.memory.touched_pages[PUBLIC_VALUES_AS as usize]
                .touched_byte_ranges(2 * PAGE_SIZE)
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_reveal_preflight_suspends_after_committed_store() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(PAGE_SIZE);
        let instructions = [
            reveal_instruction(Rv64LoadStoreOpcode::STOREW, 1, 2, 0),
            Instruction::<F>::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
            ),
            reveal_instruction(Rv64LoadStoreOpcode::STOREB, 3, 4, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(config)?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let initial = configure_reveal_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 0xaabb_ccdd), (2, 8), (3, 0xee), (4, 9)],
            &vec![0; PAGE_SIZE],
        );

        let first = instance.execute_from_state_for(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 3),
        )?;
        assert_eq!(
            first.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Suspended {
                resume_pc: 8,
                final_timestamp: 6,
            }
        );
        assert_eq!(
            first
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 5), (8, 6)]
        );
        assert_eq!(
            first
                .transcript
                .memory_log
                .iter()
                .filter(|event| event.address_space() == PUBLIC_VALUES_AS)
                .map(|event| (event.timestamp, event.pointer, event.value))
                .collect::<Vec<_>>(),
            vec![(3, 4, [0xccdd, 0xaabb, 0, 0])]
        );
        assert_eq!(
            &extract_public_values(PAGE_SIZE, &first.state.memory.memory)[8..12],
            &0xaabb_ccddu32.to_le_bytes()
        );

        let second = instance.execute_from_state_for(
            first.state,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 3),
        )?;
        assert_eq!(
            second.endpoint,
            openvm_circuit::arch::rvr::RvrPreflightEndpoint::Terminated
        );
        assert_eq!(
            second
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(8, 1), (12, 4), (12, 4)]
        );
        assert_eq!(
            second
                .transcript
                .initial_write_log
                .iter()
                .filter(|event| event.address_space == PUBLIC_VALUES_AS)
                .map(|event| (event.pointer, event.initial_value))
                .collect::<Vec<_>>(),
            vec![(4, [0xccdd, 0xaabb, 0, 0])]
        );
        assert_eq!(
            &extract_public_values(PAGE_SIZE, &second.state.memory.memory)[8..12],
            &[0xdd, 0xee, 0xbb, 0xaa]
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_reveal_preflight_fails_before_commit() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(16);
        let capacity_exe = VmExe::from(Program::from_instructions(&[
            reveal_instruction(Rv64LoadStoreOpcode::STORED, 1, 2, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ]));
        let executor = VmExecutor::new(config.clone())?;
        let instance = executor.rvr_preflight_instance(&capacity_exe, None)?;
        let initial = configure_reveal_state(
            instance.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, u64::MAX), (2, 4)],
            &[0; 16],
        );
        let error = match instance.execute_from_state(
            initial,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 3),
        ) {
            Ok(_) => panic!("undersized REVEAL reservation unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("code 2"), "{error}");

        // The effective address wraps to zero, but the non-u32 base still
        // fails closed in both execution modes.
        let address_exe = VmExe::from(Program::from_instructions(&[
            reveal_instruction(Rv64LoadStoreOpcode::STOREB, 1, 2, 1),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ]));
        let preflight = executor.rvr_preflight_instance(&address_exe, None)?;
        let invalid = configure_reveal_state(
            preflight.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 0xff), (2, u64::MAX)],
            &[0; 16],
        );
        let error = match preflight.execute_from_state(
            invalid,
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 3),
        ) {
            Ok(_) => panic!("wrapped REVEAL address unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("error code: 3"), "{error}");

        let pure = executor.rvr_instance(&address_exe, None)?;
        let invalid = configure_reveal_state(
            pure.create_initial_vm_state(Vec::<Vec<u8>>::new()),
            &[(1, 0xff), (2, u64::MAX)],
            &[0; 16],
        );
        let error = match pure.execute_from_state(invalid) {
            Ok(_) => panic!("wrapped REVEAL address unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("error code: 3"), "{error}");
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_input_exhaustion_traps() -> Result<()> {
        let instructions = [
            Instruction::<F>::from_isize(
                SystemOpcode::PHANTOM.global_opcode(),
                0,
                0,
                Rv64Phantom::HintInput as isize,
                0,
                0,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let preflight = executor.rvr_preflight_instance(&exe, None)?;
        let preflight_error = match preflight.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(2, 0),
        ) {
            Ok(_) => panic!("empty HINT_INPUT unexpectedly succeeded in preflight"),
            Err(error) => error,
        };
        assert!(preflight_error
            .to_string()
            .contains("execution returned error code: 3"));

        let pure = executor.rvr_instance(&exe, None)?;
        let pure_error = match pure.execute(Vec::<Vec<u8>>::new()) {
            Ok(_) => panic!("empty HINT_INPUT unexpectedly succeeded in pure execution"),
            Err(error) => error,
        };
        assert!(pure_error
            .to_string()
            .contains("execution returned error code: 3"));
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_preflight_rejects_timestamp_outside_proof_domain() -> Result<()> {
        let instructions = [
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    0,
                    1,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    1,
                    0,
                ],
            ),
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    2 * RV64_REGISTER_NUM_LIMBS,
                    0,
                    2,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    1,
                    0,
                ],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let mut config = test_rv64im_config();
        config.rv64i.system.memory_config.timestamp_max_bits = 2;
        let executor = VmExecutor::new(config)?;
        let instance = executor.rvr_preflight_instance(&exe, None)?;
        let error = match instance.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 4),
        ) {
            Ok(_) => panic!("out-of-domain preflight unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("outside the configured 2-bit domain"));
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_x0_schedule_does_not_change_jalr_cfg() -> Result<()> {
        let instructions = [
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    0,
                    0,
                    8,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    0,
                    0,
                ],
            ),
            Instruction::<F>::from_usize(
                Rv64JalrOpcode::JALR.global_opcode(),
                [
                    0,
                    0,
                    12,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    0,
                    0,
                ],
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 1, 0, 0),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;

        let pure = executor.instance(&exe)?.execute(Vec::<Vec<u8>>::new())?;
        assert_eq!(pure.pc(), 12);

        let preflight = executor.rvr_preflight_instance(&exe, None)?.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(3, 2),
        )?;
        assert_eq!(preflight.state.pc(), 12);
        assert_eq!(
            preflight
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![(0, 1), (4, 3), (12, 5), (12, 5)]
        );
        assert_eq!(
            preflight
                .transcript
                .memory_log
                .iter()
                .map(|event| (event.timestamp, event.pointer, event.value))
                .collect::<Vec<_>>(),
            vec![(1, 0, [0; 4]), (3, 0, [0; 4])]
        );
        Ok(())
    }

    #[test]
    #[ignore = "manual executor benchmark; builds native artifacts"]
    #[cfg(all(feature = "rvr", not(feature = "cuda")))]
    fn benchmark_rvr_preflight_against_interpreter() -> Result<()> {
        const REPETITIONS: usize = 7;
        const LOOP_COUNT: u64 = 1 << 18;

        let config = test_rv64im_config();
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let addi = |rd: usize, rs1: usize, immediate: isize| {
            Instruction::<F>::from_isize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                reg(rd) as isize,
                reg(rs1) as isize,
                immediate,
                RV64_REGISTER_AS as isize,
                RV64_IMM_AS as isize,
            )
        };
        let instructions = [
            addi(1, 0, 0),
            addi(1, 1, 1),
            Instruction::<F>::from_isize(
                BranchEqualOpcode::BNE.global_opcode(),
                reg(1) as isize,
                reg(2) as isize,
                -4,
                RV64_REGISTER_AS as isize,
                RV64_REGISTER_AS as isize,
            ),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let init_memory = LOOP_COUNT
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(2) + offset) as u32), byte))
            .collect();
        let exe =
            VmExe::new(Program::from_instructions(&instructions)).with_init_memory(init_memory);
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            openvm_riscv_circuit::Rv64ImCpuBuilder,
            config,
        )?;

        let metered = vm.metered_instance(&exe)?;
        let (segments, _) =
            metered.execute_metered(Vec::<Vec<u8>>::new(), vm.build_metered_ctx(&exe))?;
        assert_eq!(segments.len(), 1, "benchmark input must fit one segment");
        let segment = &segments[0];
        let max_instructions = usize::try_from(segment.num_insns)?;
        let max_memory_events = max_instructions
            .checked_mul(4)
            .ok_or_else(|| eyre::eyre!("benchmark memory-event capacity overflow"))?;

        let pure = vm.get_rvr_instance(&exe)?;
        let rvr = vm.executor().rvr_preflight_instance(&exe, None)?;
        if let Some(path) = std::env::var_os("RVR_PREFLIGHT_BENCH_SOURCES") {
            rvr.save_generated_sources(std::path::Path::new(&path))?;
        }
        let mut interpreter = vm.preflight_interpreter(&exe)?;
        let limits =
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(max_instructions, max_memory_events);
        let mut interpreter_times = Vec::with_capacity(REPETITIONS);
        let mut pure_times = Vec::with_capacity(REPETITIONS);
        let mut metered_times = Vec::with_capacity(REPETITIONS);
        let mut rvr_times = Vec::with_capacity(REPETITIONS);
        let mut transcript_bytes = 0usize;
        let mut reusable_transcript = None;

        for _ in 0..REPETITIONS {
            let pure_state = pure.create_initial_vm_state(Vec::<Vec<u8>>::new());
            let started = Instant::now();
            let pure_state = pure.execute_from_state(pure_state)?;
            pure_times.push(started.elapsed());

            let metered_state = metered.create_initial_vm_state(Vec::<Vec<u8>>::new());
            let metered_ctx = vm.build_metered_ctx(&exe);
            let started = Instant::now();
            let (metered_segments, metered_state) =
                metered.execute_metered_from_state(metered_state, metered_ctx)?;
            metered_times.push(started.elapsed());
            assert_eq!(metered_segments.len(), 1);

            let interpreter_state = vm.create_initial_state(&exe, Vec::<Vec<u8>>::new());
            let started = Instant::now();
            let PreflightExecutionOutput {
                system_records,
                to_state,
                ..
            } = vm.execute_preflight_for(
                &mut interpreter,
                interpreter_state,
                segment.num_insns,
                &segment.trace_heights,
            )?;
            interpreter_times.push(started.elapsed());

            let rvr_state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
            let started = Instant::now();
            let execution = match reusable_transcript.take() {
                Some(transcript) => {
                    rvr.execute_from_state_reusing(rvr_state, limits, transcript)?
                }
                None => rvr.execute_from_state(rvr_state, limits)?,
            };
            rvr_times.push(started.elapsed());

            assert_eq!(pure_state.pc(), to_state.pc());
            assert_eq!(metered_state.pc(), to_state.pc());
            assert_eq!(execution.state.pc(), to_state.pc());
            assert_eq!(
                execution.transcript.program_log.last().unwrap().timestamp,
                system_records.to_state.timestamp
            );
            let mut frequencies = vec![0u32; instructions.len()];
            for event in execution.transcript.program_log.iter().rev().skip(1) {
                frequencies[event.pc as usize / 4] += 1;
            }
            assert_eq!(frequencies, system_records.filtered_exec_frequencies);
            assert!(execution
                .transcript
                .memory_log
                .windows(2)
                .all(|events| events[0].timestamp < events[1].timestamp));
            for touched in &system_records.touched_memory {
                let values: [u16; 4] = unsafe {
                    execution
                        .state
                        .memory
                        .read(touched.address_space, touched.ptr)
                };
                assert_eq!(
                    values.map(u32::from),
                    touched.values.map(|value| value.as_canonical_u32()),
                    "final touched block differs at AS={} ptr={}",
                    touched.address_space,
                    touched.ptr
                );
                let last_event = execution
                    .transcript
                    .memory_log
                    .iter()
                    .rev()
                    .find(|event| {
                        event.address_space_and_kind & !(1 << 31) == touched.address_space
                            && event.pointer == touched.ptr
                    })
                    .expect("interpreter-touched block is absent from RVR memory log");
                assert_eq!(last_event.timestamp, touched.timestamp);
                assert_eq!(
                    last_event.value,
                    touched.values.map(|value| value.as_canonical_u32() as u16)
                );
            }

            transcript_bytes = std::mem::size_of_val(execution.transcript.program_log.as_slice())
                + std::mem::size_of_val(execution.transcript.memory_log.as_slice())
                + std::mem::size_of_val(execution.transcript.initial_write_log.as_slice());
            reusable_transcript = Some(execution.transcript);
        }

        interpreter_times.sort_unstable();
        pure_times.sort_unstable();
        metered_times.sort_unstable();
        rvr_times.sort_unstable();
        let interpreter_median = interpreter_times[REPETITIONS / 2];
        let pure_median = pure_times[REPETITIONS / 2];
        let metered_median = metered_times[REPETITIONS / 2];
        let rvr_median = rvr_times[REPETITIONS / 2];
        println!(
            "RVR_PREFLIGHT_BENCH guest_insns={} repetitions={} pure_median_us={} metered_median_us={} interpreter_median_us={} rvr_median_us={} interpreter_speedup={:.3} preflight_over_pure={:.3} preflight_over_metered={:.3} transcript_bytes={} bytes_per_insn={:.3}",
            segment.num_insns,
            REPETITIONS,
            pure_median.as_micros(),
            metered_median.as_micros(),
            interpreter_median.as_micros(),
            rvr_median.as_micros(),
            interpreter_median.as_secs_f64() / rvr_median.as_secs_f64(),
            rvr_median.as_secs_f64() / pure_median.as_secs_f64(),
            rvr_median.as_secs_f64() / metered_median.as_secs_f64(),
            transcript_bytes,
            transcript_bytes as f64 / segment.num_insns as f64,
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_preflight_logs_memory_schedule() -> Result<()> {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let reg_pointer = |index: u32| index * 4;
        let addi = |rd: usize, immediate: usize| {
            Instruction::<F>::from_usize(
                BaseAluImmOpcode::ADDI.global_opcode(),
                [
                    reg(rd),
                    0,
                    immediate,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                    1,
                    0,
                ],
            )
        };
        let memory = |opcode: Rv64LoadStoreOpcode, a: usize, base: usize| {
            Instruction::<F>::from_usize(
                opcode.global_opcode(),
                [
                    reg(a),
                    reg(base),
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            )
        };
        let instructions = [
            addi(1, 1),
            addi(2, 0x123),
            memory(Rv64LoadStoreOpcode::STOREW, 2, 1),
            memory(Rv64LoadStoreOpcode::LOADW, 3, 1),
            addi(1, 6),
            memory(Rv64LoadStoreOpcode::STOREW, 2, 1),
            memory(Rv64LoadStoreOpcode::LOADW, 0, 1),
            Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let executor = VmExecutor::new(test_rv64im_config())?;
        let execution = executor.rvr_preflight_instance(&exe, None)?.execute(
            Vec::<Vec<u8>>::new(),
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(8, 19),
        )?;

        assert_eq!(
            execution
                .transcript
                .program_log
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            vec![
                (0, 1),
                (4, 3),
                (8, 5),
                (12, 9),
                (16, 13),
                (20, 15),
                (24, 19),
                (28, 23),
                (28, 23),
            ]
        );

        let memory = &execution.transcript.memory_log;
        assert_eq!(memory.len(), 19);
        assert_eq!(
            memory
                .iter()
                .map(|event| event.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        );

        const WRITE_BIT: u32 = 1 << 31;
        let writes = memory
            .iter()
            .filter(|event| event.address_space_and_kind & WRITE_BIT != 0)
            .map(|event| (event.timestamp, event.pointer, event.value))
            .collect::<Vec<_>>();
        assert_eq!(
            writes,
            vec![
                (2, reg_pointer(1), [1, 0, 0, 0]),
                (4, reg_pointer(2), [0x123, 0, 0, 0]),
                (7, 0, [0x2300, 1, 0, 0]),
                (12, reg_pointer(3), [0x123, 0, 0, 0]),
                (14, reg_pointer(1), [6, 0, 0, 0]),
                (17, 0, [0x2300, 1, 0, 0x123]),
                (18, 4, [0, 0, 0, 0]),
            ]
        );

        assert_eq!(
            execution
                .transcript
                .initial_write_log
                .iter()
                .map(|event| (event.address_space, event.pointer, event.initial_value))
                .collect::<Vec<_>>(),
            vec![
                (RV64_REGISTER_AS, reg_pointer(1), [0; 4]),
                (RV64_REGISTER_AS, reg_pointer(2), [0; 4]),
                (RV64_MEMORY_AS, 0, [0; 4]),
                (RV64_REGISTER_AS, reg_pointer(3), [0; 4]),
                (RV64_MEMORY_AS, 4, [0; 4]),
            ]
        );
        Ok(())
    }

    #[test_case("rvr_invalid_branch_taken"; "invalid_branch_taken")]
    #[test_case("out_of_bound_reveal"; "out_of_bound_reveal")]
    #[test_case("rvr_hint_buffer_zero"; "hint_buffer_zero")]
    #[cfg(feature = "rvr")]
    fn test_rvr_example_traps(program_name: &str) {
        assert_rvr_example_traps(program_name);
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_reveal_negative_offset() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(32);
        let elf = build_example_program_at_path(
            get_programs_dir!(),
            "rvr_reveal_negative_offset",
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        let executor = VmExecutor::new(config)?;
        let state = executor.instance(&exe)?.execute(vec![])?;
        let public_values = extract_public_values(32, &state.memory.memory);

        assert_eq!(
            u64::from_le_bytes(public_values[..8].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_io() {
        execute_rvr_example_with_input("hint", vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_concurrent_host_contexts() -> Result<()> {
        const NUM_THREADS: usize = 8;
        const NUM_RUNS: usize = 8;

        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(64);
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let barrier = Barrier::new(NUM_THREADS);
        let expected_prefix = (0u8..32).collect::<Vec<_>>();

        thread::scope(|scope| {
            for _ in 0..NUM_THREADS {
                scope.spawn(|| {
                    barrier.wait();
                    for _ in 0..NUM_RUNS {
                        let state = instance.execute(vec![]).unwrap();
                        let public_values = extract_public_values(64, &state.memory.memory);
                        assert_eq!(&public_values[..32], &expected_prefix);
                        assert_eq!(
                            u64::from_le_bytes(public_values[32..40].try_into().unwrap()),
                            123
                        );
                        assert_eq!(
                            u64::from_le_bytes(public_values[40..48].try_into().unwrap()),
                            456
                        );
                    }
                });
            }
        });

        Ok(())
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_example_traps(program_name: &str) {
        assert_rvr_example_with_config_and_input_traps(program_name, test_rv64im_config(), vec![]);
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_example_with_config_traps(program_name: &str, config: Rv64ImConfig) {
        assert_rvr_example_with_config_and_input_traps(program_name, config, vec![]);
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_example_traps_with_input(program_name: &str, input: Vec<Vec<u8>>) {
        assert_rvr_example_with_config_and_input_traps(program_name, test_rv64im_config(), input);
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_example_with_config_and_input_traps(
        program_name: &str,
        config: Rv64ImConfig,
        input: Vec<Vec<u8>>,
    ) {
        let elf =
            build_example_program_at_path(get_programs_dir!(), program_name, &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let result = executor.instance(&exe).unwrap().execute(input);

        match result {
            Err(ExecutionError::RvrExecution(message)) => {
                assert_eq!(message, "execution returned error code: 3");
            }
            Err(error) => panic!("expected an RVR execution error, got {error}"),
            Ok(_) => panic!("expected RVR execution to fail"),
        }
    }

    #[test_case("fibonacci", 1)]
    fn test_rv64i(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv64IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let mut exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        change_rv64m_insn_to_nop(&mut exe);
        air_test_with_min_segments(Rv64IBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_suspend() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "fibonacci", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        #[cfg(feature = "rvr")]
        let (end_state1, end_state2) = {
            let tracking_instance = executor.rvr_instret_tracking_instance(&exe, None)?;

            let initial_pc = exe.pc_start;
            let zero_budget_state = match tracking_instance.execute_for(vec![], 0)? {
                ExecutionOutcome::Suspended(execution) => execution.state,
                ExecutionOutcome::Terminated(_) => {
                    panic!("zero-budget execution unexpectedly terminated")
                }
            };
            assert_eq!(zero_budget_state.pc(), initial_pc);

            let artifact_dir = tempfile::tempdir()?;
            let artifact_path = tracking_instance.save(artifact_dir.path())?;
            assert!(artifact_path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .ends_with("-pure-with-instret-tracking"));
            let loaded_tracking_instance =
                executor.load_instret_tracking_instance(&artifact_path, &exe)?;
            let loaded_zero_budget_state = match loaded_tracking_instance.execute_for(vec![], 0)? {
                ExecutionOutcome::Suspended(execution) => execution.state,
                ExecutionOutcome::Terminated(_) => {
                    panic!("zero-budget execution unexpectedly terminated")
                }
            };
            assert_eq!(loaded_zero_budget_state.pc(), initial_pc);

            let unlimited_instance = executor.instance(&exe)?;
            let state = tracking_instance
                .execute_for(vec![], 10)?
                .into_inner()
                .state;
            let state = tracking_instance
                .execute_from_state_for(state, 10)?
                .into_inner()
                .state;
            let end_state1 = tracking_instance.execute_from_state(state)?.state;
            let end_state2 = unlimited_instance.execute(vec![])?;
            (end_state1, end_state2)
        };
        #[cfg(not(feature = "rvr"))]
        let (end_state1, end_state2) = {
            let instance = executor.instance(&exe)?;
            let state = instance.execute_for(vec![], 10)?.into_inner();
            let state = instance.execute_from_state_for(state, 10)?.into_inner();
            let end_state1 = instance.execute_from_state(state)?;
            let end_state2 = instance.execute(vec![])?;
            (end_state1, end_state2)
        };
        assert_eq!(end_state1.pc(), end_state2.pc());
        for addr_space in 1..end_state1.memory.memory.mem.len() {
            assert_eq!(
                end_state1.memory.memory.mem[addr_space].size(),
                end_state2.memory.memory.mem[addr_space].size()
            );
            let len = end_state2.memory.memory.mem[addr_space].size();
            for i in 0..len {
                unsafe {
                    assert_eq!(
                        end_state1
                            .memory
                            .read_bytes::<1>(addr_space as u32, i as u32),
                        end_state2
                            .memory
                            .read_bytes::<1>(addr_space as u32, i as u32)
                    );
                }
            }
        }
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv64im(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Rv64MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    // Exercises the std path: guest is built with --features std, which pulls in
    // libstd compiled for riscv64im-unknown-openvm-elf and links against our PAL.
    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    #[test_case("std_collections", 1)]
    fn test_rv64im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            example_name,
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Rv64MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        let input = vec![vec![0u8, 1, 2, 3]];
        air_test_with_min_segments(Rv64ImBuilder, config, exe, input, 1);
        Ok(())
    }

    /// NOTE: This test is slow because it processes > 1MB of data. It is marked #[ignore]
    /// and can be run with: cargo test -p openvm-riscv-integration-tests test_hint_buffer_chunking
    /// -- --ignored
    #[test]
    #[ignore = "slow test: processes >1MB of data"]
    fn test_hint_buffer_chunking() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint_large_buffer", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        // Create input buffer larger than MAX_HINT_BUFFER_WORDS
        // This will require chunking to succeed
        let expected_words = MAX_HINT_BUFFER_DWORDS + 100;
        let expected_len = expected_words * RV64_REGISTER_NUM_LIMBS;

        // Create data with a pattern that can be verified
        let data: Vec<u8> = (0..expected_len).map(|i| (i % 256) as u8).collect();

        let input = vec![data];
        air_test_with_min_segments(Rv64ImBuilder, config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "read", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        #[derive(serde::Serialize)]
        struct Foo {
            bar: u32,
            baz: Vec<u32>,
        }
        let foo = Foo {
            bar: 42,
            baz: vec![0, 1, 2, 3],
        };
        let serialized_foo = openvm::serde::to_vec(&foo).unwrap();
        let input: Vec<u8> = serialized_foo
            .into_iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![input], 1);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Memory access out of bounds")]
    #[cfg(not(feature = "rvr"))]
    fn test_reveal_beyond_num_public_values_errors() {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(32);
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();

        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![]).unwrap();
    }

    #[test]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_reveal_beyond_num_public_values_errors() {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(32);
        assert_rvr_example_with_config_traps("reveal", config);
    }

    #[test]
    fn test_reveal() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values_bytes(64);
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config.clone())?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![])?;
        let final_memory = state.memory.memory;
        let hasher = vm_poseidon2_hasher::<F>();
        let md = config.as_ref().memory_config.memory_dimensions();
        let tree = MerkleTree::from_memory(&final_memory, &md, &hasher);
        let top_tree = tree.top_tree(md.addr_space_height);
        let pv_proof =
            UserPublicValuesProof::compute(config.as_ref(), &hasher, &final_memory, &top_tree);

        // `pv_proof.public_values` is the u16-packed merkle leaf representation;
        // user-facing byte content is read via `extract_public_values`.
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let expected_bytes = bytes
            .into_iter()
            .chain(
                [123, 0, 456, 0u32, 0u32, 0u32, 0u32, 0u32]
                    .into_iter()
                    .flat_map(|x| x.to_le_bytes()),
            )
            .collect::<Vec<_>>();
        assert_eq!(extract_public_values(64, &final_memory), expected_bytes);

        // Sanity-check the merkle leaves are the u16 little-endian packing of the
        // first `num_public_values` u16 cells.
        let expected_leaves: Vec<F> = expected_bytes
            .chunks_exact(2)
            .take(pv_proof.public_values.len())
            .map(|c| F::from_u16(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(pv_proof.public_values, expected_leaves);
        Ok(())
    }

    #[test]
    fn test_print() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "print", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let input = vec![vec![0u8, 0, 0, 1]];
        match instance.execute(input.clone()) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn test_hashmap() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hashmap", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "tiny-mem-test",
            ["heap-embedded-alloc"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test_case("misaligned_load", 1)]
    #[test_case("misaligned_signed_load", 1)]
    #[test_case("misaligned_store", 1)]
    #[test_case("mem_intrinsics", 1)]
    fn test_misaligned_mem_access(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Rv64MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    #[should_panic]
    #[cfg(not(feature = "rvr"))]
    fn test_load_x0() {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![]).unwrap();
    }

    #[test]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_rvr_load_x0_traps() {
        assert_rvr_example_traps("load_x0");
    }

    #[test]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_out_of_bound_mem_access() {
        assert_rvr_example_traps("out_of_bound_mem_access");
    }

    #[test_case("out_of_bound_print_str"; "print_str_out_of_bounds")]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_rvr_protected_execution_traps(program_name: &str) {
        assert_rvr_example_traps(program_name);
    }

    #[test]
    #[cfg(feature = "rvr")]
    fn test_rvr_hint_buffer_rejects_oversized_count() {
        assert_rvr_example_traps_with_input("rvr_hint_buffer_oversized", vec![vec![0u8; 8192]]);
    }

    #[test_case("getrandom", vec!["getrandom", "getrandom-unsupported"])]
    #[test_case("getrandom", vec!["getrandom"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02", "getrandom-unsupported"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02/custom"])]
    fn test_getrandom_unsupported(program: &str, features: Vec<&str>) {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            program,
            &features,
            &config,
        )
        .unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        air_test(Rv64ImBuilder, config, exe);
    }

    // For testing programs that should only execute RV64I:
    // The ELF might still have Mul instructions even though the program doesn't use them. We
    // mask those to NOP here.
    fn change_rv64m_insn_to_nop(exe: &mut VmExe<F>) {
        for (insn, _) in exe
            .program
            .instructions_and_debug_infos
            .iter_mut()
            .flatten()
        {
            if MulOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || MulHOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || DivRemOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
            {
                *insn = Instruction::default();
                insn.opcode = SystemOpcode::PHANTOM.global_opcode();
            }
        }
    }
}
