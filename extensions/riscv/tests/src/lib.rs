#[cfg(test)]
mod tests {
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    use std::{env, process::Command};
    #[cfg(feature = "rvr")]
    use std::{sync::Barrier, thread, time::Instant};

    use eyre::Result;
    #[cfg(feature = "rvr")]
    use openvm_circuit::arch::{ExecutionOutcome, PreflightExecutionOutput, VirtualMachine};
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, VmExecutor},
        system::memory::{
            merkle::{
                public_values::{extract_public_values, UserPublicValuesProof},
                MerkleTree,
            },
            online::LinearMemory,
        },
        utils::{air_test, air_test_with_min_segments, test_cpu_engine, test_system_config},
    };
    use openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::Program,
        riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, SysPhantom, SystemOpcode,
    };
    use openvm_riscv_circuit::{Rv64IBuilder, Rv64IConfig, Rv64ImBuilder, Rv64ImConfig};
    use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;
    use openvm_riscv_transpiler::{
        BaseAluImmOpcode, BaseAluOpcode, BranchEqualOpcode, DivRemOpcode, MulHOpcode, MulOpcode,
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64JalrOpcode, Rv64LoadStoreOpcode,
        Rv64MTranspilerExtension, Rv64Phantom,
    };
    use openvm_stark_sdk::{
        openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32},
        p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use strum::IntoEnumIterator;
    use test_case::test_case;

    type F = BabyBear;
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    const RVR_OOB_CHILD_ENV: &str = "OPENVM_RVR_OOB_CHILD";

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
    fn test_rvr_preflight_still_rejects_callback_phantoms() -> Result<()> {
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
        let error = match executor.rvr_preflight_instance(&exe, None) {
            Ok(_) => panic!("callback phantom unexpectedly compiled for preflight"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not support RVR preflight"));
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
            addi(2, 0, 1000),
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
        let exe = VmExe::from(Program::from_instructions(&instructions));
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            openvm_riscv_circuit::Rv64ImCpuBuilder,
            config,
        )?;

        let metered_ctx = vm.build_metered_ctx(&exe);
        let (segments, _) = vm
            .metered_instance(&exe)?
            .execute_metered(Vec::<Vec<u8>>::new(), metered_ctx)?;
        assert_eq!(segments.len(), 1, "benchmark input must fit one segment");
        let segment = &segments[0];
        let max_instructions = usize::try_from(segment.num_insns)?;
        let max_memory_events = max_instructions
            .checked_mul(4)
            .ok_or_else(|| eyre::eyre!("benchmark memory-event capacity overflow"))?;

        let rvr = vm.executor().rvr_preflight_instance(&exe, None)?;
        let mut interpreter = vm.preflight_interpreter(&exe)?;
        let limits =
            openvm_circuit::arch::rvr::RvrPreflightLimits::new(max_instructions, max_memory_events);
        let mut interpreter_times = Vec::with_capacity(REPETITIONS);
        let mut rvr_times = Vec::with_capacity(REPETITIONS);
        let mut transcript_bytes = 0usize;

        for _ in 0..REPETITIONS {
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
            let execution = rvr.execute_from_state(rvr_state, limits)?;
            rvr_times.push(started.elapsed());

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
                    touched.values.map(|value| value.as_canonical_u32())
                );
            }

            transcript_bytes = std::mem::size_of_val(execution.transcript.program_log.as_slice())
                + std::mem::size_of_val(execution.transcript.memory_log.as_slice())
                + std::mem::size_of_val(execution.transcript.initial_write_log.as_slice());
        }

        interpreter_times.sort_unstable();
        rvr_times.sort_unstable();
        let interpreter_median = interpreter_times[REPETITIONS / 2];
        let rvr_median = rvr_times[REPETITIONS / 2];
        println!(
            "RVR_PREFLIGHT_BENCH guest_insns={} repetitions={} interpreter_median_us={} rvr_median_us={} speedup={:.3} transcript_bytes={} bytes_per_insn={:.3}",
            segment.num_insns,
            REPETITIONS,
            interpreter_median.as_micros(),
            rvr_median.as_micros(),
            interpreter_median.as_secs_f64() / rvr_median.as_secs_f64(),
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

    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn assert_child_aborts(test_name: &str) {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(RVR_OOB_CHILD_ENV, "1")
            .output()
            .expect("failed to spawn self");

        if output.status.success() {
            panic!("child process succeeded; OOB access was not caught");
        }
        // Success path for these tests: relay the child failure text so the
        // caller's `#[should_panic(expected = ...)]` can match it.
        panic!("{}", String::from_utf8_lossy(&output.stderr));
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
    #[should_panic(expected = "Memory access out of bounds")]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_rvr_load_x0_traps() {
        if env::var(RVR_OOB_CHILD_ENV).is_ok() {
            execute_rvr_example("load_x0");
            return;
        }

        assert_child_aborts("tests::test_rvr_load_x0_traps");
    }

    #[test]
    #[should_panic(expected = "Memory access out of bounds")]
    #[cfg(all(feature = "rvr", not(feature = "unprotected")))]
    fn test_out_of_bound_mem_access() {
        // Child mode: triggers the OOB; abort_oob in C aborts the process.
        if env::var(RVR_OOB_CHILD_ENV).is_ok() {
            execute_rvr_example("out_of_bound_mem_access");
            return; // unreachable: abort fired
        }

        // Parent mode: spawn ourselves as the child and forward its stderr
        // as our own panic. `#[should_panic(expected = ...)]` matches iff
        // the child's rvr bounds check actually fired.
        assert_child_aborts("tests::test_out_of_bound_mem_access");
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
