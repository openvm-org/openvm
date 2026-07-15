#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use eyre::Result;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::{
        Fp2Extension, Rv64ModularBuilder, Rv64ModularConfig, Rv64ModularWithFp2Builder,
        Rv64ModularWithFp2Config,
    };
    use openvm_algebra_transpiler::{
        Fp2Opcode, Fp2TranspilerExtension, ModularTranspilerExtension,
    };
    use openvm_circuit::utils::{air_test, test_system_config};
    use openvm_ecc_circuit::SECP256K1_CONFIG;
    use openvm_instructions::exe::VmExe;
    #[cfg(feature = "rvr")]
    use openvm_pairing_guest::bls12_381::{
        BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_MODULUS, BLS12_381_ORDER,
    };
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir, NoInitFile};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[cfg(feature = "rvr")]
    use std::{collections::BTreeMap, sync::Arc};

    #[cfg(feature = "rvr")]
    use openvm_algebra_circuit::{Rv64ModularCpuBuilder, Rv64ModularWithFp2CpuBuilder};
    #[cfg(feature = "rvr")]
    use openvm_circuit::arch::rvr::preflight::RvrArenaNativeTarget;
    #[cfg(feature = "rvr")]
    use openvm_circuit::{
        arch::{
            rvr::{
                generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightEngine,
                RvrPreflightOutput, RvrPreflightRoute, VmRvrLogNativeExtension,
            },
            verify_segments, ContinuationVmProver, DenseRecordArena, MatrixRecordArena, Streams,
            VirtualMachine, VmInstance,
        },
        system::SystemRecords,
        utils::test_cpu_engine,
    };
    #[cfg(feature = "rvr")]
    use openvm_instructions::LocalOpcode;

    #[cfg(test)]
    fn test_rv64modular_config(moduli: Vec<BigUint>) -> Rv64ModularConfig {
        let mut config = Rv64ModularConfig::new(moduli);
        config.system = test_system_config();
        config
    }

    #[cfg(test)]
    fn test_rv64modularwithfp2_config(
        moduli_with_names: Vec<(String, BigUint)>,
    ) -> Rv64ModularWithFp2Config {
        let mut config = Rv64ModularWithFp2Config::new(moduli_with_names);
        *config.as_mut() = test_system_config();
        config
    }

    #[cfg(feature = "rvr")]
    fn build_rvr_modular_exe(config: &Rv64ModularConfig) -> Result<VmExe<F>> {
        build_rvr_modular_exe_named(config, "rvr_modular")
    }

    #[cfg(feature = "rvr")]
    fn build_rvr_modular_exe_named(config: &Rv64ModularConfig, example: &str) -> Result<VmExe<F>> {
        let elf = build_example_program_at_path(get_programs_dir!(), example, config)?;
        Ok(VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?)
    }

    #[cfg(feature = "rvr")]
    fn build_rvr_fp2_exe(config: &Rv64ModularWithFp2Config, example: &str) -> Result<VmExe<F>> {
        let elf = build_example_program_at_path(get_programs_dir!(), example, config)?;
        Ok(VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?)
    }

    #[cfg(feature = "rvr")]
    fn assert_system_records_eq(label: &str, interp: &SystemRecords<F>, rvr: &SystemRecords<F>) {
        assert_eq!(interp.from_state, rvr.from_state, "{label}: from_state");
        assert_eq!(interp.to_state, rvr.to_state, "{label}: to_state");
        assert_eq!(interp.exit_code, rvr.exit_code, "{label}: exit_code");
        assert_eq!(
            interp.filtered_exec_frequencies, rvr.filtered_exec_frequencies,
            "{label}: filtered_exec_frequencies"
        );
        assert_eq!(
            interp.touched_memory, rvr.touched_memory,
            "{label}: touched_memory"
        );
    }

    #[cfg(feature = "rvr")]
    fn fp2_air_ids(
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
        num_moduli: usize,
    ) -> Vec<usize> {
        let mut ids = exe
            .program
            .instructions_and_debug_infos
            .iter()
            .zip(pc_to_air_idx)
            .filter_map(|(slot, &air_idx)| {
                let (instruction, _) = slot.as_ref()?;
                let opcode = instruction.opcode.as_usize();
                (opcode >= Fp2Opcode::CLASS_OFFSET).then_some(())?;
                let relative = opcode - Fp2Opcode::CLASS_OFFSET;
                let opcode_count = Fp2Opcode::SETUP_MULDIV as usize + 1;
                (relative < num_moduli * opcode_count).then_some(())?;
                let local = relative % opcode_count;
                (local <= Fp2Opcode::SETUP_MULDIV as usize).then_some(air_idx?)
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 2, "Fp2 AddSub and MulDiv AIRs must be mapped");
        ids
    }

    #[cfg(feature = "rvr")]
    fn assert_modular_timestamp_deltas(exe: &VmExe<F>, output: &RvrPreflightOutput<F>) {
        use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode as Op;

        for (idx, entry) in output.raw_logs.program_log.iter().enumerate() {
            let pc = entry.pc as u32;
            let instruction_idx = ((pc - exe.program.pc_base) / 4) as usize;
            let Some((instruction, _)) = &exe.program.instructions_and_debug_infos[instruction_idx]
            else {
                continue;
            };
            let opcode = instruction.opcode.as_usize();
            if opcode < Op::CLASS_OFFSET {
                continue;
            }
            let local = (opcode - Op::CLASS_OFFSET) % (Op::SETUP_ISEQ as usize + 1);
            let expected_delta = if matches!(local, x if x == Op::IS_EQ as usize || x == Op::SETUP_ISEQ as usize)
            {
                2 + 2 * 4 + 1
            } else {
                2 + 1 + 2 * 4 + 4
            };
            let next_timestamp = output
                .raw_logs
                .program_log
                .get(idx + 1)
                .map(|next| next.timestamp)
                .unwrap_or(output.system_records.to_state.timestamp);
            assert_eq!(
                next_timestamp - entry.timestamp,
                expected_delta,
                "modular opcode {opcode:#x} at pc {pc:#x} timestamp delta"
            );
        }
    }

    #[cfg(feature = "rvr")]
    fn assert_rvr_differential(
        label: &str,
        exe: &VmExe<F>,
        config: &Rv64ModularConfig,
        segments: Vec<(Option<u64>, Vec<u32>)>,
    ) {
        let (mut interp_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("interpreter vm init");
        let (mut rvr_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("rvr vm init");
        let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(
            interp_vm.air_names().collect::<Vec<_>>(),
            rvr_vm.air_names().collect::<Vec<_>>(),
            "{label}: AIR order"
        );
        let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");
        let widths = rvr_vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let mut interpreter = interp_vm
            .preflight_interpreter(exe)
            .expect("interpreter preflight");
        let mut state = interp_vm.create_initial_state(exe, Streams::default());
        let segments = if segments.is_empty() {
            vec![(None, vec![32768; rvr_vm.num_airs()])]
        } else {
            segments
        };

        let segment_outputs = {
            let route = rvr_vm
                .preflight_routed_instance(exe)
                .expect("routed preflight instance");
            let RvrPreflightRoute::Rvr(instance) = route else {
                panic!("{label}: modular program must route to RVR preflight");
            };
            let mut outputs = Vec::with_capacity(segments.len());
            for (segment_idx, (num_insns, trace_heights)) in segments.into_iter().enumerate() {
                let segment_label = format!("{label}_segment_{segment_idx}");
                let from_state = state.clone();
                let interp_output = interp_vm
                    .execute_preflight(
                        &mut interpreter,
                        from_state.clone(),
                        num_insns,
                        &trace_heights,
                    )
                    .expect("interpreter execution");
                let capacities = trace_heights
                    .iter()
                    .zip(&widths)
                    .map(|(&height, &width)| (height as usize, width))
                    .collect::<Vec<_>>();
                let mut staged = Vec::new();
                let mut targets = BTreeMap::new();
                for &(air, geometry) in &instance.compiled().inline_records().arena_native_airs {
                    let (height, width) = capacities[air];
                    let (arena, target) =
                        MatrixRecordArena::<F>::stage_arena_native(height, width, &geometry);
                    targets.insert(air, target);
                    staged.push((air, geometry, arena));
                }
                let mut rvr_output = instance
                    .execute_preflight_from_state_with_arena_targets(
                        from_state.clone(),
                        Some(num_insns.unwrap_or(1_000_000)),
                        &trace_heights,
                        &targets,
                    )
                    .expect("rvr preflight execution");
                assert_system_records_eq(
                    &segment_label,
                    &interp_output.system_records,
                    &rvr_output.system_records,
                );
                assert_modular_timestamp_deltas(exe, &rvr_output);
                let mut rvr_arenas = generate_record_arenas_from_logs::<F, MatrixRecordArena<F>>(
                    &registry,
                    exe,
                    &mut rvr_output,
                    &capacities,
                    &pc_to_air_idx,
                )
                .expect("rvr log-native record assembly");
                for (air, geometry, mut arena) in staged {
                    let written = rvr_output
                        .arena_native_written
                        .iter()
                        .find(|&&(written_air, _)| written_air == air)
                        .map(|&(_, count)| count as usize)
                        .expect("arena-native AIR must report its written count");
                    arena.finish_arena_native(written, &geometry);
                    rvr_arenas[air] = arena;
                }
                state = rvr_output.to_state.clone();
                outputs.push((
                    from_state,
                    interp_output,
                    rvr_output.system_records,
                    rvr_arenas,
                ));
            }
            outputs
        };

        let interp_program = interp_vm.commit_program_on_device(&exe.program);
        interp_vm.load_program(interp_program);
        let rvr_program = rvr_vm.commit_program_on_device(&exe.program);
        rvr_vm.load_program(rvr_program);
        let mut active_modular_airs = Vec::new();

        for (segment_idx, (from_state, interp_output, rvr_system_records, rvr_arenas)) in
            segment_outputs.into_iter().enumerate()
        {
            let segment_label = format!("{label}_segment_{segment_idx}");
            interp_vm.transport_init_memory_to_device(&from_state.memory);
            let interp_ctx = interp_vm
                .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
                .expect("interpreter trace generation");
            rvr_vm.transport_init_memory_to_device(&from_state.memory);
            let rvr_ctx = rvr_vm
                .generate_proving_ctx(rvr_system_records, rvr_arenas)
                .expect("rvr trace generation");

            // HARD-5: compare only deterministic modular AIR rows. Shared
            // lookup/Poseidon2 periphery rows are excluded from raw row order;
            // full SystemRecords/touched_memory equality remains the oracle.
            let is_modular_air = |air_idx: usize| {
                air_names[air_idx].contains("FieldExpressionCoreAir")
                    || air_names[air_idx].contains("ModularIsEqualCoreAir")
            };
            let mut interp_traces = interp_ctx
                .per_trace
                .into_iter()
                .filter(|(air_idx, _)| is_modular_air(*air_idx))
                .collect::<BTreeMap<_, _>>();
            let mut rvr_traces = rvr_ctx
                .per_trace
                .into_iter()
                .filter(|(air_idx, _)| is_modular_air(*air_idx))
                .collect::<BTreeMap<_, _>>();
            let interp_air_ids = interp_traces.keys().copied().collect::<Vec<_>>();
            assert_eq!(
                interp_air_ids,
                rvr_traces.keys().copied().collect::<Vec<_>>(),
                "{segment_label}: active AIR set"
            );
            for air_idx in interp_air_ids {
                let interp_trace = interp_traces.remove(&air_idx).unwrap();
                let rvr_trace = rvr_traces.remove(&air_idx).unwrap();
                let air_name = &air_names[air_idx];
                assert_eq!(
                    interp_trace.common_main.width, rvr_trace.common_main.width,
                    "{segment_label}: {air_name} width"
                );
                if interp_trace.common_main.values != rvr_trace.common_main.values {
                    let first_mismatch = interp_trace
                        .common_main
                        .values
                        .iter()
                        .zip(&rvr_trace.common_main.values)
                        .position(|(left, right)| left != right);
                    panic!(
                        "{segment_label}: {air_name} values: left_len={} right_len={} first_mismatch={first_mismatch:?}",
                        interp_trace.common_main.values.len(),
                        rvr_trace.common_main.values.len(),
                    );
                }
                assert_eq!(
                    interp_trace.public_values, rvr_trace.public_values,
                    "{segment_label}: {air_name} public values"
                );
                active_modular_airs.push(air_idx);
            }
        }

        active_modular_airs.sort_unstable();
        active_modular_airs.dedup();
        assert_eq!(
            active_modular_airs.len(),
            3,
            "{label}: addsub, muldiv, and is-eq traces must all be active"
        );
    }

    #[cfg(feature = "rvr")]
    fn modular_dense_oracle(
        exe: &VmExe<F>,
        config: &Rv64ModularConfig,
    ) -> (SystemRecords<F>, Vec<DenseRecordArena>, u64, Vec<String>) {
        std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("oracle vm init");
        let air_names = vm.air_names().map(str::to_owned).collect::<Vec<_>>();
        let widths = vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = vm.pc_to_air_idx(exe).expect("oracle pc-to-air mapping");
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(exe)
            .expect("oracle routed instance")
        else {
            panic!("modular oracle must route to rvr")
        };
        let mut output = instance
            .execute_preflight_from_state(vm.create_initial_state(exe, Streams::default()), None)
            .expect("verbose oracle preflight");
        let retired = output
            .system_records
            .filtered_exec_frequencies
            .iter()
            .map(|&count| u64::from(count))
            .sum();
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
            &registry,
            exe,
            &mut output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("verbose oracle record assembly");
        (output.system_records, arenas, retired, air_names)
    }

    #[cfg(feature = "rvr")]
    fn modular_dense_direct(
        exe: &VmExe<F>,
        config: &Rv64ModularConfig,
        retired: u64,
    ) -> (SystemRecords<F>, Vec<DenseRecordArena>) {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("direct vm init");
        let air_names = vm.air_names().map(str::to_owned).collect::<Vec<_>>();
        let widths = vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = vm.pc_to_air_idx(exe).expect("direct pc-to-air mapping");
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(exe)
            .expect("direct routed instance")
        else {
            panic!("modular direct arm must route to rvr")
        };
        let modular_native_count = instance
            .compiled()
            .inline_records()
            .arena_native_airs
            .iter()
            .filter(|&&(air, _)| {
                air_names[air].contains("FieldExpressionCoreAir")
                    || air_names[air].contains("ModularIsEqualCoreAir")
            })
            .count();
        assert_eq!(
            modular_native_count, 3,
            "all three modular mixed-opcode AIRs must migrate atomically"
        );
        let mut staged = Vec::new();
        let mut targets = BTreeMap::new();
        for &(air, geometry) in &instance.compiled().inline_records().arena_native_airs {
            let (arena, target) =
                DenseRecordArena::stage_arena_native(heights[air] as usize, widths[air], &geometry);
            targets.insert(air, target);
            staged.push((air, geometry, arena));
        }
        let mut output = instance
            .execute_preflight_from_state_with_arena_targets(
                vm.create_initial_state(exe, Streams::default()),
                Some(retired),
                &heights,
                &targets,
            )
            .expect("direct-final modular preflight");
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let mut arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
            &registry,
            exe,
            &mut output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("direct residual record assembly");
        for (air, geometry, mut arena) in staged {
            let written = output
                .arena_native_written
                .iter()
                .find(|&&(written_air, _)| written_air == air)
                .map(|&(_, count)| count as usize)
                .expect("direct AIR must report written records");
            arena.finish_arena_native(written, &geometry);
            arenas[air] = arena;
        }
        (output.system_records, arenas)
    }

    #[cfg(feature = "rvr")]
    fn modular_dense_delta_without_arena(
        exe: &VmExe<F>,
        config: &Rv64ModularConfig,
        retired: u64,
    ) -> (SystemRecords<F>, Vec<DenseRecordArena>) {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("delta-without-arena vm init");
        let air_names = vm.air_names().map(str::to_owned).collect::<Vec<_>>();
        let widths = vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = vm
            .pc_to_air_idx(exe)
            .expect("delta-without-arena pc-to-air mapping");
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(exe)
            .expect("delta-without-arena routed instance")
        else {
            panic!("delta-without-arena arm must route to rvr")
        };
        for (slot, air) in pc_to_air_idx.iter().enumerate() {
            if air.is_some_and(|air| {
                air_names[air].contains("FieldExpressionCoreAir")
                    || air_names[air].contains("ModularIsEqualCoreAir")
            }) {
                assert!(
                    !instance.compiled().inline_records().pc_slots[slot],
                    "custom modular slot {slot} must fail closed to verbose assembly"
                );
            }
        }
        let mut output = instance
            .execute_preflight_from_state_with_arena_targets(
                vm.create_initial_state(exe, Streams::default()),
                Some(retired),
                &heights,
                &BTreeMap::new(),
            )
            .expect("delta-without-arena preflight");
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
            &registry,
            exe,
            &mut output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("delta-without-arena verbose record assembly");
        (output.system_records, arenas)
    }

    #[cfg(feature = "rvr")]
    fn fp2_dense_oracle(
        exe: &VmExe<F>,
        config: &Rv64ModularWithFp2Config,
    ) -> (SystemRecords<F>, Vec<DenseRecordArena>, u64, Vec<usize>) {
        std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularWithFp2CpuBuilder,
            config.clone(),
        )
        .expect("Fp2 oracle vm init");
        let widths = vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = vm.pc_to_air_idx(exe).expect("Fp2 oracle pc-to-air mapping");
        let fp2_airs = fp2_air_ids(exe, &pc_to_air_idx, config.fp2.supported_moduli.len());
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(exe)
            .expect("Fp2 oracle routed instance")
        else {
            panic!("Fp2 oracle must route to rvr")
        };
        let mut output = instance
            .execute_preflight_from_state(vm.create_initial_state(exe, Streams::default()), None)
            .expect("verbose Fp2 oracle preflight");
        let retired = output
            .system_records
            .filtered_exec_frequencies
            .iter()
            .map(|&count| u64::from(count))
            .sum();
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
            &registry,
            exe,
            &mut output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("verbose Fp2 record assembly");
        (output.system_records, arenas, retired, fp2_airs)
    }

    #[cfg(feature = "rvr")]
    fn fp2_dense_direct(
        exe: &VmExe<F>,
        config: &Rv64ModularWithFp2Config,
        retired: u64,
    ) -> (SystemRecords<F>, Vec<DenseRecordArena>) {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularWithFp2CpuBuilder,
            config.clone(),
        )
        .expect("Fp2 direct vm init");
        let widths = vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = vm.pc_to_air_idx(exe).expect("Fp2 direct pc-to-air mapping");
        let fp2_airs = fp2_air_ids(exe, &pc_to_air_idx, config.fp2.supported_moduli.len());
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(exe)
            .expect("Fp2 direct routed instance")
        else {
            panic!("Fp2 direct arm must route to rvr")
        };
        let native_fp2_count = instance
            .compiled()
            .inline_records()
            .arena_native_airs
            .iter()
            .filter(|&&(air, _)| fp2_airs.contains(&air))
            .count();
        assert_eq!(
            native_fp2_count, 2,
            "Fp2 Add/Sub/Setup and Mul/Div/Setup AIRs must migrate atomically"
        );
        let mut staged = Vec::new();
        let mut targets = BTreeMap::new();
        for &(air, geometry) in &instance.compiled().inline_records().arena_native_airs {
            let (arena, target) =
                DenseRecordArena::stage_arena_native(heights[air] as usize, widths[air], &geometry);
            targets.insert(air, target);
            staged.push((air, geometry, arena));
        }
        let mut output = instance
            .execute_preflight_from_state_with_arena_targets(
                vm.create_initial_state(exe, Streams::default()),
                Some(retired),
                &heights,
                &targets,
            )
            .expect("direct-final Fp2 preflight");
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let mut arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
            &registry,
            exe,
            &mut output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("Fp2 direct residual record assembly");
        for (air, geometry, mut arena) in staged {
            let written = output
                .arena_native_written
                .iter()
                .find(|&&(written_air, _)| written_air == air)
                .map(|&(_, count)| count as usize)
                .expect("Fp2 direct AIR must report written records");
            arena.finish_arena_native(written, &geometry);
            arenas[air] = arena;
        }
        (output.system_records, arenas)
    }

    #[cfg(feature = "rvr")]
    fn assert_fp2_rvr_interpreter_parity(exe: &VmExe<F>, config: &Rv64ModularWithFp2Config) {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (mut interp_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularWithFp2CpuBuilder,
            config.clone(),
        )
        .expect("Fp2 interpreter vm init");
        let (mut rvr_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularWithFp2CpuBuilder,
            config.clone(),
        )
        .expect("Fp2 rvr vm init");
        let widths = rvr_vm
            .pk()
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect::<Vec<_>>();
        let heights = vec![32768u32; rvr_vm.num_airs()];
        let capacities = heights
            .iter()
            .zip(&widths)
            .map(|(&height, &width)| (height as usize, width))
            .collect::<Vec<_>>();
        let pc_to_air_idx = rvr_vm
            .pc_to_air_idx(exe)
            .expect("Fp2 parity pc-to-air mapping");
        let fp2_airs = fp2_air_ids(exe, &pc_to_air_idx, config.fp2.supported_moduli.len());
        let mut interpreter = interp_vm
            .preflight_interpreter(exe)
            .expect("Fp2 interpreter preflight");
        let state = interp_vm.create_initial_state(exe, Streams::default());
        let interp_output = interp_vm
            .execute_preflight(&mut interpreter, state.clone(), None, &heights)
            .expect("Fp2 interpreter execution");
        let RvrPreflightRoute::Rvr(instance) = rvr_vm
            .preflight_routed_instance(exe)
            .expect("Fp2 parity routed instance")
        else {
            panic!("Fp2 parity must route to rvr")
        };
        let mut staged = Vec::new();
        let mut targets = BTreeMap::new();
        for &(air, geometry) in &instance.compiled().inline_records().arena_native_airs {
            let (arena, target) = MatrixRecordArena::<F>::stage_arena_native(
                heights[air] as usize,
                widths[air],
                &geometry,
            );
            targets.insert(air, target);
            staged.push((air, geometry, arena));
        }
        let mut rvr_output = instance
            .execute_preflight_from_state_with_arena_targets(
                state.clone(),
                Some(1_000_000),
                &heights,
                &targets,
            )
            .expect("Fp2 rvr execution");
        assert_system_records_eq(
            "Fp2 direct-final interpreter parity",
            &interp_output.system_records,
            &rvr_output.system_records,
        );
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        let mut rvr_arenas = generate_record_arenas_from_logs::<F, MatrixRecordArena<F>>(
            &registry,
            exe,
            &mut rvr_output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("Fp2 parity residual assembly");
        for (air, geometry, mut arena) in staged {
            let written = rvr_output
                .arena_native_written
                .iter()
                .find(|&&(written_air, _)| written_air == air)
                .map(|&(_, count)| count as usize)
                .expect("Fp2 parity AIR must report written records");
            arena.finish_arena_native(written, &geometry);
            rvr_arenas[air] = arena;
        }

        let interp_program = interp_vm.commit_program_on_device(&exe.program);
        interp_vm.load_program(interp_program);
        let rvr_program = rvr_vm.commit_program_on_device(&exe.program);
        rvr_vm.load_program(rvr_program);
        interp_vm.transport_init_memory_to_device(&state.memory);
        let interp_ctx = interp_vm
            .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
            .expect("Fp2 interpreter tracegen");
        rvr_vm.transport_init_memory_to_device(&state.memory);
        let rvr_ctx = rvr_vm
            .generate_proving_ctx(rvr_output.system_records, rvr_arenas)
            .expect("Fp2 rvr tracegen");
        let mut interp_traces = interp_ctx
            .per_trace
            .into_iter()
            .filter(|(air, _)| fp2_airs.contains(air))
            .collect::<BTreeMap<_, _>>();
        let mut rvr_traces = rvr_ctx
            .per_trace
            .into_iter()
            .filter(|(air, _)| fp2_airs.contains(air))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(interp_traces.len(), 2, "both Fp2 AIRs must be active");
        assert_eq!(
            interp_traces.keys().collect::<Vec<_>>(),
            rvr_traces.keys().collect::<Vec<_>>()
        );
        for air in fp2_airs {
            let interp = interp_traces
                .remove(&air)
                .expect("active interpreter Fp2 AIR");
            let rvr = rvr_traces.remove(&air).expect("active rvr Fp2 AIR");
            assert_eq!(interp.common_main.width, rvr.common_main.width);
            assert_eq!(interp.common_main.values, rvr.common_main.values);
            assert_eq!(interp.public_values, rvr.public_values);
        }
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    }

    #[test]
    fn test_moduli_setup() -> Result<()> {
        let moduli = ["4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", "1000000000000000003", "2305843009213693951"]
            .map(|s| BigUint::from_str(s).unwrap());
        let config = test_rv64modular_config(moduli.to_vec());
        let elf = build_example_program_at_path(get_programs_dir!(), "moduli_setup", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(Rv64ModularBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_modular() -> Result<()> {
        let config = test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]);
        let elf = build_example_program_at_path(get_programs_dir!(), "little", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularBuilder, config, openvm_exe);
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_modular_rvr_preflight_differential() -> Result<()> {
        let config = test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]);
        let exe = build_rvr_modular_exe(&config)?;
        assert_rvr_differential("modular_single", &exe, &config, Vec::new());
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_modular_rvr_direct_final_bytes_match_verbose_twice() -> Result<()> {
        let bls12_381_fq = BigUint::from_str(
            "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        )?;
        for (label, config, example) in [
            (
                "32-byte",
                test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]),
                "rvr_modular",
            ),
            (
                "48-byte",
                test_rv64modular_config(vec![bls12_381_fq]),
                "rvr_modular_48",
            ),
        ] {
            let exe = build_rvr_modular_exe_named(&config, example)?;
            let (oracle_system, oracle_arenas, retired, air_names) =
                modular_dense_oracle(&exe, &config);
            let modular_airs = air_names
                .iter()
                .enumerate()
                .filter_map(|(air, name)| {
                    (name.contains("FieldExpressionCoreAir")
                        || name.contains("ModularIsEqualCoreAir"))
                    .then_some(air)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                modular_airs.len(),
                3,
                "{label}: expected three modular AIRs"
            );

            for pass in 0..2 {
                let (direct_system, direct_arenas) = modular_dense_direct(&exe, &config, retired);
                assert_system_records_eq(
                    &format!("{label} modular direct-final byte oracle pass {pass}"),
                    &oracle_system,
                    &direct_system,
                );
                for &air in &modular_airs {
                    assert_eq!(
                        oracle_arenas[air].allocated(),
                        direct_arenas[air].allocated(),
                        "{label}, pass {pass}, air {air} ({}): direct-final bytes",
                        air_names[air]
                    );
                }
            }
            if label == "32-byte" {
                let (fallback_system, fallback_arenas) =
                    modular_dense_delta_without_arena(&exe, &config, retired);
                assert_system_records_eq(
                    "delta custom records without arena-native target",
                    &oracle_system,
                    &fallback_system,
                );
                for &air in &modular_airs {
                    assert_eq!(
                        oracle_arenas[air].allocated(),
                        fallback_arenas[air].allocated(),
                        "air {air} ({}): arena-disabled delta must use verbose bytes",
                        air_names[air]
                    );
                }
            }
        }

        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_modular_rvr_preflight_multi_segment_differential() -> Result<()> {
        let mut config = test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]);
        config.system.segmentation_max_memory = 1;
        let exe = build_rvr_modular_exe(&config)?;
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularCpuBuilder,
            config.clone(),
        )
        .expect("metered vm init");
        let metered_ctx = vm.build_metered_ctx(&exe);
        let metered = vm.metered_interpreter(&exe).expect("metered interpreter");
        let (segments, _) = metered
            .execute_metered(Streams::default(), metered_ctx)
            .expect("metered execution");
        assert!(
            segments.len() > 1,
            "tight memory limit must force multiple modular segments"
        );
        let segments = segments
            .into_iter()
            .map(|segment| (Some(segment.num_insns), segment.trace_heights))
            .collect();
        assert_rvr_differential("modular_multi", &exe, &config, segments);
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_modular_rvr_multi_segment_proves_and_verifies() -> Result<()> {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let mut config = test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]);
        config.system.segmentation_max_memory = 1;
        let exe = build_rvr_modular_exe(&config)?;
        let (vm, pk) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ModularCpuBuilder, config)
                .expect("proof vm init");
        assert!(
            vm.preflight_routed_instance(&exe)
                .expect("proof route")
                .is_rvr(),
            "modular proof must use rvr preflight"
        );
        let vk = pk.get_vk();
        let cached_program_trace = vm.commit_program_on_device(&exe.program);
        let mut instance =
            VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("proof instance init");
        instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
        let proof = ContinuationVmProver::prove(&mut instance, Streams::default())
            .expect("modular continuation prove");
        assert!(
            proof.per_segment.len() > 1,
            "expected multiple proof segments"
        );
        verify_segments(&instance.vm.engine, &vk, &proof.per_segment)
            .expect("verify modular proof segments");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_fp2_rvr_direct_final_bytes_match_verbose_twice_and_interpreter() -> Result<()> {
        let bls12_381_fq = BigUint::from_str(
            "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        )?;
        for (label, config, example) in [
            (
                "32-byte",
                test_rv64modularwithfp2_config(vec![(
                    "Complex".to_string(),
                    SECP256K1_CONFIG.modulus.clone(),
                )]),
                "complex_secp256k1",
            ),
            (
                "48-byte",
                test_rv64modularwithfp2_config(vec![("BlsFp2".to_string(), bls12_381_fq)]),
                "rvr_fp2_48",
            ),
        ] {
            let exe = build_rvr_fp2_exe(&config, example)?;
            let (oracle_system, oracle_arenas, retired, fp2_airs) = fp2_dense_oracle(&exe, &config);
            for pass in 0..2 {
                let (direct_system, direct_arenas) = fp2_dense_direct(&exe, &config, retired);
                assert_system_records_eq(
                    &format!("{label} Fp2 direct-final byte oracle pass {pass}"),
                    &oracle_system,
                    &direct_system,
                );
                for &air in &fp2_airs {
                    assert_eq!(
                        oracle_arenas[air].allocated(),
                        direct_arenas[air].allocated(),
                        "{label}, pass {pass}, Fp2 air {air}: direct-final bytes"
                    );
                }
            }
            assert_fp2_rvr_interpreter_parity(&exe, &config);
        }
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_fp2_rvr_multi_segment_proves_and_verifies() -> Result<()> {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let mut config = test_rv64modularwithfp2_config(vec![(
            "Complex".to_string(),
            SECP256K1_CONFIG.modulus.clone(),
        )]);
        config.as_mut().segmentation_max_memory = 1;
        let exe = build_rvr_fp2_exe(&config, "complex_secp256k1")?;
        let (vm, pk) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ModularWithFp2CpuBuilder,
            config,
        )
        .expect("Fp2 proof vm init");
        assert!(
            vm.preflight_routed_instance(&exe)
                .expect("Fp2 proof route")
                .is_rvr(),
            "Fp2 proof must use rvr preflight"
        );
        let vk = pk.get_vk();
        let cached_program_trace = vm.commit_program_on_device(&exe.program);
        let mut instance =
            VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("Fp2 proof instance");
        instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
        let proof = ContinuationVmProver::prove(&mut instance, Streams::default())
            .expect("Fp2 continuation prove");
        assert!(
            proof.per_segment.len() > 1,
            "expected multiple Fp2 segments"
        );
        verify_segments(&instance.vm.engine, &vk, &proof.per_segment)
            .expect("verify Fp2 proof segments");
        std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
        Ok(())
    }

    #[test]
    fn test_complex_two_moduli() -> Result<()> {
        let config = test_rv64modularwithfp2_config(vec![
            (
                "Complex1".to_string(),
                BigUint::from_str("998244353").unwrap(),
            ),
            (
                "Complex2".to_string(),
                BigUint::from_str("1000000007").unwrap(),
            ),
        ]);
        let elf =
            build_example_program_at_path(get_programs_dir!(), "complex_two_moduli", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularWithFp2Builder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex_redundant_modulus() -> Result<()> {
        let config = Rv64ModularWithFp2Config {
            modular: test_rv64modular_config(vec![
                BigUint::from_str("998244353").unwrap(),
                BigUint::from_str("1000000007").unwrap(),
                BigUint::from_str("1000000009").unwrap(),
                BigUint::from_str("987898789").unwrap(),
            ]),
            fp2: Fp2Extension::new(vec![(
                "Complex2".to_string(),
                BigUint::from_str("1000000009").unwrap(),
            )]),
        };
        let elf = build_example_program_at_path(
            get_programs_dir!(),
            "complex_redundant_modulus",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularWithFp2Builder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex() -> Result<()> {
        let config = test_rv64modularwithfp2_config(vec![(
            "Complex".to_string(),
            SECP256K1_CONFIG.modulus.clone(),
        )]);
        let elf = build_example_program_at_path(get_programs_dir!(), "complex_secp256k1", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularWithFp2Builder, config, openvm_exe);
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_bls12_381_rvr_equivalence() -> Result<()> {
        let mut modular =
            Rv64ModularConfig::new(vec![BLS12_381_MODULUS.clone(), BLS12_381_ORDER.clone()]);
        modular.system = test_system_config().with_public_values_bytes(32);
        let config = Rv64ModularWithFp2Config {
            modular,
            fp2: Fp2Extension::new(vec![(
                BLS12_381_COMPLEX_STRUCT_NAME.to_string(),
                BLS12_381_MODULUS.clone(),
            )]),
        };
        let elf = build_example_program_at_path(get_programs_dir!(), "bls12_381_rvr", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularWithFp2Builder, config, openvm_exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_invalid_setup() {
        let config = test_rv64modular_config(vec![
            BigUint::from_str("998244353").unwrap(),
            BigUint::from_str("1000000007").unwrap(),
        ]);
        let elf = build_example_program_at_path(
            get_programs_dir!(),
            "invalid_setup",
            // We don't want init.rs to be generated for this test because we are testing an
            // invalid moduli_init! call
            &NoInitFile,
        )
        .unwrap();
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )
        .unwrap();
        air_test(Rv64ModularBuilder, config, openvm_exe);
    }

    #[test]
    fn test_sqrt() -> Result<()> {
        let config = test_rv64modular_config(vec![SECP256K1_CONFIG.modulus.clone()]);
        let elf = build_example_program_at_path(get_programs_dir!(), "sqrt", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64ModularBuilder, config, openvm_exe);
        Ok(())
    }
}
