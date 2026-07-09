#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use eyre::Result;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::{
        Fp2Extension, Rv64ModularBuilder, Rv64ModularConfig, Rv64ModularWithFp2Builder,
        Rv64ModularWithFp2Config,
    };
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
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
    use std::collections::BTreeMap;

    #[cfg(feature = "rvr")]
    use openvm_algebra_circuit::Rv64ModularCpuBuilder;
    #[cfg(feature = "rvr")]
    use openvm_circuit::{
        arch::{
            rvr::{
                generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightOutput,
                RvrPreflightRoute, VmRvrLogNativeExtension,
            },
            MatrixRecordArena, Streams, VirtualMachine,
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
        let elf = build_example_program_at_path(get_programs_dir!(), "rvr_modular", config)?;
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
                let rvr_output = instance
                    .execute_preflight_from_state(from_state.clone(), num_insns)
                    .expect("rvr preflight execution");
                assert_system_records_eq(
                    &segment_label,
                    &interp_output.system_records,
                    &rvr_output.system_records,
                );
                assert_modular_timestamp_deltas(exe, &rvr_output);
                let capacities = trace_heights
                    .iter()
                    .zip(&widths)
                    .map(|(&height, &width)| (height as usize, width))
                    .collect::<Vec<_>>();
                let rvr_arenas = generate_record_arenas_from_logs::<F, MatrixRecordArena<F>>(
                    &registry,
                    exe,
                    &rvr_output,
                    &capacities,
                    &pc_to_air_idx,
                )
                .expect("rvr log-native record assembly");
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
