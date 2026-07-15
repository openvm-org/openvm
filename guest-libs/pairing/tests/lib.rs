#![allow(non_snake_case)]

#[cfg(feature = "bn254")]
mod bn254 {
    use std::iter;

    use eyre::Result;
    use halo2curves_axiom::{
        bn256::{Fq12, Fq2, Fr, G1Affine, G2Affine},
        ff::Field,
    };
    use openvm_algebra_circuit::{Fp2Extension, Rv64ModularConfig};
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::utils::{
        air_test, air_test_impl, air_test_with_min_segments, test_system_config,
        TestStarkEngine as Engine,
    };
    use openvm_ecc_circuit::{
        CurveConfig, Rv64WeierstrassBuilder, Rv64WeierstrassConfig, WeierstrassExtension,
    };
    use openvm_ecc_guest::{
        algebra::{field::FieldExtension, IntMod},
        AffinePoint,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_instructions::exe::VmExe;
    use openvm_pairing_circuit::{
        PairingCurve, PairingExtension, Rv64PairingBuilder, Rv64PairingConfig,
    };
    use openvm_pairing_guest::{
        bn254::{BN254_COMPLEX_STRUCT_NAME, BN254_MODULUS},
        halo2curves_shims::bn254::Bn254,
        pairing::{EvaluatedLine, FinalExp, LineMulDType, MillerStep, MultiMillerLoop},
    };
    use openvm_pairing_transpiler::PairingTranspilerExtension;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::{openvm_stark_backend::SystemParams, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use rand08::SeedableRng;

    type F = BabyBear;

    #[cfg(test)]
    pub fn get_testing_config() -> Rv64PairingConfig {
        let primes = [BN254_MODULUS.clone()];
        let complex_struct_names = [BN254_COMPLEX_STRUCT_NAME.to_string()];
        let primes_with_names = complex_struct_names
            .into_iter()
            .zip(primes.clone())
            .collect::<Vec<_>>();
        Rv64PairingConfig {
            modular: Rv64ModularConfig::new(primes.to_vec()),
            fp2: Fp2Extension::new(primes_with_names),
            weierstrass: WeierstrassExtension::new(vec![]),
            pairing: PairingExtension::new(vec![PairingCurve::Bn254]),
        }
    }

    #[cfg(test)]
    fn test_rv64weierstrass_config(curves: Vec<CurveConfig>) -> Rv64WeierstrassConfig {
        let mut config = Rv64WeierstrassConfig::new(curves);
        *config.as_mut() = test_system_config();
        config
    }

    #[test]
    fn test_bn_ec() -> Result<()> {
        let curve = PairingCurve::Bn254.curve_config();
        let config = test_rv64weierstrass_config(vec![curve]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bn_ec",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_bn254_fp12_mul() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "fp12_mul",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(2);
        let f0 = Fq12::random(&mut rng);
        let f1 = Fq12::random(&mut rng);
        let r = f0 * f1;

        let io = [f0, f1, r]
            .into_iter()
            .flat_map(|fp12| fp12.to_coeffs())
            .flat_map(|fp2| fp2.to_bytes())
            .collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io], 1);
        Ok(())
    }

    #[test]
    fn test_bn254_line_functions() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_line",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(2);
        let a = G2Affine::random(&mut rng);
        let b = G2Affine::random(&mut rng);
        let c = G2Affine::random(&mut rng);

        let f = Fq12::random(&mut rng);
        let l0 = EvaluatedLine::<Fq2> { b: a.x, c: a.y };
        let l1 = EvaluatedLine::<Fq2> { b: b.x, c: b.y };

        // Test mul_013_by_013
        let r0 = Bn254::mul_013_by_013(&l0, &l1);
        let io0 = [l0, l1]
            .into_iter()
            .flat_map(|fp2| fp2.into_iter())
            .chain(r0)
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        // Test mul_by_01234
        let x = [c.x, c.y, b.x, b.y, a.x];
        let r1 = Bn254::mul_by_01234(&f, &x);
        let io1 = iter::empty()
            .chain(f.to_coeffs())
            .chain(x)
            .chain(r1.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bn254_miller_step() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_miller_step",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(20);
        let S = G2Affine::random(&mut rng);
        let Q = G2Affine::random(&mut rng);

        let s = AffinePoint::new(S.x, S.y);
        let q = AffinePoint::new(Q.x, Q.y);

        // Test miller_double_step
        let (pt, l) = Bn254::miller_double_step(&s);
        let io0 = [s.x, s.y, pt.x, pt.y, l.b, l.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        // Test miller_double_and_add_step
        let (pt, l0, l1) = Bn254::miller_double_and_add_step(&s, &q);
        let io1 = [s.x, s.y, q.x, q.y, pt.x, pt.y, l0.b, l0.c, l1.b, l1.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bn254_miller_loop() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_miller_loop",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [S * Fr::from(1), S * Fr::from(2)];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [Q * Fr::from(2), Q * Fr::from(1)];

        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Test miller_loop
        let f = Bn254::multi_miller_loop(&s, &q);
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .chain(f.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bn254_pairing_check() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_check",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [
            G1Affine::from(S * Fr::from(1)),
            G1Affine::from(S * Fr::from(2)),
        ];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [
            G2Affine::from(Q * Fr::from(2)),
            G2Affine::from(Q * Fr::from(1)),
        ];

        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Gather inputs
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bn254_pairing_check_fallback() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_check_fallback",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [
            G1Affine::from(S * Fr::from(1)),
            G1Affine::from(S * Fr::from(2)),
        ];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [
            G2Affine::from(Q * Fr::from(2)),
            G2Affine::from(Q * Fr::from(1)),
        ];

        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Gather inputs
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();
        // Don't run debugger because it's slow
        air_test_impl::<Engine, _>(
            SystemParams::new_for_testing(22),
            Rv64PairingBuilder,
            get_testing_config(),
            openvm_exe,
            vec![io_all],
            1,
            false,
        )?;
        Ok(())
    }

    #[test]
    fn test_bn254_final_exp_hint() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bn_final_exp_hint",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let P = G1Affine::generator();
        let Q = G2Affine::generator();
        let ps = vec![AffinePoint::new(P.x, P.y), AffinePoint::new(P.x, -P.y)];
        let qs = vec![AffinePoint::new(Q.x, Q.y), AffinePoint::new(Q.x, Q.y)];
        let f = Bn254::multi_miller_loop(&ps, &qs);
        let (c, s) = Bn254::final_exp_hint(&f);
        let ps = ps
            .into_iter()
            .map(|pt| {
                let [x, y] = [pt.x, pt.y]
                    .map(|x| openvm_pairing::bn254::Fp::from_le_bytes_unchecked(&x.to_bytes()));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let qs = qs
            .into_iter()
            .map(|pt| {
                let [x, y] =
                    [pt.x, pt.y].map(|x| openvm_pairing::bn254::Fp2::from_bytes(&x.to_bytes()));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let [c, s] = [c, s].map(|x| openvm_pairing::bn254::Fp12::from_bytes(&x.to_bytes()));
        let io = (ps, qs, (c, s));
        let io = openvm::serde::to_vec(&io).unwrap();
        let io = io.into_iter().flat_map(|w| w.to_le_bytes()).collect();
        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io], 1);
        Ok(())
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    #[cfg(feature = "rvr")]
    use std::{
        collections::{BTreeMap, BTreeSet},
        ffi::OsString,
        sync::Mutex,
    };

    use eyre::Result;
    use halo2curves_axiom::{
        bls12_381::{Fq12, Fq2, Fr, G1Affine, G2Affine},
        ff::Field,
    };
    use num_bigint::BigUint;
    use num_traits::{self, FromPrimitive};
    use openvm_algebra_circuit::{Fp2Extension, Rv64ModularConfig};
    #[cfg(feature = "rvr")]
    use openvm_algebra_transpiler::{Fp2Opcode, Rv64ModularArithmeticOpcode};
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::{
        arch::instructions::exe::VmExe,
        utils::{
            air_test, air_test_impl, air_test_with_min_segments, test_system_config,
            TestStarkEngine as Engine,
        },
    };
    #[cfg(feature = "rvr")]
    use openvm_circuit::{
        arch::{
            rvr::preflight::RvrArenaNativeTarget,
            rvr::{
                generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightOutput,
                RvrPreflightRoute, VmRvrLogNativeExtension,
            },
            MatrixRecordArena, Streams, VirtualMachine,
        },
        system::SystemRecords,
        utils::test_cpu_engine,
    };
    use openvm_ecc_circuit::{
        CurveConfig, Rv64WeierstrassBuilder, Rv64WeierstrassConfig, WeierstrassExtension,
    };
    use openvm_ecc_guest::{
        algebra::{field::FieldExtension, IntMod},
        AffinePoint,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    #[cfg(feature = "rvr")]
    use openvm_instructions::{LocalOpcode, SystemOpcode};
    #[cfg(feature = "rvr")]
    use openvm_pairing_circuit::Rv64PairingCpuBuilder;
    use openvm_pairing_circuit::{
        PairingCurve, PairingExtension, Rv64PairingBuilder, Rv64PairingConfig,
    };
    use openvm_pairing_guest::{
        bls12_381::{
            BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_ECC_STRUCT_NAME, BLS12_381_MODULUS,
            BLS12_381_ORDER,
        },
        halo2curves_shims::bls12_381::Bls12_381,
        pairing::{EvaluatedLine, FinalExp, LineMulMType, MillerStep, MultiMillerLoop},
    };
    #[cfg(feature = "rvr")]
    use openvm_pairing_transpiler::PairingPhantom;
    use openvm_pairing_transpiler::PairingTranspilerExtension;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    #[cfg(feature = "rvr")]
    use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeField32;
    use openvm_stark_sdk::{openvm_stark_backend::SystemParams, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use rand08::SeedableRng;

    type F = BabyBear;

    #[cfg(feature = "rvr")]
    static RVR_COMPILE_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[cfg(feature = "rvr")]
    struct ArenaNativeEnvGuard(Option<OsString>);

    #[cfg(feature = "rvr")]
    impl ArenaNativeEnvGuard {
        fn disable() -> Self {
            let previous = std::env::var_os("OPENVM_RVR_ARENA_NATIVE");
            std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
            Self(previous)
        }
    }

    #[cfg(feature = "rvr")]
    impl Drop for ArenaNativeEnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                std::env::set_var("OPENVM_RVR_ARENA_NATIVE", previous);
            } else {
                std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
            }
        }
    }

    #[cfg(test)]
    pub fn get_testing_config() -> Rv64PairingConfig {
        let primes = [BLS12_381_MODULUS.clone()];
        let complex_struct_names = [BLS12_381_COMPLEX_STRUCT_NAME.to_string()];
        let primes_with_names = complex_struct_names
            .into_iter()
            .zip(primes.clone())
            .collect::<Vec<_>>();
        Rv64PairingConfig {
            modular: Rv64ModularConfig::new(primes.to_vec()),
            fp2: Fp2Extension::new(primes_with_names),
            weierstrass: WeierstrassExtension::new(vec![]),
            pairing: PairingExtension::new(vec![PairingCurve::Bls12_381]),
        }
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
    fn assert_pairing_timestamp_deltas(
        exe: &VmExe<F>,
        output: &RvrPreflightOutput<F>,
        pc_to_air_idx: &[Option<usize>],
        saw_modular_48: &mut bool,
        saw_fp2_48: &mut bool,
    ) {
        let modular_count = Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize + 1;
        let fp2_count = Fp2Opcode::SETUP_MULDIV as usize + 1;
        for (idx, entry) in output.raw_logs.program_log.iter().enumerate() {
            let pc = entry.pc();
            let instruction_idx = ((pc - exe.program.pc_base) / 4) as usize;
            let Some((instruction, _)) = &exe.program.instructions_and_debug_infos[instruction_idx]
            else {
                continue;
            };
            let opcode = instruction.opcode.as_usize();
            let expected_delta = if (Rv64ModularArithmeticOpcode::CLASS_OFFSET
                ..Rv64ModularArithmeticOpcode::CLASS_OFFSET + modular_count)
                .contains(&opcode)
            {
                *saw_modular_48 = true;
                let local = opcode - Rv64ModularArithmeticOpcode::CLASS_OFFSET;
                if matches!(
                    local,
                    x if x == Rv64ModularArithmeticOpcode::IS_EQ as usize
                        || x == Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize
                ) {
                    2 + 2 * 6 + 1
                } else {
                    2 + 1 + 2 * 6 + 6
                }
            } else if (Fp2Opcode::CLASS_OFFSET..Fp2Opcode::CLASS_OFFSET + fp2_count)
                .contains(&opcode)
            {
                *saw_fp2_48 = true;
                2 + 1 + 2 * 12 + 12
            } else {
                continue;
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
                "pairing arithmetic opcode {opcode:#x} at pc {pc:#x} timestamp delta"
            );
        }

        // Arena-native instructions intentionally bypass the compact program
        // log. Their complete records, including adapter timestamps, are
        // compared byte-for-byte with the interpreter trace below. Count an
        // opcode family as exercised when its mapped direct arena wrote rows.
        for (slot, entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
            let Some((instruction, _)) = entry else {
                continue;
            };
            let Some(air) = pc_to_air_idx[slot] else {
                continue;
            };
            let direct_rows = output
                .arena_native_written
                .iter()
                .find(|&&(written_air, _)| written_air == air)
                .map(|&(_, rows)| rows)
                .unwrap_or_default();
            if direct_rows == 0 {
                continue;
            }
            let opcode = instruction.opcode.as_usize();
            if (Rv64ModularArithmeticOpcode::CLASS_OFFSET
                ..Rv64ModularArithmeticOpcode::CLASS_OFFSET + modular_count)
                .contains(&opcode)
            {
                *saw_modular_48 = true;
            } else if (Fp2Opcode::CLASS_OFFSET..Fp2Opcode::CLASS_OFFSET + fp2_count)
                .contains(&opcode)
            {
                *saw_fp2_48 = true;
            }
        }
    }

    #[cfg(feature = "rvr")]
    fn assert_bls12_381_rvr_differential(
        label: &str,
        exe: &VmExe<F>,
        config: &Rv64PairingConfig,
        input: Vec<u8>,
        segments: Vec<(Option<u64>, Vec<u32>)>,
        require_modular_48: bool,
    ) {
        let (mut interp_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64PairingCpuBuilder,
            config.clone(),
        )
        .expect("interpreter vm init");
        let (mut rvr_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64PairingCpuBuilder,
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
        let mut state = interp_vm.create_initial_state(exe, Streams::new(vec![input]));
        let segments = if segments.is_empty() {
            vec![(None, vec![32768; rvr_vm.num_airs()])]
        } else {
            segments
        };
        let mut saw_modular_48 = false;
        let mut saw_fp2_48 = false;

        let segment_outputs = {
            let route = rvr_vm
                .preflight_routed_instance(exe)
                .expect("routed preflight instance");
            let RvrPreflightRoute::Rvr(instance) = route else {
                panic!("{label}: pairing program must route to RVR preflight");
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
                let retired_instructions = interp_output
                    .system_records
                    .filtered_exec_frequencies
                    .iter()
                    .map(|&count| u64::from(count))
                    .sum::<u64>();
                if let Some(expected) = num_insns {
                    assert_eq!(
                        retired_instructions, expected,
                        "{segment_label}: interpreter retired instruction count"
                    );
                }
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
                        Some(retired_instructions),
                        &trace_heights,
                        &targets,
                    )
                    .expect("rvr preflight execution");
                assert_system_records_eq(
                    &segment_label,
                    &interp_output.system_records,
                    &rvr_output.system_records,
                );
                assert_pairing_timestamp_deltas(
                    exe,
                    &rvr_output,
                    &pc_to_air_idx,
                    &mut saw_modular_48,
                    &mut saw_fp2_48,
                );
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
        let mut active_pairing_airs = BTreeSet::new();

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

            let is_pairing_air = |air_idx: usize| {
                air_names[air_idx].contains("FieldExpressionCoreAir")
                    || air_names[air_idx].contains("ModularIsEqualCoreAir")
            };
            let mut interp_traces = interp_ctx.per_trace.into_iter().collect::<BTreeMap<_, _>>();
            let mut rvr_traces = rvr_ctx.per_trace.into_iter().collect::<BTreeMap<_, _>>();
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
                    interp_trace.public_values, rvr_trace.public_values,
                    "{segment_label}: {air_name} public values"
                );
                if !is_pairing_air(air_idx) {
                    // HARD-5: shared lookup/periphery rows can be emitted in different DashMap
                    // iteration order. Their row bytes are deliberately excluded; SystemRecords
                    // and these order-independent public values remain the equality oracle.
                    continue;
                }
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
                active_pairing_airs.insert(air_idx);
            }
        }

        if require_modular_48 {
            assert!(
                saw_modular_48,
                "{label}: BLS12-381 must exercise the 48-byte/BLOCKS=6 VecHeap path"
            );
        }
        assert!(
            saw_fp2_48,
            "{label}: BLS12-381 Miller loop must exercise the 96-byte/BLOCKS=12 Fp2 path"
        );
        assert!(
            active_pairing_airs.len() >= if require_modular_48 { 4 } else { 2 },
            "{label}: required deterministic modular/Fp2 traces must all be active"
        );
    }

    #[cfg(feature = "rvr")]
    fn build_bls12_381_miller_case(config: &Rv64PairingConfig) -> Result<(VmExe<F>, Vec<u8>)> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "rvr_pairing_miller_loop",
            ["bls12_381"],
            config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let generator_g1 = G1Affine::generator();
        let generator_g2 = G2Affine::generator();
        let p = AffinePoint::new(generator_g1.x, generator_g1.y);
        let q = AffinePoint::new(generator_g2.x, generator_g2.y);
        let f = Bls12_381::multi_miller_loop(std::slice::from_ref(&p), std::slice::from_ref(&q));
        let base_sum = p.x + p.y;
        let input = [p.x, p.y]
            .into_iter()
            .flat_map(|element| element.to_bytes())
            .chain(
                [q.x, q.y]
                    .into_iter()
                    .chain(f.to_coeffs())
                    .flat_map(|fp2| fp2.to_coeffs())
                    .flat_map(|element| element.to_bytes()),
            )
            .chain(base_sum.to_bytes())
            .collect();
        Ok((exe, input))
    }

    #[cfg(feature = "rvr")]
    fn build_bls12_381_miller_step_case(
        config: &Rv64PairingConfig,
    ) -> Result<(VmExe<F>, Vec<u8>)> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_miller_step",
            ["bls12_381"],
            config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(88);
        let point_s = G2Affine::random(&mut rng);
        let point_q = G2Affine::random(&mut rng);
        let s = AffinePoint::new(point_s.x, point_s.y);
        let q = AffinePoint::new(point_q.x, point_q.y);
        let (double_point, double_line) = Bls12_381::miller_double_step(&s);
        let io0 = [
            s.x,
            s.y,
            double_point.x,
            double_point.y,
            double_line.b,
            double_line.c,
        ]
        .into_iter()
        .flat_map(|element| element.to_bytes());
        let (add_point, line0, line1) = Bls12_381::miller_double_and_add_step(&s, &q);
        let io1 = [
            s.x,
            s.y,
            q.x,
            q.y,
            add_point.x,
            add_point.y,
            line0.b,
            line0.c,
            line1.b,
            line1.c,
        ]
        .into_iter()
        .flat_map(|element| element.to_bytes());
        Ok((exe, io0.chain(io1).collect()))
    }

    #[cfg(test)]
    fn test_rv64weierstrass_config(curves: Vec<CurveConfig>) -> Rv64WeierstrassConfig {
        let mut config = Rv64WeierstrassConfig::new(curves);
        *config.as_mut() = test_system_config();
        config
    }

    #[test]
    fn test_bls_ec() -> Result<()> {
        let curve = CurveConfig {
            struct_name: BLS12_381_ECC_STRUCT_NAME.to_string(),
            modulus: BLS12_381_MODULUS.clone(),
            scalar: BLS12_381_ORDER.clone(),
            a: BigUint::ZERO,
            b: BigUint::from_u8(4).unwrap(),
        };
        let mut config = test_rv64weierstrass_config(vec![curve]);
        *config.as_mut() = test_system_config().with_public_values_bytes(32);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bls_ec",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_bls12_381_fp12_mul() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "fp12_mul",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(50);
        let f0 = Fq12::random(&mut rng);
        let f1 = Fq12::random(&mut rng);
        let r = f0 * f1;

        let io = [f0, f1, r]
            .into_iter()
            .flat_map(|fp12| fp12.to_coeffs())
            .flat_map(|fp2| fp2.to_bytes())
            .collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io], 1);
        Ok(())
    }

    #[test]
    fn test_bls12_381_line_functions() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_line",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(5);
        let a = G2Affine::random(&mut rng);
        let b = G2Affine::random(&mut rng);
        let c = G2Affine::random(&mut rng);

        let f = Fq12::random(&mut rng);
        let l0 = EvaluatedLine::<Fq2> { b: a.x, c: a.y };
        let l1 = EvaluatedLine::<Fq2> { b: b.x, c: b.y };

        // Test mul_023_by_023
        let r0 = Bls12_381::mul_023_by_023(&l0, &l1);
        let io0 = [l0, l1]
            .into_iter()
            .flat_map(|fp2| fp2.into_iter())
            .chain(r0)
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        // Test mul_by_02345
        let x = [c.x, c.y, b.x, b.y, a.x];
        let r1 = Bls12_381::mul_by_02345(&f, &x);
        let io1 = f
            .to_coeffs()
            .into_iter()
            .chain(x)
            .chain(r1.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bls12_381_miller_step() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_miller_step",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand08::rngs::StdRng::seed_from_u64(88);
        let S = G2Affine::random(&mut rng);
        let Q = G2Affine::random(&mut rng);

        let s = AffinePoint::new(S.x, S.y);
        let q = AffinePoint::new(Q.x, Q.y);

        // Test miller_double_step
        let (pt, l) = Bls12_381::miller_double_step(&s);
        let io0 = [s.x, s.y, pt.x, pt.y, l.b, l.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        // Test miller_double_and_add_step
        let (pt, l0, l1) = Bls12_381::miller_double_and_add_step(&s, &q);
        let io1 = [s.x, s.y, q.x, q.y, pt.x, pt.y, l0.b, l0.c, l1.b, l1.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[test]
    fn test_bls12_381_miller_loop() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_miller_loop",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [
            G1Affine::from(S * Fr::from(1)),
            G1Affine::from(S * Fr::from(2)),
        ];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [
            G2Affine::from(Q * Fr::from(2)),
            G2Affine::from(Q * Fr::from(1)),
        ];

        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Test miller_loop
        let f = Bls12_381::multi_miller_loop(&s, &q);
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .chain(f.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_bls12_381_rvr_miller_loop_differential() -> Result<()> {
        let _compile_env_lock = RVR_COMPILE_ENV_LOCK
            .lock()
            .expect("RVR compile environment lock poisoned");
        let mut config = get_testing_config();
        *config.as_mut() = test_system_config();
        config.as_mut().segmentation_max_memory = 512 * 1024 * 1024;
        let (step_exe, step_input) = build_bls12_381_miller_step_case(&config)?;

        assert_bls12_381_rvr_differential(
            "bls12_381_miller_step_single",
            &step_exe,
            &config,
            step_input,
            Vec::new(),
            false,
        );

        // HARD-4: the aggregate one-pair Miller program produces more than 65,536 records in at
        // least one deterministic arithmetic chip, so it cannot be replayed as one fixed-capacity
        // segment. This is not a basic-block overflow: the metered interpreter below successfully
        // cuts the program at instruction boundaries and every resulting segment fits. Keep this
        // full-loop fixture multi-segment, while the step fixture above provides the single-segment
        // byte-for-byte trace oracle.
        let (exe, input) = build_bls12_381_miller_case(&config)?;
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64PairingCpuBuilder,
            config.clone(),
        )
        .expect("metered vm init");
        let metered_ctx = vm.build_metered_ctx(&exe);
        let metered = vm.metered_interpreter(&exe).expect("rvr metered instance");
        let (segments, _) = metered
            .execute_metered(Streams::new(vec![input.clone()]), metered_ctx)
            .expect("rvr metered execution");
        assert!(
            segments.len() > 1,
            "tight memory limit must force multiple pairing segments"
        );
        assert!(
            segments.len() <= 16,
            "pairing differential fixture must keep replay cost bounded"
        );
        let segments = segments
            .into_iter()
            .map(|segment| (Some(segment.num_insns), segment.trace_heights))
            .collect();
        assert_bls12_381_rvr_differential(
            "bls12_381_miller_multi",
            &exe,
            &config,
            input,
            segments,
            true,
        );
        Ok(())
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn test_bls12_381_rvr_final_exp_hint_system_records() -> Result<()> {
        let _compile_env_lock = RVR_COMPILE_ENV_LOCK
            .lock()
            .expect("RVR compile environment lock poisoned");
        let mut config = get_testing_config();
        *config.as_mut() = test_system_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bls_final_exp_hint",
            ["bls12_381"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let generator_g1 = G1Affine::generator();
        let generator_g2 = G2Affine::generator();
        let ps = vec![
            AffinePoint::new(generator_g1.x, generator_g1.y),
            AffinePoint::new(generator_g1.x, -generator_g1.y),
        ];
        let qs = vec![
            AffinePoint::new(generator_g2.x, generator_g2.y),
            AffinePoint::new(generator_g2.x, generator_g2.y),
        ];
        let f = Bls12_381::multi_miller_loop(&ps, &qs);
        let expected = Bls12_381::final_exp_hint(&f);
        let ps = ps
            .into_iter()
            .map(|point| {
                let [x, y] = [point.x, point.y].map(|element| {
                    openvm_pairing::bls12_381::Fp::from_le_bytes_unchecked(&element.to_bytes())
                });
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let qs = qs
            .into_iter()
            .map(|point| {
                let [x, y] = [point.x, point.y]
                    .map(|element| openvm_pairing::bls12_381::Fp2::from_bytes(&element.to_bytes()));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let expected = [expected.0, expected.1]
            .map(|element| openvm_pairing::bls12_381::Fp12::from_bytes(&element.to_bytes()));
        let input = openvm::serde::to_vec(&(ps, qs, (expected[0].clone(), expected[1].clone())))?
            .into_iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();

        // This fixture inspects the compact program and memory logs directly.
        // Arena-native custom-family emission intentionally bypasses those log
        // entries, so compile the target-less diagnostic route explicitly.
        let _arena_native_env = ArenaNativeEnvGuard::disable();
        let (interp_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64PairingCpuBuilder,
            config.clone(),
        )?;
        let (rvr_vm, _) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64PairingCpuBuilder, config)?;
        let mut interpreter = interp_vm.preflight_interpreter(&exe)?;
        let state = interp_vm.create_initial_state(&exe, Streams::new(vec![input]));
        let trace_heights = vec![32768; interp_vm.num_airs()];
        let interp_output =
            interp_vm.execute_preflight(&mut interpreter, state.clone(), None, &trace_heights)?;
        let route = rvr_vm.preflight_routed_instance(&exe)?;
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("BLS12-381 final-exp hint must route to RVR preflight");
        };
        let rvr_output = instance.execute_preflight_from_state(state, None)?;
        assert_system_records_eq(
            "bls12_381_final_exp_hint",
            &interp_output.system_records,
            &rvr_output.system_records,
        );

        let hint_entry_idx = rvr_output
            .raw_logs
            .program_log
            .iter()
            .position(|entry| {
                let pc = entry.pc();
                let instruction_idx = ((pc - exe.program.pc_base) / 4) as usize;
                let Some((instruction, _)) =
                    &exe.program.instructions_and_debug_infos[instruction_idx]
                else {
                    return false;
                };
                instruction.opcode == SystemOpcode::PHANTOM.global_opcode()
                    && instruction.c.as_canonical_u32() as u16
                        == PairingPhantom::HintFinalExp as u16
            })
            .expect("pairing hint phantom program-log entry");
        let hint_entry = &rvr_output.raw_logs.program_log[hint_entry_idx];
        let next_timestamp = rvr_output.raw_logs.program_log[hint_entry_idx + 1].timestamp;
        assert_eq!(
            next_timestamp - hint_entry.timestamp,
            1,
            "pairing hint phantom must consume exactly one timestamp"
        );
        assert!(
            rvr_output
                .raw_logs
                .memory_log
                .iter()
                .all(|entry| entry.timestamp != hint_entry.timestamp),
            "pairing hint operands are interpreter-untraced and must emit no preflight memory log"
        );
        Ok(())
    }

    #[test]
    fn test_bls12_381_pairing_check() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_check",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [
            G1Affine::from(S * Fr::from(1)),
            G1Affine::from(S * Fr::from(2)),
        ];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [
            G2Affine::from(Q * Fr::from(2)),
            G2Affine::from(Q * Fr::from(1)),
        ];
        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Gather inputs
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io_all], 1);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_bls12_381_pairing_check_fallback() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "pairing_check_fallback",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let S = G1Affine::generator();
        let Q = G2Affine::generator();

        let mut S_mul = [
            G1Affine::from(S * Fr::from(1)),
            G1Affine::from(S * Fr::from(2)),
        ];
        S_mul[1].y = -S_mul[1].y;
        let Q_mul = [
            G2Affine::from(Q * Fr::from(2)),
            G2Affine::from(Q * Fr::from(1)),
        ];
        let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
        let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

        // Gather inputs
        let io0 = s
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter().flat_map(|fp| fp.to_bytes()))
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();
        // Don't run debugger because it's slow
        air_test_impl::<Engine, _>(
            SystemParams::new_for_testing(22),
            Rv64PairingBuilder,
            get_testing_config(),
            openvm_exe,
            vec![io_all],
            1,
            false,
        )?;
        Ok(())
    }

    #[test]
    fn test_bls12_381_final_exp_hint() -> Result<()> {
        let config = get_testing_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bls_final_exp_hint",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let P = G1Affine::generator();
        let Q = G2Affine::generator();
        let ps = vec![AffinePoint::new(P.x, P.y), AffinePoint::new(P.x, -P.y)];
        let qs = vec![AffinePoint::new(Q.x, Q.y), AffinePoint::new(Q.x, Q.y)];
        let f = Bls12_381::multi_miller_loop(&ps, &qs);
        let (c, s) = Bls12_381::final_exp_hint(&f);
        let ps = ps
            .into_iter()
            .map(|pt| {
                let [x, y] = [pt.x, pt.y]
                    .map(|x| openvm_pairing::bls12_381::Fp::from_le_bytes_unchecked(&x.to_bytes()));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let qs = qs
            .into_iter()
            .map(|pt| {
                let [x, y] =
                    [pt.x, pt.y].map(|x| openvm_pairing::bls12_381::Fp2::from_bytes(&x.to_bytes()));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        let [c, s] = [c, s].map(|x| openvm_pairing::bls12_381::Fp12::from_bytes(&x.to_bytes()));
        let io = (ps, qs, (c, s));
        let io = openvm::serde::to_vec(&io).unwrap();
        let io = io.into_iter().flat_map(|w| w.to_le_bytes()).collect();
        air_test_with_min_segments(Rv64PairingBuilder, config, openvm_exe, vec![io], 1);
        Ok(())
    }
}
