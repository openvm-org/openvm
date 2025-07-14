#![allow(non_snake_case)]

#[cfg(feature = "bn254")]
mod bn254 {
    use std::iter;

    use eyre::Result;
    use halo2curves_axiom::{
        bn256::{Fq12, Fq2, Fr, G1Affine, G2Affine},
        ff::Field,
    };
    use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::{
        arch::SystemConfig,
        utils::{air_test, air_test_impl, air_test_with_min_segments},
    };
    use openvm_ecc_circuit::{Rv32WeierstrassConfig, WeierstrassExtension};
    use openvm_ecc_guest::{
        algebra::{field::FieldExtension, IntMod},
        AffinePoint,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_instructions::exe::VmExe;
    use openvm_pairing_circuit::{PairingCurve, PairingExtension, Rv32PairingConfig};
    use openvm_pairing_guest::{
        bn254::{BN254_COMPLEX_STRUCT_NAME, BN254_MODULUS},
        halo2curves_shims::bn254::Bn254,
        pairing::{EvaluatedLine, FinalExp, LineMulDType, MillerStep, MultiMillerLoop},
    };
    use openvm_pairing_transpiler::PairingTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{openvm_stark_backend::p3_field::FieldAlgebra, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use rand::SeedableRng;

    type F = BabyBear;

    #[cfg(test)]
    pub fn get_testing_config() -> Rv32PairingConfig {
        let primes = [BN254_MODULUS.clone()];
        let complex_struct_names = [BN254_COMPLEX_STRUCT_NAME.to_string()];
        let primes_with_names = complex_struct_names
            .into_iter()
            .zip(primes.clone())
            .collect::<Vec<_>>();
        Rv32PairingConfig {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes.to_vec()),
            fp2: Fp2Extension::new(primes_with_names),
            weierstrass: WeierstrassExtension::new(vec![]),
            pairing: PairingExtension::new(vec![PairingCurve::Bn254]),
        }
    }

    #[test]
    fn test_bn_ec() -> Result<()> {
        let curve = PairingCurve::Bn254.curve_config();
        let config = Rv32WeierstrassConfig::new(vec![curve]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bn_ec",
            ["bn254"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let f0 = Fq12::random(&mut rng);
        let f1 = Fq12::random(&mut rng);
        let r = f0 * f1;

        let io = [f0, f1, r]
            .into_iter()
            .flat_map(|fp12| fp12.to_coeffs())
            .flat_map(|fp2| fp2.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
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
            .map(FieldAlgebra::from_canonical_u8)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(20);
        let S = G2Affine::random(&mut rng);
        let Q = G2Affine::random(&mut rng);

        let s = AffinePoint::new(S.x, S.y);
        let q = AffinePoint::new(Q.x, Q.y);

        // Test miller_double_step
        let (pt, l) = Bn254::miller_double_step(&s);
        let io0 = [s.x, s.y, pt.x, pt.y, l.b, l.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        // Test miller_double_and_add_step
        let (pt, l0, l1) = Bn254::miller_double_and_add_step(&s, &q);
        let io1 = [s.x, s.y, q.x, q.y, pt.x, pt.y, l0.b, l0.c, l1.b, l1.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .chain(f.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();
        // Don't run debugger because it's slow
        air_test_impl(get_testing_config(), openvm_exe, vec![io_all], 1, false);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
        let io = io
            .into_iter()
            .flat_map(|w| w.to_le_bytes())
            .map(F::from_canonical_u8)
            .collect();
        air_test_with_min_segments(config, openvm_exe, vec![io], 1);
        Ok(())
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use eyre::Result;
    use halo2curves_axiom::{
        bls12_381::{Fq12, Fq2, Fr, G1Affine, G2Affine},
        ff::Field,
    };
    use num_bigint::BigUint;
    use num_traits::{self, FromPrimitive};
    use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::{
        arch::{instructions::exe::VmExe, SystemConfig},
        utils::{air_test, air_test_impl, air_test_with_min_segments},
    };
    use openvm_ecc_circuit::{CurveConfig, Rv32WeierstrassConfig, WeierstrassExtension};
    use openvm_ecc_guest::{
        algebra::{field::FieldExtension, IntMod},
        AffinePoint,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_pairing_circuit::{PairingCurve, PairingExtension, Rv32PairingConfig};
    use openvm_pairing_guest::{
        bls12_381::{
            BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_ECC_STRUCT_NAME, BLS12_381_MODULUS,
            BLS12_381_ORDER,
        },
        halo2curves_shims::bls12_381::Bls12_381,
        pairing::{EvaluatedLine, FinalExp, LineMulMType, MillerStep, MultiMillerLoop},
    };
    use openvm_pairing_transpiler::PairingTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{openvm_stark_backend::p3_field::FieldAlgebra, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use rand::SeedableRng;

    type F = BabyBear;

    #[cfg(test)]
    pub fn get_testing_config() -> Rv32PairingConfig {
        let primes = [BLS12_381_MODULUS.clone()];
        let complex_struct_names = [BLS12_381_COMPLEX_STRUCT_NAME.to_string()];
        let primes_with_names = complex_struct_names
            .into_iter()
            .zip(primes.clone())
            .collect::<Vec<_>>();
        Rv32PairingConfig {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes.to_vec()),
            fp2: Fp2Extension::new(primes_with_names),
            weierstrass: WeierstrassExtension::new(vec![]),
            pairing: PairingExtension::new(vec![PairingCurve::Bls12_381]),
        }
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
        let config = Rv32WeierstrassConfig::new(vec![curve]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "bls_ec",
            ["bls12_381"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(50);
        let f0 = Fq12::random(&mut rng);
        let f1 = Fq12::random(&mut rng);
        let r = f0 * f1;

        let io = [f0, f1, r]
            .into_iter()
            .flat_map(|fp12| fp12.to_coeffs())
            .flat_map(|fp2| fp2.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
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
            .map(FieldAlgebra::from_canonical_u8)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(PairingTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Fp2TranspilerExtension),
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(88);
        let S = G2Affine::random(&mut rng);
        let Q = G2Affine::random(&mut rng);

        let s = AffinePoint::new(S.x, S.y);
        let q = AffinePoint::new(Q.x, Q.y);

        // Test miller_double_step
        let (pt, l) = Bls12_381::miller_double_step(&s);
        let io0 = [s.x, s.y, pt.x, pt.y, l.b, l.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        // Test miller_double_and_add_step
        let (pt, l0, l1) = Bls12_381::miller_double_and_add_step(&s, &q);
        let io1 = [s.x, s.y, q.x, q.y, pt.x, pt.y, l0.b, l0.c, l1.b, l1.c]
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .chain(f.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();

        air_test_with_min_segments(config, openvm_exe, vec![io_all], 1);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io1 = q
            .into_iter()
            .flat_map(|pt| [pt.x, pt.y].into_iter())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .map(FieldAlgebra::from_canonical_u8)
            .collect::<Vec<_>>();

        let io_all = io0.into_iter().chain(io1).collect::<Vec<_>>();
        // Don't run debugger because it's slow
        air_test_impl(get_testing_config(), openvm_exe, vec![io_all], 1, false);
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
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
        let io = io
            .into_iter()
            .flat_map(|w| w.to_le_bytes())
            .map(F::from_canonical_u8)
            .collect();
        air_test_with_min_segments(config, openvm_exe, vec![io], 1);
        Ok(())
    }
}
