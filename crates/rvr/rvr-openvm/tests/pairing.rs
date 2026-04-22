//! Pairing extension integration tests.
//!
//! Tests pairing guest programs (fp12_mul, pairing_check) against the
//! OpenVM interpreter using the RVR execution pipeline.

#[path = "utils.rs"]
mod utils;

use std::{path::PathBuf, process::Command};

use eyre::Result;
use halo2curves_axiom::{
    bls12_381::{
        Fq12 as BlsFq12, Fq2 as BlsFq2, Fr as BlsFr, G1Affine as BlsG1Affine,
        G2Affine as BlsG2Affine,
    },
    bn256::{Fq12 as BnFq12, Fr as BnFr, G1Affine as BnG1Affine, G2Affine as BnG2Affine},
    ff::Field,
};
use openvm_algebra_circuit::Fp2Extension;
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_ecc_circuit::WeierstrassExtension;
use openvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
use openvm_instructions::exe::VmExe;
use openvm_pairing_circuit::{
    PairingCurve, PairingExtension as OvmPairingExtension, Rv32PairingConfig, Rv32PairingCpuBuilder,
};
use openvm_pairing_guest::{
    bls12_381::{BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_MODULUS},
    bn254::{BN254_COMPLEX_STRUCT_NAME, BN254_MODULUS},
};
use openvm_pairing_transpiler::PairingTranspilerExtension;
use openvm_rv32im_transpiler::*;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use rand08::SeedableRng;
use rvr_openvm_ext_algebra::AlgebraExtension;
use rvr_openvm_ext_pairing::PairingExtension;
use utils::{ExecutionMode, F};

// ── Config ──────────────────────────────────────────────────────────────

fn get_testing_config() -> Rv32PairingConfig {
    let primes = vec![BN254_MODULUS.clone()];
    let complex_struct_names = vec![BN254_COMPLEX_STRUCT_NAME.to_string()];
    let primes_with_names = complex_struct_names
        .into_iter()
        .zip(primes.clone())
        .collect::<Vec<_>>();
    Rv32PairingConfig {
        modular: openvm_algebra_circuit::Rv32ModularConfig::new(primes),
        fp2: Fp2Extension::new(primes_with_names),
        weierstrass: WeierstrassExtension::new(vec![]),
        pairing: OvmPairingExtension::new(vec![PairingCurve::Bn254]),
    }
}

fn get_bls12_381_testing_config() -> Rv32PairingConfig {
    let primes = vec![BLS12_381_MODULUS.clone()];
    let complex_struct_names = vec![BLS12_381_COMPLEX_STRUCT_NAME.to_string()];
    let primes_with_names = complex_struct_names
        .into_iter()
        .zip(primes.clone())
        .collect::<Vec<_>>();
    Rv32PairingConfig {
        modular: openvm_algebra_circuit::Rv32ModularConfig::new(primes),
        fp2: Fp2Extension::new(primes_with_names),
        weierstrass: WeierstrassExtension::new(vec![]),
        pairing: OvmPairingExtension::new(vec![PairingCurve::Bls12_381]),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn pairing_programs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../openvm/guest-libs/pairing/tests/programs")
}

fn transpile_with_pairing(elf: openvm_transpiler::elf::Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(PairingTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?)
}

fn build_algebra_staticlib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let algebra_ffi_crate = manifest_dir.join("../extensions/algebra/ffi");
    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&algebra_ffi_crate)
        .output()
        .expect("failed to run cargo build for algebra-ffi extension");
    if !output.status.success() {
        panic!(
            "Failed to build algebra-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let workspace_root = manifest_dir.join("../..");
    let lib_path = workspace_root.join("target/release/librvr_openvm_ext_algebra_ffi.a");
    assert!(
        lib_path.exists(),
        "Algebra FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_pairing_staticlib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pairing_ffi_crate = manifest_dir.join("../extensions/pairing/ffi");
    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&pairing_ffi_crate)
        .output()
        .expect("failed to run cargo build for pairing-ffi extension");
    if !output.status.success() {
        panic!(
            "Failed to build pairing-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let workspace_root = manifest_dir.join("../..");
    let lib_path = workspace_root.join("target/release/librvr_openvm_ext_pairing_ffi.a");
    assert!(
        lib_path.exists(),
        "Pairing FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn make_algebra_ext(moduli: Vec<num_bigint::BigUint>) -> AlgebraExtension {
    AlgebraExtension::new(build_algebra_staticlib(), moduli.clone(), moduli)
}

fn make_pairing_ext() -> PairingExtension {
    PairingExtension::new(build_pairing_staticlib())
}

// ── fp12_mul test (BN254, no phantom instruction) ───────────────────────

fn build_fp12_mul_exe() -> Result<VmExe<F>> {
    let config = get_testing_config();
    let elf = build_example_program_at_path_with_features(
        pairing_programs_dir(),
        "fp12_mul",
        ["bn254"],
        &config,
    )?;
    transpile_with_pairing(elf)
}

fn build_fp12_mul_input() -> Vec<Vec<F>> {
    let mut rng = rand08::rngs::StdRng::seed_from_u64(2);
    let f0 = BnFq12::random(&mut rng);
    let f1 = BnFq12::random(&mut rng);
    let r = f0 * f1;

    let io: Vec<F> = [f0, f1, r]
        .into_iter()
        .flat_map(|fp12: BnFq12| fp12.to_coeffs())
        .flat_map(|fp2: halo2curves_axiom::bn256::Fq2| fp2.to_bytes())
        .map(F::from_u8)
        .collect();

    vec![io]
}

fn run_fp12_mul(label: &str, mode: ExecutionMode) -> Result<()> {
    let exe = build_fp12_mul_exe()?;
    let input = build_fp12_mul_input();
    let config = get_testing_config();

    let mut harness = utils::VmTestHarness::new(config, Rv32PairingCpuBuilder)?;
    harness.register(make_algebra_ext(vec![BN254_MODULUS.clone()]));
    harness.register(make_pairing_ext());
    harness.compare(label, &exe, input, mode)
}

#[test]
fn test_bn254_fp12_mul() -> Result<()> {
    run_fp12_mul("bn254_fp12_mul", ExecutionMode::Pure)
}

#[test]
fn test_bn254_fp12_mul_metered_cost() -> Result<()> {
    run_fp12_mul("bn254_fp12_mul_metered_cost", ExecutionMode::MeteredCost)
}

#[test]
fn test_bn254_fp12_mul_metered() -> Result<()> {
    run_fp12_mul("bn254_fp12_mul_metered", ExecutionMode::Metered)
}

// ── pairing_check test (BN254, uses HintFinalExp phantom) ───────────────

fn build_pairing_check_exe() -> Result<VmExe<F>> {
    let config = get_testing_config();
    let elf = build_example_program_at_path_with_features(
        pairing_programs_dir(),
        "pairing_check",
        ["bn254"],
        &config,
    )?;
    transpile_with_pairing(elf)
}

fn build_pairing_check_input() -> Vec<Vec<F>> {
    use halo2curves_axiom::bn256::{Fq, Fq2};

    let s_gen = BnG1Affine::generator();
    let q_gen = BnG2Affine::generator();

    // e(1*G1, 2*G2) * e(-(2*G1), 1*G2) = 1
    let mut s_mul = [
        BnG1Affine::from(s_gen * BnFr::from(1)),
        BnG1Affine::from(s_gen * BnFr::from(2)),
    ];
    s_mul[1].y = -s_mul[1].y;
    let q_mul = [
        BnG2Affine::from(q_gen * BnFr::from(2)),
        BnG2Affine::from(q_gen * BnFr::from(1)),
    ];

    let s = s_mul.map(|p| AffinePoint::new(p.x, p.y));
    let q = q_mul.map(|p| AffinePoint::new(p.x, p.y));

    // Serialize G1 points
    let io0: Vec<F> = s
        .into_iter()
        .flat_map(|pt: AffinePoint<Fq>| [pt.x, pt.y].into_iter().flat_map(|fp: Fq| fp.to_bytes()))
        .map(F::from_u8)
        .collect();

    // Serialize G2 points
    let io1: Vec<F> = q
        .into_iter()
        .flat_map(|pt: AffinePoint<Fq2>| [pt.x, pt.y].into_iter())
        .flat_map(|fp2: Fq2| fp2.to_coeffs())
        .flat_map(|fp: Fq| fp.to_bytes())
        .map(F::from_u8)
        .collect();

    let io_all: Vec<F> = io0.into_iter().chain(io1).collect();
    vec![io_all]
}

fn run_pairing_check(label: &str, mode: ExecutionMode) -> Result<()> {
    let exe = build_pairing_check_exe()?;
    let input = build_pairing_check_input();
    let config = get_testing_config();

    let mut harness = utils::VmTestHarness::new(config, Rv32PairingCpuBuilder)?;
    harness.register(make_algebra_ext(vec![BN254_MODULUS.clone()]));
    harness.register(make_pairing_ext());
    harness.compare(label, &exe, input, mode)
}

fn run_bls12_381_fp12_mul(label: &str, mode: ExecutionMode) -> Result<()> {
    let exe = build_bls12_381_fp12_mul_exe()?;
    let input = build_bls12_381_fp12_mul_input();
    let mut harness =
        utils::VmTestHarness::new(get_bls12_381_testing_config(), Rv32PairingCpuBuilder)?;
    harness.register(make_algebra_ext(vec![BLS12_381_MODULUS.clone()]));
    harness.register(make_pairing_ext());
    harness.compare(label, &exe, input, mode)
}

fn run_bls12_381_pairing_check(label: &str, mode: ExecutionMode) -> Result<()> {
    let exe = build_bls12_381_pairing_check_exe()?;
    let input = build_bls12_381_pairing_check_input();
    let mut harness =
        utils::VmTestHarness::new(get_bls12_381_testing_config(), Rv32PairingCpuBuilder)?;
    harness.register(make_algebra_ext(vec![BLS12_381_MODULUS.clone()]));
    harness.register(make_pairing_ext());
    harness.compare(label, &exe, input, mode)
}

#[test]
fn test_bn254_pairing_check() -> Result<()> {
    run_pairing_check("bn254_pairing_check", ExecutionMode::Pure)
}

#[test]
fn test_bn254_pairing_check_metered_cost() -> Result<()> {
    run_pairing_check(
        "bn254_pairing_check_metered_cost",
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_bn254_pairing_check_metered() -> Result<()> {
    run_pairing_check("bn254_pairing_check_metered", ExecutionMode::Metered)
}

fn build_bls12_381_fp12_mul_exe() -> Result<VmExe<F>> {
    let config = get_bls12_381_testing_config();
    let elf = build_example_program_at_path_with_features(
        pairing_programs_dir(),
        "fp12_mul",
        ["bls12_381"],
        &config,
    )?;
    transpile_with_pairing(elf)
}

fn build_bls12_381_fp12_mul_input() -> Vec<Vec<F>> {
    let mut rng = rand08::rngs::StdRng::seed_from_u64(50);
    let f0 = BlsFq12::random(&mut rng);
    let f1 = BlsFq12::random(&mut rng);
    let r = f0 * f1;

    let io: Vec<F> = [f0, f1, r]
        .into_iter()
        .flat_map(|fp12: BlsFq12| fp12.to_coeffs())
        .flat_map(|fp2: BlsFq2| fp2.to_bytes())
        .map(F::from_u8)
        .collect();

    vec![io]
}

fn build_bls12_381_pairing_check_exe() -> Result<VmExe<F>> {
    let config = get_bls12_381_testing_config();
    let elf = build_example_program_at_path_with_features(
        pairing_programs_dir(),
        "pairing_check",
        ["bls12_381"],
        &config,
    )?;
    transpile_with_pairing(elf)
}

fn build_bls12_381_pairing_check_input() -> Vec<Vec<F>> {
    use halo2curves_axiom::bls12_381::{Fq, Fq2};

    let s_gen = BlsG1Affine::generator();
    let q_gen = BlsG2Affine::generator();

    let mut s_mul = [
        BlsG1Affine::from(s_gen * BlsFr::from(1)),
        BlsG1Affine::from(s_gen * BlsFr::from(2)),
    ];
    s_mul[1].y = -s_mul[1].y;
    let q_mul = [
        BlsG2Affine::from(q_gen * BlsFr::from(2)),
        BlsG2Affine::from(q_gen * BlsFr::from(1)),
    ];

    let s = s_mul.map(|p| AffinePoint::new(p.x, p.y));
    let q = q_mul.map(|p| AffinePoint::new(p.x, p.y));

    let io0: Vec<F> = s
        .into_iter()
        .flat_map(|pt: AffinePoint<Fq>| [pt.x, pt.y].into_iter().flat_map(|fp: Fq| fp.to_bytes()))
        .map(F::from_u8)
        .collect();

    let io1: Vec<F> = q
        .into_iter()
        .flat_map(|pt: AffinePoint<Fq2>| [pt.x, pt.y].into_iter())
        .flat_map(|fp2: Fq2| fp2.to_coeffs())
        .flat_map(|fp: Fq| fp.to_bytes())
        .map(F::from_u8)
        .collect();

    vec![io0.into_iter().chain(io1).collect()]
}

#[test]
fn test_bls12_381_fp12_mul() -> Result<()> {
    run_bls12_381_fp12_mul("bls12_381_fp12_mul", ExecutionMode::Pure)
}

#[test]
fn test_bls12_381_fp12_mul_metered_cost() -> Result<()> {
    run_bls12_381_fp12_mul(
        "bls12_381_fp12_mul_metered_cost",
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_bls12_381_fp12_mul_metered() -> Result<()> {
    run_bls12_381_fp12_mul("bls12_381_fp12_mul_metered", ExecutionMode::Metered)
}

#[test]
fn test_bls12_381_pairing_check() -> Result<()> {
    run_bls12_381_pairing_check("bls12_381_pairing_check", ExecutionMode::Pure)
}

#[test]
fn test_bls12_381_pairing_check_metered_cost() -> Result<()> {
    run_bls12_381_pairing_check(
        "bls12_381_pairing_check_metered_cost",
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_bls12_381_pairing_check_metered() -> Result<()> {
    run_bls12_381_pairing_check("bls12_381_pairing_check_metered", ExecutionMode::Metered)
}
