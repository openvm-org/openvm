#![cfg(feature = "rvr")]

//! Algebra extension integration tests.

use std::{path::PathBuf, process::Command, str::FromStr};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_circuit::{
    Rv32ModularConfig, Rv32ModularCpuBuilder, Rv32ModularWithFp2Config,
    Rv32ModularWithFp2CpuBuilder,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_ecc_circuit::SECP256K1_CONFIG;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_transpiler::*;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use rvr_openvm_ext_algebra::AlgebraExtension;
use rvr_openvm_test_utils::{self as utils, workspace_root, ExecutionMode, F};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn algebra_programs_dir() -> PathBuf {
    workspace_root().join("extensions/algebra/tests/programs")
}

fn transpile_with_modular(elf: openvm_transpiler::elf::Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?)
}

fn transpile_with_fp2(elf: openvm_transpiler::elf::Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?)
}

fn build_algebra_staticlib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let algebra_ffi_crate = manifest_dir.join("ffi");
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
    let lib_path = workspace_root().join("target/release/librvr_openvm_ext_algebra_ffi.a");
    assert!(
        lib_path.exists(),
        "Algebra FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn make_modular_ext(moduli: Vec<BigUint>) -> AlgebraExtension {
    AlgebraExtension::new(build_algebra_staticlib(), moduli, vec![])
}

fn make_modular_with_fp2_ext(moduli: Vec<BigUint>, fp2_moduli: Vec<BigUint>) -> AlgebraExtension {
    AlgebraExtension::new(build_algebra_staticlib(), moduli, fp2_moduli)
}

// ── Modular: little.rs ───────────────────────────────────────────────────────

fn build_little_exe(config: &Rv32ModularConfig) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        algebra_programs_dir(),
        "little",
        [],
        config,
    )?;
    transpile_with_modular(elf)
}

#[test]
fn test_modular_little() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_little_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare("modular_little", &exe, vec![], ExecutionMode::Pure)
}

#[test]
fn test_modular_little_metered_cost() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_little_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare(
        "modular_little_metered_cost",
        &exe,
        vec![],
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_modular_little_metered() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_little_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare(
        "modular_little_metered",
        &exe,
        vec![],
        ExecutionMode::Metered,
    )
}

// ── Modular: moduli_setup.rs ─────────────────────────────────────────────────

fn build_moduli_setup_exe(config: &Rv32ModularConfig) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        algebra_programs_dir(),
        "moduli_setup",
        [],
        config,
    )?;
    transpile_with_modular(elf)
}

fn moduli_setup_moduli() -> Vec<BigUint> {
    [
        "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        "1000000000000000003",
        "2305843009213693951",
    ].map(|s| BigUint::from_str(s).unwrap()).to_vec()
}

#[test]
fn test_moduli_setup() -> Result<()> {
    let moduli = moduli_setup_moduli();
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_moduli_setup_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare("moduli_setup", &exe, vec![], ExecutionMode::Pure)
}

#[test]
fn test_moduli_setup_metered_cost() -> Result<()> {
    let moduli = moduli_setup_moduli();
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_moduli_setup_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare(
        "moduli_setup_metered_cost",
        &exe,
        vec![],
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_moduli_setup_metered() -> Result<()> {
    let moduli = moduli_setup_moduli();
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_moduli_setup_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare("moduli_setup_metered", &exe, vec![], ExecutionMode::Metered)
}

// ── Fp2: complex_two_moduli.rs ───────────────────────────────────────────────

fn build_complex_two_moduli_exe(config: &Rv32ModularWithFp2Config) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        algebra_programs_dir(),
        "complex_two_moduli",
        [],
        config,
    )?;
    transpile_with_fp2(elf)
}

fn complex_two_moduli_config() -> (Rv32ModularWithFp2Config, Vec<BigUint>, Vec<BigUint>) {
    let specs = [("Complex1", "998244353"), ("Complex2", "1000000007")];
    let moduli: Vec<BigUint> = specs
        .iter()
        .map(|(_, s)| BigUint::from_str(s).unwrap())
        .collect();
    let fp2_moduli = moduli.clone();
    let moduli_with_names = specs
        .iter()
        .map(|(n, s)| (n.to_string(), BigUint::from_str(s).unwrap()))
        .collect();
    let config = Rv32ModularWithFp2Config::new(moduli_with_names);
    (config, moduli, fp2_moduli)
}

#[test]
fn test_complex_two_moduli() -> Result<()> {
    let (config, moduli, fp2_moduli) = complex_two_moduli_config();
    let exe = build_complex_two_moduli_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularWithFp2CpuBuilder)?;
    let ext = make_modular_with_fp2_ext(moduli, fp2_moduli);
    harness.register(ext);
    harness.compare("complex_two_moduli", &exe, vec![], ExecutionMode::Pure)
}

#[test]
fn test_complex_two_moduli_metered_cost() -> Result<()> {
    let (config, moduli, fp2_moduli) = complex_two_moduli_config();
    let exe = build_complex_two_moduli_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularWithFp2CpuBuilder)?;
    let ext = make_modular_with_fp2_ext(moduli, fp2_moduli);
    harness.register(ext);
    harness.compare(
        "complex_two_moduli_metered_cost",
        &exe,
        vec![],
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_complex_two_moduli_metered() -> Result<()> {
    let (config, moduli, fp2_moduli) = complex_two_moduli_config();
    let exe = build_complex_two_moduli_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularWithFp2CpuBuilder)?;
    let ext = make_modular_with_fp2_ext(moduli, fp2_moduli);
    harness.register(ext);
    harness.compare(
        "complex_two_moduli_metered",
        &exe,
        vec![],
        ExecutionMode::Metered,
    )
}

// ── Sqrt (phantom instructions: HintNonQr + HintSqrt) ───────────────────────

fn build_sqrt_exe(config: &Rv32ModularConfig) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        algebra_programs_dir(),
        "sqrt",
        [],
        config,
    )?;
    transpile_with_modular(elf)
}

#[test]
fn test_sqrt() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_sqrt_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare("sqrt", &exe, vec![], ExecutionMode::Pure)
}

#[test]
fn test_sqrt_metered_cost() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_sqrt_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare(
        "sqrt_metered_cost",
        &exe,
        vec![],
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_sqrt_metered() -> Result<()> {
    let moduli = vec![SECP256K1_CONFIG.modulus.clone()];
    let config = Rv32ModularConfig::new(moduli.clone());
    let exe = build_sqrt_exe(&config)?;
    let mut harness = utils::VmTestHarness::new(config, Rv32ModularCpuBuilder)?;
    let ext = make_modular_ext(moduli);
    harness.register(ext);
    harness.compare("sqrt_metered", &exe, vec![], ExecutionMode::Metered)
}
