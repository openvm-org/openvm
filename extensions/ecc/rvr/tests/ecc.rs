#![cfg(feature = "rvr")]

//! ECC extension integration tests.
//!
//! Builds a guest program that uses elliptic curve operations on secp256k1,
//! transpiles it with the ECC + algebra transpiler extensions, compiles with rvr
//! using both the algebra and ECC rvr extensions, executes, and compares against
//! the OpenVM interpreter.

use std::{path::PathBuf, process::Command};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_transpiler::ModularTranspilerExtension;
use openvm_ecc_circuit::{
    CurveConfig, Rv32WeierstrassConfig, Rv32WeierstrassCpuBuilder, P256_CONFIG, SECP256K1_CONFIG,
};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_transpiler::*;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm_ext_algebra::AlgebraExtension;
use rvr_openvm_ext_ecc::EccExtension;
use rvr_openvm_test_utils::{self as utils, workspace_root, ExecutionMode, F};

/// Curve IDs.
const CURVE_K256: u32 = 0;
const CURVE_P256: u32 = 1;

fn ecc_programs_dir() -> PathBuf {
    workspace_root().join("extensions/ecc/tests/programs")
}

fn transpile_with_ecc(elf: Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(EccTranspilerExtension),
    )?)
}

fn build_staticlib(ffi_dir: PathBuf, lib_name: &str) -> PathBuf {
    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&ffi_dir)
        .output()
        .unwrap_or_else(|_| panic!("failed to run cargo build for {}", ffi_dir.display()));
    if !output.status.success() {
        panic!(
            "Failed to build staticlib at {}:\n{}",
            ffi_dir.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let lib_path = workspace_root().join(format!("target/release/{lib_name}"));
    assert!(
        lib_path.exists(),
        "staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_algebra_staticlib() -> PathBuf {
    let ffi_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../algebra/rvr/ffi");
    build_staticlib(ffi_dir, "librvr_openvm_ext_algebra_ffi.a")
}

fn build_ecc_staticlib() -> PathBuf {
    let ffi_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ffi");
    build_staticlib(ffi_dir, "librvr_openvm_ext_ecc_ffi.a")
}

fn build_ecc_config(curves: Vec<CurveConfig>) -> Rv32WeierstrassConfig {
    Rv32WeierstrassConfig::new(curves)
}

fn build_ecc_exe() -> Result<VmExe<F>> {
    let config = build_ecc_config(vec![SECP256K1_CONFIG.clone()]);
    let elf =
        build_example_program_at_path_with_features(ecc_programs_dir(), "ec", ["k256"], &config)?;
    transpile_with_ecc(elf)
}

fn build_ecc_nonzero_a_exe() -> Result<VmExe<F>> {
    let config = build_ecc_config(vec![P256_CONFIG.clone()]);
    let elf = build_example_program_at_path_with_features(
        ecc_programs_dir(),
        "ec_nonzero_a",
        ["p256"],
        &config,
    )?;
    transpile_with_ecc(elf)
}

fn build_ecc_two_curves_exe() -> Result<VmExe<F>> {
    let config = build_ecc_config(vec![SECP256K1_CONFIG.clone(), P256_CONFIG.clone()]);
    let elf = build_example_program_at_path_with_features(
        ecc_programs_dir(),
        "ec_two_curves",
        ["k256", "p256"],
        &config,
    )?;
    transpile_with_ecc(elf)
}

fn curve_moduli(curves: &[CurveConfig]) -> Vec<BigUint> {
    curves
        .iter()
        .flat_map(|curve| [curve.modulus.clone(), curve.scalar.clone()])
        .collect()
}

fn register_extensions(
    harness: &mut utils::VmTestHarness<Rv32WeierstrassCpuBuilder>,
    curve_ids: Vec<u32>,
    moduli: Vec<BigUint>,
) {
    // Algebra extension handles modular arithmetic opcodes
    let algebra_ext = AlgebraExtension::new(build_algebra_staticlib(), moduli, vec![]);
    // ECC extension handles Weierstrass opcodes
    let ecc_ext = EccExtension::new(build_ecc_staticlib(), curve_ids);

    harness.register(algebra_ext);
    harness.register(ecc_ext);
}

// ── Pure execution test ─────────────────────────────────────────────────────

#[test]
fn test_ecc_k256() -> Result<()> {
    let exe = build_ecc_exe()?;
    let curves = vec![SECP256K1_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(&mut harness, vec![CURVE_K256], curve_moduli(&curves));

    harness.compare("ecc_k256", &exe, vec![], ExecutionMode::Pure)
}

// ── Metered cost execution test ─────────────────────────────────────────────

#[test]
fn test_ecc_k256_metered_cost() -> Result<()> {
    let exe = build_ecc_exe()?;
    let curves = vec![SECP256K1_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(&mut harness, vec![CURVE_K256], curve_moduli(&curves));

    harness.compare(
        "ecc_k256_metered_cost",
        &exe,
        vec![],
        ExecutionMode::MeteredCost,
    )
}

// ── Metered (segmentation) execution test ───────────────────────────────────

#[test]
fn test_ecc_k256_metered() -> Result<()> {
    let exe = build_ecc_exe()?;
    let curves = vec![SECP256K1_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(&mut harness, vec![CURVE_K256], curve_moduli(&curves));

    harness.compare("ecc_k256_metered", &exe, vec![], ExecutionMode::Metered)
}

#[test]
fn test_ecc_p256() -> Result<()> {
    let exe = build_ecc_nonzero_a_exe()?;
    let curves = vec![P256_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(&mut harness, vec![CURVE_P256], curve_moduli(&curves));

    harness.compare("ecc_p256", &exe, vec![], ExecutionMode::Pure)
}

#[test]
fn test_ecc_p256_metered() -> Result<()> {
    let exe = build_ecc_nonzero_a_exe()?;
    let curves = vec![P256_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(&mut harness, vec![CURVE_P256], curve_moduli(&curves));

    harness.compare("ecc_p256_metered", &exe, vec![], ExecutionMode::Metered)
}

#[test]
fn test_ecc_two_curves() -> Result<()> {
    let exe = build_ecc_two_curves_exe()?;
    let curves = vec![SECP256K1_CONFIG.clone(), P256_CONFIG.clone()];

    let mut harness =
        utils::VmTestHarness::new(build_ecc_config(curves.clone()), Rv32WeierstrassCpuBuilder)?;
    register_extensions(
        &mut harness,
        vec![CURVE_K256, CURVE_P256],
        curve_moduli(&curves),
    );

    harness.compare("ecc_two_curves", &exe, vec![], ExecutionMode::Pure)
}
