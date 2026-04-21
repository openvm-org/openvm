//! Int256 extension integration tests.

use std::{path::PathBuf, process::Command};

use eyre::Result;
use openvm_bigint_circuit::{Int256Rv32Config, Int256Rv32CpuBuilder};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_transpiler::*;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm_ext_bigint::Int256Extension;

use super::utils::{self, ExecutionMode, F};

// ── Int256-specific helpers ─────────────────────────────────────────────────

fn transpile_with_int256(elf: Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Int256TranspilerExtension),
    )?)
}

fn build_int256_staticlib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ffi_crate = manifest_dir.join("../extensions/bigint/ffi");

    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&ffi_crate)
        .output()
        .expect("failed to run cargo build for int256-ffi extension");

    if !output.status.success() {
        panic!(
            "Failed to build int256-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let workspace_root = manifest_dir.join("../..");
    let lib_path = workspace_root.join("target/release/librvr_openvm_ext_bigint_ffi.a");
    assert!(
        lib_path.exists(),
        "Int256 FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_matrix_power_exe() -> Result<VmExe<F>> {
    let programs_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/programs/ruint");
    let elf = build_example_program_at_path_with_features::<&str>(
        programs_dir,
        "matrix_power",
        [],
        &Int256Rv32Config::default(),
    )?;
    transpile_with_int256(elf)
}

fn make_int256_ext(harness: &utils::VmTestHarness<Int256Rv32CpuBuilder>) -> Int256Extension {
    let ctx = harness.rvr_extension_ctx().unwrap();
    Int256Extension::new(&ctx, build_int256_staticlib()).unwrap()
}

fn run_matrix_power(label: &str, mode: ExecutionMode) -> Result<()> {
    let exe = build_matrix_power_exe()?;
    let mut harness = utils::VmTestHarness::new(Int256Rv32Config::default(), Int256Rv32CpuBuilder)?;
    harness.register(make_int256_ext(&harness));
    harness.compare(label, &exe, vec![], mode)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_int256_matrix_power() -> Result<()> {
    run_matrix_power("int256_matrix_power", ExecutionMode::Pure)
}

#[test]
fn test_int256_metered_cost() -> Result<()> {
    run_matrix_power("int256_metered_cost", ExecutionMode::MeteredCost)
}

#[test]
fn test_int256_metered() -> Result<()> {
    run_matrix_power("int256_metered", ExecutionMode::Metered)
}
