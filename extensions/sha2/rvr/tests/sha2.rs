#![cfg(feature = "rvr")]

//! SHA-2 extension integration tests.
//!
//! Builds a guest program that uses sha2, transpiles it with the sha2
//! transpiler extension, compiles with rvr using the sha2 rvr extension,
//! executes, and compares against the OpenVM interpreter.

use std::{path::PathBuf, process::Command};

use eyre::Result;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_transpiler::*;
use openvm_sha2_circuit::{Sha2Rv32Config, Sha2Rv32CpuBuilder};
use openvm_sha2_transpiler::Sha2TranspilerExtension;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm_ext_sha2::Sha2Extension;
use rvr_openvm_test_utils::{self as utils, workspace_root, ExecutionMode, F};
use sha2::{Digest, Sha256, Sha512};

// ── SHA2-specific helpers ─────────────────────────────────────────────────

fn encode_u32_bytes(val: u32) -> Vec<F> {
    val.to_le_bytes().iter().map(|&b| F::from_u8(b)).collect()
}

fn encode_vec_u8_serde(data: &[u8]) -> Vec<F> {
    let mut result = encode_u32_bytes(data.len() as u32);
    for &b in data {
        result.extend(encode_u32_bytes(b as u32));
    }
    result
}

fn build_sha2_input_stream(sha2_type: u32) -> Vec<Vec<F>> {
    let test_inputs: Vec<Vec<u8>> = vec![vec![], vec![0xCC], b"hello world".to_vec()];

    let mut stream = Vec::new();
    stream.push(encode_u32_bytes(sha2_type));
    // Second read(): num_test_vectors
    stream.push(encode_u32_bytes(test_inputs.len() as u32));

    for input in &test_inputs {
        let expected = match sha2_type {
            256 => Sha256::digest(input).to_vec(),
            512 => Sha512::digest(input).to_vec(),
            _ => panic!("unsupported sha2_type: {sha2_type}"),
        };
        stream.push(encode_vec_u8_serde(input));
        stream.push(encode_vec_u8_serde(&expected));
    }

    stream
}

fn build_sha256_input_stream() -> Vec<Vec<F>> {
    build_sha2_input_stream(256)
}

fn build_sha512_input_stream() -> Vec<Vec<F>> {
    build_sha2_input_stream(512)
}

fn sha2_programs_dir() -> PathBuf {
    workspace_root().join("guest-libs/sha2/tests/programs")
}

fn transpile_with_sha2(elf: Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Sha2TranspilerExtension),
    )?)
}

fn build_sha2_staticlib() -> PathBuf {
    let sha2_ffi_crate = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ffi");

    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&sha2_ffi_crate)
        .output()
        .expect("failed to run cargo build for sha2-ffi extension");

    if !output.status.success() {
        panic!(
            "Failed to build sha2-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let lib_path = workspace_root().join("target/release/librvr_openvm_ext_sha2_ffi.a");
    assert!(
        lib_path.exists(),
        "SHA2 FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_sha2_exe() -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        sha2_programs_dir(),
        "sha2",
        [],
        &Sha2Rv32Config::default(),
    )?;
    transpile_with_sha2(elf)
}

fn make_sha2_ext(harness: &utils::VmTestHarness<Sha2Rv32CpuBuilder>) -> Sha2Extension {
    let ctx = harness.rvr_extension_ctx().unwrap();
    Sha2Extension::new(&ctx, build_sha2_staticlib()).unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_sha256() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha256_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare("sha256", &exe, input, ExecutionMode::Pure)
}

#[test]
fn test_sha256_metered_cost() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha256_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare(
        "sha256_metered_cost",
        &exe,
        input,
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_sha256_metered() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha256_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare("sha256_metered", &exe, input, ExecutionMode::Metered)
}

#[test]
fn test_sha512() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha512_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare("sha512", &exe, input, ExecutionMode::Pure)
}

#[test]
fn test_sha512_metered_cost() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha512_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare(
        "sha512_metered_cost",
        &exe,
        input,
        ExecutionMode::MeteredCost,
    )
}

#[test]
fn test_sha512_metered() -> Result<()> {
    let exe = build_sha2_exe()?;
    let input = build_sha512_input_stream();

    let mut harness = utils::VmTestHarness::new(Sha2Rv32Config::default(), Sha2Rv32CpuBuilder)?;
    let ext = make_sha2_ext(&harness);
    harness.register(ext);

    harness.compare("sha512_metered", &exe, input, ExecutionMode::Metered)
}
