//! Keccak-256 extension integration tests.
//!
//! Builds a guest program that uses keccak256, transpiles it with the keccak
//! transpiler extension, compiles with rvr using the keccak rvr extension,
//! executes, and compares against the OpenVM interpreter.

#[path = "utils.rs"]
mod utils;

use eyre::Result;
use openvm_instructions::exe::VmExe;
use openvm_keccak256_circuit::{Keccak256Rv32Config, Keccak256Rv32CpuBuilder};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_transpiler::*;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm_ext_keccak::KeccakExtension;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Command;
use utils::{ExecutionMode, F};

// ── Keccak-specific helpers ─────────────────────────────────────────────────

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

fn build_keccak_input_stream() -> VecDeque<Vec<F>> {
    use tiny_keccak::{Hasher, Keccak};

    let test_inputs: Vec<Vec<u8>> = vec![vec![], vec![0xCC], b"hello world".to_vec()];

    let mut stream = VecDeque::new();
    stream.push_back(encode_u32_bytes(test_inputs.len() as u32));

    for input in &test_inputs {
        let mut hasher = Keccak::v256();
        hasher.update(input);
        let mut expected = [0u8; 32];
        hasher.finalize(&mut expected);

        stream.push_back(encode_vec_u8_serde(input));
        stream.push_back(encode_vec_u8_serde(&expected));
    }

    stream
}

fn keccak_programs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../openvm/guest-libs/keccak256/tests/programs")
}

fn transpile_with_keccak(elf: Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension),
    )?)
}

/// Build the keccak extension staticlib and return its path.
fn build_keccak_staticlib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let keccak_ffi_crate = manifest_dir.join("../extensions/keccak/ffi");

    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&keccak_ffi_crate)
        .output()
        .expect("failed to run cargo build for keccak-ffi extension");

    if !output.status.success() {
        panic!(
            "Failed to build keccak-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let workspace_root = manifest_dir.join("../..");
    let lib_path = workspace_root.join("target/release/librvr_openvm_ext_keccak_ffi.a");
    assert!(
        lib_path.exists(),
        "Keccak FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_keccak_exe() -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features::<&str>(
        keccak_programs_dir(),
        "keccak",
        [],
        &Keccak256Rv32Config::default(),
    )?;
    transpile_with_keccak(elf)
}

fn make_keccak_ext(harness: &utils::VmTestHarness<Keccak256Rv32CpuBuilder>) -> KeccakExtension {
    let ctx = harness.rvr_extension_ctx().unwrap();
    KeccakExtension::new(&ctx, build_keccak_staticlib())
}

// ── Pure execution test ─────────────────────────────────────────────────────

#[test]
fn test_keccak256() -> Result<()> {
    let exe = build_keccak_exe()?;
    let input = build_keccak_input_stream().into_iter().collect::<Vec<_>>();

    let mut harness =
        utils::VmTestHarness::new(Keccak256Rv32Config::default(), Keccak256Rv32CpuBuilder)?;
    let ext = make_keccak_ext(&harness);
    harness.register(ext);

    harness.compare("keccak256", &exe, input, ExecutionMode::Pure)
}

// ── Metered cost execution test ─────────────────────────────────────────────

#[test]
fn test_keccak256_metered_cost() -> Result<()> {
    let exe = build_keccak_exe()?;
    let input = build_keccak_input_stream().into_iter().collect::<Vec<_>>();

    let mut harness =
        utils::VmTestHarness::new(Keccak256Rv32Config::default(), Keccak256Rv32CpuBuilder)?;
    let ext = make_keccak_ext(&harness);
    harness.register(ext);

    harness.compare(
        "keccak256_metered_cost",
        &exe,
        input,
        ExecutionMode::MeteredCost,
    )
}

// ── Metered (segmentation) execution test ───────────────────────────────────

#[test]
fn test_keccak256_metered() -> Result<()> {
    let exe = build_keccak_exe()?;
    let input = build_keccak_input_stream().into_iter().collect::<Vec<_>>();

    let mut harness =
        utils::VmTestHarness::new(Keccak256Rv32Config::default(), Keccak256Rv32CpuBuilder)?;
    let ext = make_keccak_ext(&harness);
    harness.register(ext);

    harness.compare("keccak256_metered", &exe, input, ExecutionMode::Metered)
}
