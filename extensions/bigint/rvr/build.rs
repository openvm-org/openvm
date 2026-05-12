// Build the bigint FFI staticlib alongside this crate so callers don't have
// to. The path to the resulting `librvr_openvm_ext_bigint_ffi.a` is exposed
// to the source via the `RVR_BIGINT_FFI_STATICLIB` cargo env var.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let ffi_manifest = manifest_dir.join("ffi/Cargo.toml");
    // Use a private target dir to avoid lock contention with the outer cargo.
    let ffi_target_dir = out_dir.join("ffi-target");

    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(&cargo)
        .args([
            "build",
            "--release",
            "--config",
            "profile.release.lto=false",
            "--manifest-path",
        ])
        .arg(&ffi_manifest)
        .arg("--target-dir")
        .arg(&ffi_target_dir)
        .status()
        .expect("failed to spawn cargo for rvr-openvm-ext-bigint-ffi");
    assert!(
        status.success(),
        "cargo build for rvr-openvm-ext-bigint-ffi failed"
    );

    let lib_path = ffi_target_dir.join("release/librvr_openvm_ext_bigint_ffi.a");
    assert!(
        lib_path.exists(),
        "expected staticlib at {} after cargo build",
        lib_path.display()
    );

    println!(
        "cargo:rustc-env=RVR_BIGINT_FFI_STATICLIB={}",
        lib_path.display()
    );
    println!("cargo:rerun-if-changed=ffi/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/src");
}
