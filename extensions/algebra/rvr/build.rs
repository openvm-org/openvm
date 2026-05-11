// Build the modular and fp2 Rust FFI staticlibs (Rust-only) and publish
// their paths via `RVR_ALGEBRA_{MODULAR,FP2}_FFI_STATICLIB`. The modular
// extension's lift-time C source and libsecp256k1 inputs are registered
// in `src/lib.rs`'s `ModularRvrExtension`, not here.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let modular_path = build_subffi(
        &manifest_dir.join("ffi/modular"),
        &out_dir.join("modular-ffi-target"),
        "librvr_openvm_ext_algebra_modular_ffi.a",
        "rvr-openvm-ext-algebra-modular-ffi",
    );
    let fp2_path = build_subffi(
        &manifest_dir.join("ffi/fp2"),
        &out_dir.join("fp2-ffi-target"),
        "librvr_openvm_ext_algebra_fp2_ffi.a",
        "rvr-openvm-ext-algebra-fp2-ffi",
    );

    println!(
        "cargo:rustc-env=RVR_ALGEBRA_MODULAR_FFI_STATICLIB={}",
        modular_path.display()
    );
    println!(
        "cargo:rustc-env=RVR_ALGEBRA_FP2_FFI_STATICLIB={}",
        fp2_path.display()
    );
    println!("cargo:rerun-if-changed=ffi/common/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/common/src/lib.rs");
    println!("cargo:rerun-if-changed=ffi/modular/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/modular/src/lib.rs");
    println!("cargo:rerun-if-changed=ffi/fp2/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/fp2/src/lib.rs");
}

fn build_subffi(crate_dir: &Path, target_dir: &Path, lib_name: &str, crate_name: &str) -> PathBuf {
    let manifest = crate_dir.join("Cargo.toml");

    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(&cargo)
        .args([
            "build",
            "--release",
            "--config",
            "profile.release.lto=false",
            "--manifest-path",
        ])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target_dir)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn cargo for {crate_name}: {e}"));
    assert!(status.success(), "cargo build for {crate_name} failed");

    let lib_path = target_dir.join("release").join(lib_name);
    assert!(
        lib_path.exists(),
        "expected staticlib at {} after cargo build",
        lib_path.display()
    );
    lib_path
}
