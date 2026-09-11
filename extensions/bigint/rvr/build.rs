// Build the bigint FFI staticlib alongside this crate so callers don't have
// to. The path to the resulting `librvr_openvm_ext_bigint_ffi.a` is exposed
// to the source via the `RVR_BIGINT_FFI_STATICLIB` cargo env var.

use std::{env, path::PathBuf};

use rvr_openvm_build::build_rust_staticlib;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let ffi_manifest = manifest_dir.join("ffi/Cargo.toml");
    // Use a private target dir to avoid lock contention with the outer cargo.
    let ffi_target_dir = out_dir.join("ffi-target");

    let lib_path = build_rust_staticlib(
        &ffi_manifest,
        &ffi_target_dir,
        "librvr_openvm_ext_bigint_ffi.a",
        "rvr-openvm-ext-bigint-ffi",
    );

    println!(
        "cargo:rustc-env=RVR_BIGINT_FFI_STATICLIB={}",
        lib_path.display()
    );
    println!("cargo:rerun-if-changed=ffi/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/src");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/src");
}
