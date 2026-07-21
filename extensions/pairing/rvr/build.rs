// Build the pairing FFI staticlib alongside this crate so callers don't have
// to. The path to the resulting `librvr_openvm_ext_pairing_ffi.a` is exposed
// to the source via the `RVR_PAIRING_FFI_STATICLIB` cargo env var.

use std::{env, path::PathBuf};

use rvr_openvm_build::build_rust_staticlib_with_link_args;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let ffi_manifest = manifest_dir.join("ffi/Cargo.toml");
    let ffi_target_dir = out_dir.join("ffi-target");

    let staticlib = build_rust_staticlib_with_link_args(
        &ffi_manifest,
        &ffi_target_dir,
        "librvr_openvm_ext_pairing_ffi.a",
        "rvr-openvm-ext-pairing-ffi",
    );

    println!(
        "cargo:rustc-env=RVR_PAIRING_FFI_STATICLIB={}",
        staticlib.archive_path.display()
    );
    println!(
        "cargo:rustc-env=RVR_PAIRING_FFI_NATIVE_LINK_ARGS={}",
        staticlib.native_link_args.join(" ")
    );
    println!("cargo:rerun-if-changed=ffi/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/src");
    println!("cargo:rerun-if-changed=../../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/src");
}
