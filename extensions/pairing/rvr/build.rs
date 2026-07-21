// Build the pairing FFI staticlib alongside this crate so callers don't have
// to. The path to the resulting `librvr_openvm_ext_pairing_ffi.a` is exposed
// to the source via the `RVR_PAIRING_FFI_STATICLIB` cargo env var.

use std::{env, path::PathBuf};

use rvr_openvm_build::build_rust_staticlib_with_features;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let ffi_manifest = manifest_dir.join("ffi/Cargo.toml");
    let ffi_target_dir = out_dir.join("ffi-target");

    let features = if supports_halo2curves_asm() {
        &["halo2curves-asm"][..]
    } else {
        &[]
    };
    let lib_path = build_rust_staticlib_with_features(
        &ffi_manifest,
        &ffi_target_dir,
        "librvr_openvm_ext_pairing_ffi.a",
        "rvr-openvm-ext-pairing-ffi",
        features,
    );

    println!(
        "cargo:rustc-env=RVR_PAIRING_FFI_STATICLIB={}",
        lib_path.display()
    );
    println!("cargo:rerun-if-changed=ffi/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/src");
}

fn supports_halo2curves_asm() -> bool {
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return false;
    }

    // Cargo expands `target-cpu` and `target-feature` settings into this list.
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_feature = |required| {
        target_features
            .split(',')
            .any(|feature| feature == required)
    };
    // Halo2curves' x86-64 field implementation uses both instruction sets.
    has_feature("adx") && has_feature("bmi2")
}
