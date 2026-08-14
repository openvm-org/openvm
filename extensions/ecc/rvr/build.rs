// Build the ECC FFI staticlib alongside this crate so callers don't have
// to. The path to the resulting `librvr_openvm_ext_ecc_ffi.a` is exposed
// to the source via the `RVR_ECC_FFI_STATICLIB` cargo env var. The MCL
// submodule is cmake-built the same way and exposed via `RVR_ECC_MCL_STATICLIB`.

use std::{
    env,
    path::{Path, PathBuf},
};

use rvr_openvm_build::build_rust_staticlib;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let mcl_staticlib = build_mcl_staticlib(&manifest_dir, &out_dir);

    let ffi_manifest = manifest_dir.join("ffi/Cargo.toml");
    let ffi_target_dir = out_dir.join("ffi-target");

    let lib_path = build_rust_staticlib(
        &ffi_manifest,
        &ffi_target_dir,
        "librvr_openvm_ext_ecc_ffi.a",
        "rvr-openvm-ext-ecc-ffi",
    );

    println!(
        "cargo:rustc-env=RVR_ECC_FFI_STATICLIB={}",
        lib_path.display()
    );
    println!(
        "cargo:rustc-env=RVR_ECC_MCL_STATICLIB={}",
        mcl_staticlib.display()
    );
    println!("cargo:rerun-if-changed=ffi/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/src");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/src");
}

fn build_mcl_staticlib(manifest_dir: &Path, out_dir: &Path) -> PathBuf {
    let mcl = manifest_dir.join("ffi/native/mcl");
    assert!(
        mcl.join("CMakeLists.txt").exists(),
        "MCL submodule missing; run `git submodule update --init extensions/ecc/rvr/ffi/native/mcl`"
    );

    let mut config = cmake::Config::new(&mcl);
    config.out_dir(out_dir.join("mcl"));
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        // MCL compiles LLVM IR directly on non-x86 targets and requires clang++.
        config.define("CMAKE_CXX_COMPILER", "clang++");
    }
    let installed = config.build();
    let archive = installed.join("lib/libmcl.a");
    assert!(
        archive.exists(),
        "expected MCL static library at {}",
        archive.display()
    );
    println!("cargo:rerun-if-changed={}", mcl.display());
    archive
}
