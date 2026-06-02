// Build the modular and fp2 Rust FFI staticlibs (Rust-only) and publish
// their paths via `RVR_ALGEBRA_{MODULAR,FP2}_FFI_STATICLIB`. The modular
// extension's lift-time C source and libsecp256k1 inputs are registered
// in `src/lib.rs`'s `ModularRvrExtension`, not here.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use rvr_openvm_build::build_rust_staticlib;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    generate_secp256k1_files(&manifest_dir, &out_dir);

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
    println!("cargo:rerun-if-changed=../../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/src");
}

fn generate_secp256k1_files(manifest_dir: &Path, out_dir: &Path) {
    let root = manifest_dir.join("ffi/modular/secp256k1");
    let mut files = Vec::new();
    collect_c_files(&root.join("src"), &mut files);
    collect_c_files(&root.join("include"), &mut files);
    files.sort();

    let mut generated = String::from("pub(crate) const SECP256K1_C_FILES: &[(&str, &str)] = &[\n");
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&root).expect("secp path outside root");
        let output_name = Path::new("secp256k1").join(relative);
        generated.push_str(&format!("    ({output_name:?}, include_str!({path:?})),\n",));
    }
    generated.push_str("];\n");
    fs::write(out_dir.join("secp256k1_files.rs"), generated)
        .expect("failed to write generated secp256k1 file list");
}

fn collect_c_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read secp256k1 directory {}: {e}", dir.display()))
    {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read secp256k1 dir entry: {e}"))
            .path();
        if path.is_dir() {
            collect_c_files(&path, files);
        } else if should_skip_secp256k1_file(&path) {
            println!("cargo:rerun-if-changed={}", path.display());
        } else if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("c" | "h")
        ) {
            files.push(path);
        }
    }
}

fn should_skip_secp256k1_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    name.starts_with("bench")
        || name.starts_with("tests")
        || name.starts_with("ctime_")
        || name.starts_with("valgrind")
}

fn build_subffi(crate_dir: &Path, target_dir: &Path, lib_name: &str, crate_name: &str) -> PathBuf {
    build_rust_staticlib(
        &crate_dir.join("Cargo.toml"),
        target_dir,
        lib_name,
        crate_name,
    )
}
