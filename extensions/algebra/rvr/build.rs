// Build the algebra FFI staticlibs and generate the libsecp256k1 source list
// embedded by the RVR extension.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use rvr_openvm_build::{build_rust_staticlib, default_compiler_command, ensure_clang_compiler};

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    generate_secp256k1_file_list(&manifest_dir, &out_dir);

    let blst_staticlib = build_blst_staticlib(&manifest_dir, &out_dir);

    let modular_staticlib = build_rust_ffi_staticlib(
        &manifest_dir.join("ffi/modular"),
        &out_dir.join("modular-ffi-target"),
        "librvr_openvm_ext_algebra_modular_ffi.a",
        "rvr-openvm-ext-algebra-modular-ffi",
    );
    let fp2_staticlib = build_rust_ffi_staticlib(
        &manifest_dir.join("ffi/fp2"),
        &out_dir.join("fp2-ffi-target"),
        "librvr_openvm_ext_algebra_fp2_ffi.a",
        "rvr-openvm-ext-algebra-fp2-ffi",
    );

    println!(
        "cargo:rustc-env=RVR_ALGEBRA_MODULAR_FFI_STATICLIB={}",
        modular_staticlib.display()
    );
    println!(
        "cargo:rustc-env=RVR_ALGEBRA_FP2_FFI_STATICLIB={}",
        fp2_staticlib.display()
    );
    println!(
        "cargo:rustc-env=RVR_ALGEBRA_BLST_STATICLIB={}",
        blst_staticlib.display()
    );
    println!("cargo:rerun-if-changed=ffi/common/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/common/src/lib.rs");
    println!("cargo:rerun-if-changed=ffi/modular/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/modular/src/lib.rs");
    println!("cargo:rerun-if-changed=ffi/fp2/Cargo.toml");
    println!("cargo:rerun-if-changed=ffi/fp2/src/lib.rs");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../crates/rvr/rvr-openvm-ffi-common/src");
}

fn build_blst_staticlib(manifest_dir: &Path, out_dir: &Path) -> PathBuf {
    let blst = manifest_dir.join("ffi/modular/blst");
    let server = blst.join("src/server.c");
    let assembly = blst.join("build/assembly.S");
    assert!(
        server.exists() && assembly.exists(),
        "blst submodule missing; run `git submodule update --init extensions/algebra/rvr/ffi/modular/blst`"
    );

    let compiler = default_compiler_command();
    ensure_clang_compiler(&compiler).unwrap_or_else(|error| {
        panic!("rvr-openvm-ext-algebra requires clang to build blst: {error}")
    });

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");
    let mut build = cc::Build::new();
    build
        .cargo_metadata(false)
        .compiler(compiler)
        .opt_level(3)
        .pic(true)
        .flag("-fno-builtin")
        .flag("-fintegrated-as")
        .flag_if_supported("-Wno-gcc-install-dir-libstdcxx")
        .flag_if_supported("-Wno-unused-command-line-argument")
        .out_dir(out_dir)
        .file(&server);

    // BLST ships assembly for x86-64 and AArch64 and uses C on other targets.
    if matches!(target_arch.as_str(), "x86_64" | "aarch64") {
        build.file(&assembly);
    } else {
        build.define("__BLST_NO_ASM__", None);
    }
    if target_arch == "x86_64" {
        // Match BLST upstream and avoid costly transitions between AVX and SSE code.
        build.flag("-mno-avx");
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let has_target_feature = |feature| target_features.split(',').any(|item| item == feature);
        let explicit_target_cpu = env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .contains("target-cpu=");
        // Prefer Cargo's target features; inspect the CPU only for native builds.
        if has_target_feature("adx") {
            build.define("__ADX__", None);
        } else if explicit_target_cpu {
            // BLST portable mode supports explicit targets below SSSE3.
            if !has_target_feature("ssse3") {
                build.define("__BLST_PORTABLE__", None);
            }
        } else if env::var("HOST") == env::var("TARGET") {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("adx") {
                build.define("__ADX__", None);
            }
        }
    }

    build.compile("blst");
    println!("cargo:rerun-if-changed={}", blst.display());
    println!("cargo:rerun-if-env-changed=RVR_CC");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CFLAGS");
    let archive = if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        "blst.lib"
    } else {
        "libblst.a"
    };
    out_dir.join(archive)
}

fn generate_secp256k1_file_list(manifest_dir: &Path, out_dir: &Path) {
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

fn build_rust_ffi_staticlib(
    crate_dir: &Path,
    target_dir: &Path,
    lib_name: &str,
    crate_name: &str,
) -> PathBuf {
    build_rust_staticlib(
        &crate_dir.join("Cargo.toml"),
        target_dir,
        lib_name,
        crate_name,
    )
}
