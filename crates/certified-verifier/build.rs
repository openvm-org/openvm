// Build script for the vendored Lean Swirl verifier (see README.md).
// Compiles the Lean-generated C sources under csrc/ into a static library
// linked into this crate, using `leanc` from the pinned Lean toolchain.
// `swirl_dump_proof` is an executable, as it is a wire-format test utility.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Lean toolchain the vendored C sources were generated with. The
/// generated C is ABI-coupled to this runtime version; keep in sync with
/// README.md when refreshing the vendored sources.
const LEAN_TOOLCHAIN: &str = "leanprover/lean4:v4.26.0";

fn main() {
    println!("cargo:rerun-if-env-changed=ELAN_HOME");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=src/ffi");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    ensure_pinned_leanc();
    let lean_prefix = lean_prefix();

    let mut sources = Vec::new();
    collect_c_files(&csrc, &mut sources);
    assert!(!sources.is_empty(), "no C sources under {}", csrc.display());

    // Compile the verifier's Lean-generated C link closure, excluding both
    // executable entry points. Flags mirror what lake used to build the
    // verifier in-repo (minus -DLEAN_EXPORTING / -fvisibility, which only
    // matter for shared-library builds).
    sources.retain(|path| {
        path.file_name()
            .is_none_or(|name| name != "SwirlVerifyMain.c")
    });
    let mut objects: Vec<PathBuf> = std::thread::scope(|scope| {
        let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut handles = Vec::new();
        for chunk in sources.chunks(sources.len().div_ceil(jobs)) {
            let csrc = &csrc;
            let out_dir = &out_dir;
            handles.push(scope.spawn(move || {
                let mut objs = Vec::new();
                for src in chunk {
                    let rel = src.strip_prefix(csrc).unwrap();
                    let obj = out_dir.join(format!(
                        "{}.o",
                        rel.with_extension("").to_string_lossy().replace('/', "_")
                    ));
                    run(leanc(&[
                        "-c",
                        "-O3",
                        "-DNDEBUG",
                        "-fwrapv",
                        "-o",
                        obj.to_str().unwrap(),
                        src.to_str().unwrap(),
                    ]));
                    objs.push(obj);
                }
                objs
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    let ffi_src = manifest_dir.join("src/ffi/swirl_verify.c");
    let ffi_obj = out_dir.join("openvm_swirl_verify_ffi.o");
    run(leanc(&[
        "-c",
        "-O3",
        "-DNDEBUG",
        "-fwrapv",
        "-o",
        ffi_obj.to_str().unwrap(),
        ffi_src.to_str().unwrap(),
    ]));
    objects.push(ffi_obj);

    let verifier_lib = out_dir.join("libopenvm_swirl_verifier.a");
    let mut archive = Command::new(lean_prefix.join("bin/llvm-ar"));
    archive.args(["crs", verifier_lib.to_str().unwrap()]);
    archive.args(&objects);
    run(archive);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=openvm_swirl_verifier");

    // These are the static libraries `leanc` uses for an executable link.
    // Most are discarded by the final link, but listing the complete pinned
    // toolchain set keeps this in step with Lean's generated-code ABI.
    println!(
        "cargo:rustc-link-search=native={}",
        lean_prefix.join("lib/lean").display()
    );
    for lib in ["leancpp", "Init", "Std", "Lean", "leanrt", "Lake"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    println!(
        "cargo:rustc-link-search=native={}",
        lean_prefix.join("lib").display()
    );
    for lib in ["gmp", "uv"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    let dump_main = csrc.join("Tools/SwirlDumpProof.c");
    let dump_obj = out_dir.join("dump_Tools_SwirlDumpProof.o");
    run(leanc(&[
        "-c",
        "-O3",
        "-DNDEBUG",
        "-fwrapv",
        "-o",
        dump_obj.to_str().unwrap(),
        dump_main.to_str().unwrap(),
    ]));
    let raw_obj = out_dir.join("Swirl_Protocol_Noninteractive_Wire_Raw.o");
    assert!(
        raw_obj.exists(),
        "swirl_dump_proof needs {}",
        raw_obj.display()
    );
    let dump_bin = out_dir.join("swirl_dump_proof");
    run(leanc(&[
        "-o",
        dump_bin.to_str().unwrap(),
        dump_obj.to_str().unwrap(),
        raw_obj.to_str().unwrap(),
    ]));
}

fn leanc(args: &[&str]) -> Command {
    let mut cmd = Command::new("elan");
    cmd.args(["run", LEAN_TOOLCHAIN, "leanc"]);
    cmd.args(args);
    cmd
}

fn ensure_pinned_leanc() {
    let output = leanc(&["--version"]).output().unwrap_or_else(|e| {
        panic!(
            "building `openvm-certified-verifier` needs Lean toolchain \
             {LEAN_TOOLCHAIN}, but failed to run `elan`: {e}. Install elan \
             (https://github.com/leanprover/elan) and run \
             `elan toolchain install {LEAN_TOOLCHAIN}`."
        )
    });
    assert!(
        output.status.success(),
        "failed to run `leanc` from Lean toolchain {LEAN_TOOLCHAIN}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
}

fn lean_prefix() -> PathBuf {
    let output = Command::new("elan")
        .args(["run", LEAN_TOOLCHAIN, "lean", "--print-prefix"])
        .output()
        .unwrap_or_else(|e| panic!("failed to query Lean toolchain prefix: {e}"));
    assert!(
        output.status.success(),
        "failed to query Lean toolchain prefix: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

fn collect_c_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_c_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "c")
            && path
                .file_name()
                .is_none_or(|name| name != "SwirlDumpProof.c")
        {
            out.push(path);
        }
    }
    out.sort();
}

fn run(mut cmd: Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(status.success(), "command failed ({status}): {cmd:?}");
}
